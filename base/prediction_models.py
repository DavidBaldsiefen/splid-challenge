import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers, optimizers
import matplotlib.pyplot as plt
import numpy as np


class Prediction_Model():
    def __init__(self, seed=None):
        self._seed=seed
        self._model = None
        self._rnd_gen = np.random.default_rng(self._seed)

    def compile(self, optimizer=keras.optimizers.legacy.Adam(), loss_fn=keras.losses.MeanSquaredError(), metrics=['mse']):
        # TODO: try RMSprop?
        assert(self._model is not None)
        self._model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    def evaluate(self, ds, verbose=1):
        assert(self._model is not None)
        return self._model.evaluate(ds, verbose=verbose)
    
    def fit(self, train_ds, val_ds=None, epochs=100, early_stopping=0, save_best_only=False, best_model_filepath='best_model.hdf5', target_metric='val_accuracy', callbacks=[], plot_hist=False, verbose=1):
        assert(self._model is not None)

        # add callbacks
        if early_stopping > 0:
            callbacks.append(keras.callbacks.EarlyStopping(monitor=target_metric,
                                           mode='auto',
                                           patience=early_stopping,
                                           verbose=verbose))
        if save_best_only:
            callbacks.append(keras.callbacks.ModelCheckpoint(filepath=best_model_filepath,
                                        monitor=target_metric,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        verbose=verbose))
            
        if verbose>0:
            print(f"Starting training. Optimizing \"{target_metric}\"")

        hist = self._model.fit(train_ds, validation_data=val_ds, verbose=verbose, epochs=epochs, callbacks=callbacks)
        self._hist = hist

        if save_best_only:
            self._model.load_weights(best_model_filepath)

        if verbose>0:
            print(f"Finished training after {len(hist.history['loss'])} epochs." + f"Lowest {target_metric}: {np.min(hist.history[target_metric]):.4}  (epoch {np.argmin(hist.history[target_metric]) + 1})" if target_metric in hist.history.keys() else "")

        if plot_hist: self.plot_hist(hist)
        
        return hist
    
    def plot_hist(self, hist, keys=None):
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        for key in (hist.history.keys() if keys is None else keys):
            ax.plot(hist.history[key][1:], label=key)
        ax.legend()
        fig.show()

    def summary(self):
        assert(self.model is not None)
        self._model.summary()

    def createInitializer(self, initializer, mean=None, stddev=None):
        layer_seed = self._rnd_gen.integers(9999999)
        if initializer == 'glorot_uniform':
            return keras.initializers.GlorotUniform(seed=layer_seed)
        elif initializer == 'zeros':
            return keras.initializers.Zeros()
        elif initializer == 'random_normal':
            return keras.initializers.RandomNormal(mean=0.0 if mean is None else mean, stddev=0.05 if stddev is None else stddev, seed=layer_seed)
        elif initializer == 'orthogonal':
            return keras.initializers.Orthogonal(seed=layer_seed)
        else:
            print(f"Warning: unknown initializer \"{initializer}\"")
    
    @property
    def model(self):
        return self._model
    
    @property
    def seed(self):
        return self._seed
    
    @property
    def hist(self):
        return self._hist

class Baseline_NN(Prediction_Model):
    def __init__(self, seed=None):
        "creates a baseline nn that simply propagates inputs to outputs"
        super().__init__(seed)
 
        inputs = layers.Input(shape=(1,))
        self._model = keras.Model(inputs=inputs, outputs=inputs)

        self.compile(metrics=[])
    
    def evaluate(self, ds, ds_gen=None, feature='Log_PCCDYN_SlidingRMS', verbose=1):
        # get ds shapes
        in_shape = ds.element_spec[0].shape.as_list()
        out_shape = ds.element_spec[1].shape.as_list()
        is_batched = in_shape[0] is None
        time_dim = 1 if is_batched else 0
        feat_dim = 2 if is_batched else 1

        if (in_shape[feat_dim]  > 1 or out_shape[feat_dim] > 1) and ds_gen is None:
            print(f"Warning: baseline model evaluated on inputs/outputs with multiple features, but index of \'{feature}\' is unknwon! Try passing ds_gen. ")

        # get the indices of the desired feature
        input_feature_index = 0 if ds_gen is None else ds_gen.input_feature_indices[feature]
        label_feature_index = 0 if ds_gen is None else ds_gen.label_feature_indices[feature]

        # for the baseline, we only use the newest input on the main feature to estimate that feature on the last future label
        def remove_unused_features_and_inputs(inputs, labels):
            new_inputs = inputs[:,in_shape[time_dim]-1:,input_feature_index] if is_batched else  inputs[in_shape[time_dim]-1:,input_feature_index]
            new_labels = labels[:,out_shape[time_dim]-1:,label_feature_index] if is_batched else labels[out_shape[time_dim]-1:,label_feature_index]
            return new_inputs, new_labels
        
        ds = ds.map(remove_unused_features_and_inputs)

        if verbose>0:
            print("Modified ds to use newest input, latest feature label. New shapes:\n", ds.element_spec)

        return super().evaluate(ds, verbose)

class Linear_NN(Prediction_Model):
    def __init__(self, ds, seed=None):
        "Create a linear model, with input shape according to ds"
        super().__init__(seed)

        in_shape = ds.element_spec[0].shape.as_list()
        in_shape = in_shape[1:] if in_shape[0] is None else in_shape # remove batch dimension
        
        inputs = layers.Input(shape=in_shape)
        flattened = layers.Flatten()(inputs)
        output = layers.Dense(units=1,
                               kernel_initializer=self.createInitializer('glorot_uniform'),
                               bias_initializer=self.createInitializer('zeros'))(flattened)
        self._model = keras.Model(inputs=inputs, outputs=output)

        self.compile()

class Dense_NN(Prediction_Model):
    def __init__(self, ds, input_dropout=0.0, mixed_dropout=0.0, conv1d_layers=[], dense_layers=[32,32], l2_reg=0.0, lr_scheduler=[], seed=None):
        "Create a model with dense and convolutional layers, meant to predict a single output feature at one timestep"
        super().__init__(seed)

        # determine input shape
        in_shape = ds.element_spec[0].shape.as_list()
        in_shape = in_shape[1:] if in_shape[0] is None else in_shape # remove batch dimension

        inputs = layers.Input(shape=in_shape, name='Input')
        x = inputs

        for filters, kernel_size in conv1d_layers:
            x = layers.Conv1D(filters, kernel_size, activation="relu",
                              kernel_regularizer=regularizers.l2(l2_reg),
                              kernel_initializer=self.createInitializer('glorot_uniform'),
                              bias_initializer=self.createInitializer('zeros')
                              )(x)

        x = layers.Flatten()(x)
        if input_dropout > 0.0:
            x = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x)
        for units in dense_layers:
            x = layers.Dense(units=units,
                               activation="relu",
                               kernel_regularizer=regularizers.l2(l2_reg),
                               kernel_initializer=self.createInitializer('glorot_uniform'),
                               bias_initializer=self.createInitializer('zeros'))(x)
            if mixed_dropout > 0.0:
                x = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x)
        outputs = []
        for out_idx, out_feature in enumerate(ds.element_spec[1]):
            # adapt number of neurons to match number of classes... not 20 for _Node and _Type
            n_units = 20
            if '_Node' in out_feature:
                n_units = 5
            elif '_Type' in out_feature:
                n_units = 4
            output = layers.Dense(units=n_units,
                                activation="linear",
                                kernel_regularizer=regularizers.l2(l2_reg),
                                kernel_initializer=self.createInitializer('glorot_uniform'),
                                bias_initializer=self.createInitializer('zeros'),
                                name=out_feature)(x)
            outputs.append(output)
        self._model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer=keras.optimizers.Adam()
        if lr_scheduler:
            if len(lr_scheduler) == 2:
                lr_scheduler = [0.001] + lr_scheduler # add learning rate
            lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_scheduler[0], decay_steps=lr_scheduler[1], decay_rate=lr_scheduler[2], staircase=True)
            optimizer=keras.optimizers.Adam(lr_schedule)

        self.compile(optimizer=optimizer, loss_fn=[tf.losses.SparseCategoricalCrossentropy(from_logits=True) for _ in range(len(ds.element_spec[1]))], metrics=['accuracy'])

class CNN(Prediction_Model):
    def __init__(self, ds, input_dropout=0.0, mixed_dropout=0.0, conv_layers=[64,64,64], l2_reg=0.0, lr_scheduler=[], seed=None):
        "Create a convolutional model, meant to predict a single output feature at one timestep"
        super().__init__(seed)

        # determine input shape
        in_shape = ds.element_spec[0].shape.as_list()
        in_shape = in_shape[1:] if in_shape[0] is None else in_shape # remove batch dimension
        
        inputs = layers.Input(shape=in_shape)
        x = inputs

        if input_dropout > 0.0:
            x = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x)
        for layer_filters in conv_layers:
            x = keras.layers.Conv1D(filters=layer_filters, kernel_size=3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            if mixed_dropout > 0.0:
                x = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Flatten()(x)
            
        output = layers.Dense(units=20,
                            activation="linear",
                            kernel_regularizer=regularizers.l2(l2_reg),
                            kernel_initializer=self.createInitializer('glorot_uniform'),
                            bias_initializer=self.createInitializer('zeros'))(x)
        self._model = keras.Model(inputs=inputs, outputs=output)

        optimizer=keras.optimizers.Adam()
        if lr_scheduler:
            if len(lr_scheduler) == 2:
                lr_scheduler = [0.001] + lr_scheduler # add learning rate
            lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_scheduler[0], decay_steps=lr_scheduler[1], decay_rate=lr_scheduler[2], staircase=True)
            optimizer=keras.optimizers.Adam(lr_schedule)

        self.compile(optimizer=optimizer, loss_fn=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

class Anchored_Regularizer(tf.keras.regularizers.Regularizer):
    """A Regularizer that regularizes around an anchor distribution"""
    def __init__(self, data_noise, stddev, weights_shape, seed):
        self.data_noise = data_noise
        self.stddev = stddev
        self.lambda_anchor = data_noise/(stddev**2)

        # sample anchor weights
        anchor_initializer =  tf.random_normal_initializer(mean=0.0, stddev=stddev, seed=seed)
        self.anchor_weights=anchor_initializer(shape=weights_shape)

    def __call__(self, x):
        loss = self.lambda_anchor * tf.reduce_sum(tf.square(x - self.anchor_weights))
        return loss
    
    def get_config(self):
        return {'data_noise' : float(self.data_noise),
                'stddev' : float(self.stddev),
                'weights_shape' : float(self.weights_shape),
                'seed' : float(self.seed)}

class Anchored_Dense_NN(Prediction_Model):
    def __init__(self, ds, input_dropout=0.0, mixed_dropout=0.0, dense_layers=[32,32],
                 data_noise=0.001, weights_stddev=[np.sqrt(10), np.sqrt(10)], biases_stddev=[np.sqrt(10), np.sqrt(10)],
                 layer_initializer='random_normal',
                 lr_scheduler=[], seed=None):
        "Create a model with dense layers, intended for anchored ensembling"
        assert(len(dense_layers) == len(weights_stddev) == len(biases_stddev))
        super().__init__(seed)

        in_shape = ds.element_spec[0].shape.as_list()
        in_shape = in_shape[1:] if in_shape[0] is None else in_shape # remove batch dimension

        # Create input layer
        inputs = layers.Input(shape=in_shape)
        x = layers.Flatten()(inputs)
        if input_dropout > 0.0:
            x = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x)
        
        # add anchored dense layers
        for idx, units in enumerate(dense_layers):

            x = layers.Dense(units=units,
                            activation="relu",
                            kernel_initializer=self.createInitializer(layer_initializer, mean=0.0, stddev=weights_stddev[idx]),
                            bias_initializer=self.createInitializer(layer_initializer, mean=0.0, stddev=biases_stddev[idx]),
                            kernel_regularizer=Anchored_Regularizer(data_noise=data_noise, stddev=weights_stddev[idx],
                                                                    weights_shape=(dense_layers[idx-1] if idx>0 else in_shape[0]*in_shape[1], units),
                                                                    seed=self._rnd_gen.integers(9999999)),
                            bias_regularizer=Anchored_Regularizer(data_noise=data_noise, stddev=biases_stddev[idx],
                                                                    weights_shape=(units,),
                                                                    seed=self._rnd_gen.integers(9999999)),
                            )(x)
            if mixed_dropout > 0.0:
                x = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x)

        # add output layer
        output_weights_stddev=np.sqrt(1.0/dense_layers[-1]) # Normal scaling? Taken from paper, where layers are assigned var=1/n_hidden
        output = layers.Dense(units=1,
                               activation="linear",
                               kernel_initializer=self.createInitializer(layer_initializer, mean=0.0, stddev=output_weights_stddev),
                               kernel_regularizer=Anchored_Regularizer(data_noise=data_noise, stddev=output_weights_stddev,
                                                                       weights_shape=(dense_layers[idx-1], 1),
                                                                       seed=self._rnd_gen.integers(9999999)),
                               use_bias=False)(x)
        self._model = keras.Model(inputs=inputs, outputs=output)

        optimizer=keras.optimizers.Adam()
        if lr_scheduler:
            if len(lr_scheduler) == 2:
                lr_scheduler = [0.001] + lr_scheduler # add learning rate
            lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_scheduler[0], decay_steps=lr_scheduler[1], decay_rate=lr_scheduler[2], staircase=True)
            optimizer=keras.optimizers.Adam(lr_schedule)

        self.compile(optimizer=optimizer)

class LSTM_NN(Prediction_Model):
    def __init__(self, ds, input_dropout=0.0, mixed_dropout=0.0, lstm_layers=[32,32], dense_layers=[], future_lstm_layers=[], future_dense_layers=[], l2_reg=0.0, lr_scheduler=[], seed=None):
        "Create a model with lstm and dense layers, meant to predict a single output feature at one timestep. Support future-knowledge inputs."
        super().__init__(seed)
        
        # determine input shape(s) and setup input layer(s)
        input_layers = []
        if isinstance(ds.element_spec[0], tuple):
            # multiple inputs
            in_shape_hist = ds.element_spec[0][0].shape.as_list()
            in_shape_hist = in_shape_hist[1:] if in_shape_hist[0] is None else in_shape_hist # remove batch dimension
            
            in_shape_fut = ds.element_spec[0][1].shape.as_list()
            in_shape_fut = in_shape_fut[1:] if in_shape_fut[0] is None else in_shape_fut # remove batch dimension
            history_inputs = layers.Input(shape=in_shape_hist, name="Inputs_History")
            future_inputs = layers.Input(shape=in_shape_fut, name="Inputs_Future")
            input_layers=[history_inputs, future_inputs]
        else:
            in_shape_hist = ds.element_spec[0].shape.as_list()
            in_shape_hist = in_shape_hist[1:] if in_shape_hist[0] is None else in_shape_hist # remove batch dimension
            input_layers = [layers.Input(shape=in_shape_hist, name="Inputs_History")]

        # TODO: add convolutional layers

        ### HISTORY LSTM STACK ##########
        x_h = input_layers[0]
        if input_dropout > 0.0:
            x_h = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x_h)
        for layer_id, layer_units in enumerate(lstm_layers):
            x_h = layers.LSTM(layer_units, 
                              return_sequences=(layer_id!=(len(lstm_layers)-1)),
                              kernel_regularizer=regularizers.l2(l2_reg),
                              kernel_initializer=self.createInitializer('glorot_uniform'),
                              recurrent_initializer=self.createInitializer('orthogonal'),
                              bias_initializer=self.createInitializer('zeros')
                              )(x_h)
            if mixed_dropout > 0.0:
                x_h = layers.Dropout(mixed_dropout, seed=self._rnd_gen.integers(9999999))(x_h)
        x_h = layers.Flatten()(x_h)
        #################################

        x = x_h

        ### FUTURE LSTM + DENSE STACK ###
        if len(input_layers) > 1:
            x_f = input_layers[1]
            if input_dropout > 0.0:
                x_f = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x_f)
            for layer_id, layer_units in enumerate(future_lstm_layers):
                x_f = layers.LSTM(layer_units, 
                              return_sequences=(layer_id!=(len(future_lstm_layers)-1)),
                              kernel_regularizer=regularizers.l2(l2_reg),
                              kernel_initializer=self.createInitializer('glorot_uniform'),
                              recurrent_initializer=self.createInitializer('orthogonal'),
                              bias_initializer=self.createInitializer('zeros')
                              )(x_f)
                if mixed_dropout > 0.0:
                    x_f = layers.Dropout(mixed_dropout)(x_f)
            x_f = layers.Flatten()(x_f)
            for units in future_dense_layers:
                x_f = layers.Dense(units=units,
                               activation="relu",
                               kernel_regularizer=regularizers.l2(l2_reg),
                               kernel_initializer=self.createInitializer('glorot_uniform'),
                               bias_initializer=self.createInitializer('zeros'))(x_f)
            x = layers.Concatenate()([x_h, x_f])
        #################################

        # DENSE LAYERS
        for units in dense_layers:
            x = layers.Dense(units=units,
                               activation="relu",
                               kernel_regularizer=regularizers.l2(l2_reg),
                               kernel_initializer=self.createInitializer('glorot_uniform'),
                               bias_initializer=self.createInitializer('zeros'))(x)
            
        # OUTPUT LAYER
        output = layers.Dense(units=1,
                               activation="linear",
                               kernel_regularizer=regularizers.l2(l2_reg),
                               kernel_initializer=self.createInitializer('glorot_uniform'),
                               bias_initializer=self.createInitializer('zeros'))(x)
        
        self._model = keras.Model(inputs=input_layers, outputs=output)

        optimizer=keras.optimizers.Adam()
        if lr_scheduler:
            if len(lr_scheduler) == 2:
                lr_scheduler = [0.001] + lr_scheduler # add learning rate
            lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_scheduler[0], decay_steps=lr_scheduler[1], decay_rate=lr_scheduler[2], staircase=True)
            optimizer=keras.optimizers.Adam(lr_schedule)

        self.compile(optimizer=optimizer)


class AR_LSTM(Prediction_Model):
    def __init__(self, ds, input_dropout=0.0, mixed_dropout=0.0, lstm_layers=[32,32], pccdyn_importance=0.1, l2_reg=0.0, lr_scheduler=[], seed=None):
        "Create an autoregressive model with lstm layers, meant to predict multiple features over multiple timesteps."""
        super().__init__(seed)

        in_shape = ds.element_spec[0].shape.as_list()
        in_shape = in_shape[1:] if in_shape[0] is None else in_shape # remove batch dimension
        out_shape = ds.element_spec[1].shape.as_list()
        out_shape = out_shape[1:] if out_shape[0] is None else out_shape # remove batch dimension

        lstm_cells = []
        for units in lstm_layers:
            if mixed_dropout < 0.0:
                print("Warning: Setting dropout in LSTMCell is undeterministic.")
                lstm_cell = keras.layers.LSTMCell(units,
                                                  kernel_initializer=self.createInitializer('glorot_uniform'),
                                                  recurrent_initializer=self.createInitializer('orthogonal'),
                                                  bias_initializer=self.createInitializer('zeros'),
                                                  kernel_regularizer=regularizers.l2(l2_reg),
                                                  dropout=-mixed_dropout)
            else:
                lstm_cell = keras.layers.LSTMCell(units,
                                                  kernel_initializer=self.createInitializer('glorot_uniform'),
                                                  recurrent_initializer=self.createInitializer('orthogonal'),
                                                  bias_initializer=self.createInitializer('zeros'),
                                                  kernel_regularizer=regularizers.l2(l2_reg))
            lstm_cells.append(lstm_cell)

        input_dropout_layer = keras.layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999)) if input_dropout > 0.0 else None
        mixed_dropout_layer = keras.layers.Dropout(mixed_dropout, seed=self._rnd_gen.integers(9999999)) if mixed_dropout > 0.0 else None

        output_dense_layer = layers.Dense(units=out_shape[1],
                               activation="linear",
                               kernel_regularizer=regularizers.l2(l2_reg),
                               kernel_initializer=self.createInitializer('glorot_uniform'),
                               bias_initializer=self.createInitializer('zeros'))
                
        
        self._model = FeedBack(out_shape[0], lstm_cells, output_dense_layer, input_dropout_layer, mixed_dropout_layer)
        self._model.build(ds.element_spec[0].shape)

        optimizer=keras.optimizers.Adam()
        if lr_scheduler:
            if len(lr_scheduler) == 2:
                lr_scheduler = [0.001] + lr_scheduler # add learning rate
            lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_scheduler[0], decay_steps=lr_scheduler[1], decay_rate=lr_scheduler[2], staircase=True)
            optimizer=keras.optimizers.Adam(lr_schedule)

        self.compile(optimizer=optimizer, loss_fn=Custom_AR_Loss(pccdyn_importance, out_steps=out_shape[0]))

class Custom_AR_Loss(keras.losses.Loss):
    def __init__(self, pccdyn_importance=0.0, out_steps=10):
        super().__init__()
        
        if(pccdyn_importance < 0.0 or pccdyn_importance > 1.0):
            print("Warning: pccdyn_importance has to be in range [0.0,1.0]. Set to 0.0")
            pccdyn_importance=0.0
        self.pccdyn_importance = pccdyn_importance
        self.inverse_pccdyn_importance = 1.0-pccdyn_importance
        self.out_steps_minus1=out_steps-1

    def call(self, y_true, y_pred):
        # for us, the interesting value is at y_true[:,9,0]
        squared_errors = tf.math.square(y_true-y_pred)
        squared_diff_main_value = squared_errors[:,self.out_steps_minus1,0]
        loss = self.inverse_pccdyn_importance * tf.reduce_mean(squared_errors) + self.pccdyn_importance * squared_diff_main_value
        #loss = tf.reduce_mean(squared_errors) + self.pccdyn_importance * squared_diff_main_value
        return loss

class FeedBack(keras.Model):
    def __init__(self, prediction_timesteps, lstm_cells, output_layer, input_dropout_layer=None, mixed_dropout_layer=None):
        super().__init__()
        self.out_steps = prediction_timesteps

        self.input_dropout_layer = input_dropout_layer
        self.mixed_dropout_layer = mixed_dropout_layer
        
        # stack lstm layers
        self.stacked_lstm_cells = keras.layers.StackedRNNCells(lstm_cells)

        # Also wrap the LSTMCells in an RNN to simplify the `warmup` method.
        self.lstm_rnn = keras.layers.RNN(self.stacked_lstm_cells, return_state=True, unroll=True)

        self.dense = output_layer


    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        #if self.dropout_layer:
        #    inputs = self.dropout_layer(inputs) # apply dropout

        if self.input_dropout_layer:
            inputs = self.input_dropout_layer(inputs)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            if self.mixed_dropout_layer:
                x = self.mixed_dropout_layer(x)
            x, state = self.stacked_lstm_cells(x, states=state,
                                    training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        
        return predictions
