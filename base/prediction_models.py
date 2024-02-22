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
        # TODO: try RMSprop? in general, different optimizers
        assert(self._model is not None)
        self._model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    def evaluate(self, ds, verbose=1):
        assert(self._model is not None)
        return self._model.evaluate(ds, verbose=verbose)
    
    def predict(self, ds, verbose=1):
        assert(self._model is not None)
        return self._model.predict(ds, verbose=verbose)
    
    def fit(self, train_ds, val_ds=None, epochs=100, early_stopping=0, save_best_only=False, best_model_filepath='best_model.hdf5', target_metric='val_accuracy', callbacks=[], class_weight=None, plot_hist=False, verbose=1):
        assert(self._model is not None)

        callbacks=callbacks

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

        self._hist = self._model.fit(train_ds, validation_data=val_ds, verbose=verbose, epochs=epochs, class_weight=class_weight, callbacks=callbacks)

        if save_best_only:
            self._model.load_weights(best_model_filepath)

        if verbose>0:
            print(f"Finished training after {len(self._hist.history['loss'])} epochs.")

        if verbose>0:
            print("Evaluating model:")
            self._model.evaluate(train_ds, verbose=verbose)
            if val_ds is not None:
                self._model.evaluate(val_ds, verbose=verbose)

        if plot_hist: self.plot_hist(self._hist)
        
        return self._hist
    
    def plot_hist(self, hist, custom_keys=None):
        hist_keys = list(hist.history.keys())
        loss_keys = [k for k in hist_keys if 'loss' in k]
        acc_keys = [k for k in hist_keys if 'accuracy' in k]
        other_keys = custom_keys
        fig, axes = plt.subplots(nrows=1, ncols=2 if not other_keys else 3, figsize=(6 if not other_keys else 9,3))
        plt.tight_layout()
        for key in loss_keys:
            axes[0].plot(hist.history[key][1:], label=str(key), linestyle='dashed' if 'val' in key else '-')
            axes[0].legend()
        for key in acc_keys:
            axes[1].plot(hist.history[key][1:], label=str(key), linestyle='dashed' if 'val' in key else '-')
            axes[1].legend()
        if other_keys:
            for key in other_keys:
                axes[2].plot(hist.history[key][1:], label=str(key), linestyle='dashed' if 'val' in key else '-')
                axes[2].legend()
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

    def load_model(self, path):
        self._model = tf.keras.models.load_model(path)
    
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

class AsymmetricMSE(tf.keras.losses.Loss):
    """Loss function that penalizes over- (a>0) or underestimation (a<0). -1<a<1"""
    def __init__(self, alpha):
        super().__init__()
        assert(-1.0<alpha and alpha<1.0)
        self.alpha = alpha

    def __call__(self, y_true, y_pred, sample_weight=None):
        diff = y_pred - y_true
        loss = tf.reduce_mean(tf.square(diff) * tf.square(tf.sign(diff) + self.alpha))
        return loss
    
    def get_config(self):
        return {'alpha' : float(self.alpha)}

class Dense_NN(Prediction_Model):
    def __init__(self, ds,
                 input_dropout=0.0,
                 mixed_dropout_dense=0.0,
                 mixed_dropout_cnn=0.0,
                 mixed_dropout_lstm=0.0,
                 mixed_batchnorm=False,
                 conv1d_layers=[],
                 convlstm1d_layers=[],
                 conv2d_layers=[],
                 lstm_layers=[],
                 dense_layers=[32,32],
                 l2_reg=0.0,
                 lr_scheduler=[],
                 output_type='classification', # 'binary', 'regression'
                 final_activation=None,
                 asymmetric_loss=0.0,
                 seed=None):
        "Create a model with dense and convolutional layers, meant to predict a single output feature at one timestep"
        super().__init__(seed)

        assert(output_type in ['classification', 'binary', 'regression'])

        # determine input shape
        in_shape = ds.element_spec[0].shape.as_list()
        in_shape = in_shape[1:] if in_shape[0] is None else in_shape # remove batch dimension

        # input layer
        inputs = layers.Input(shape=in_shape, name='Input')
        x = inputs

        if input_dropout > 0.0:
            x = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x)

        # CNN stack
        for filters, kernel_size in conv1d_layers:
            x = layers.Conv1D(filters, kernel_size, activation=None,
                              kernel_regularizer=regularizers.l2(l2_reg),
                              kernel_initializer=self.createInitializer('glorot_uniform'),
                              bias_initializer=self.createInitializer('zeros')
                              )(x)
            if mixed_batchnorm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            if mixed_dropout_cnn > 0.0:
                x = layers.Dropout(mixed_dropout_cnn, seed=self._rnd_gen.integers(9999999))(x)

        for layer_id, (filters, kernel_size) in enumerate(convlstm1d_layers):
            if layer_id == 0:
                x = layers.Reshape((in_shape+[1]))(x)
            x = layers.ConvLSTM1D(filters=filters, kernel_size=kernel_size, activation='tanh',
                              return_sequences=(layer_id!=(len(convlstm1d_layers)-1)),
                              kernel_regularizer=regularizers.l2(l2_reg),
                              recurrent_activation='sigmoid',
                              kernel_initializer=self.createInitializer('glorot_uniform'),
                              recurrent_initializer=self.createInitializer('orthogonal'),
                              bias_initializer=self.createInitializer('zeros'),
                              dropout=mixed_dropout_cnn, # Adding dropout here reduces reproducability, but we dont have that anyway due to GPU computing
                              recurrent_dropout=0.0 # ! be aware this uses mixed dropout CNN!
                              )(x)
            if mixed_batchnorm:
                x = layers.BatchNormalization()(x)
            # if mixed_dropout_cnn > 0.0:
            #     x = layers.Dropout(mixed_dropout_cnn, seed=self._rnd_gen.integers(9999999))(x)

        if conv2d_layers:
            x = layers.Reshape((65,12,1))(x)
        for filters, kernel_size in conv2d_layers:
            x = layers.Conv2D(filters, kernel_size, activation=None,
                              kernel_regularizer=regularizers.l2(l2_reg),
                              kernel_initializer=self.createInitializer('glorot_uniform'),
                              bias_initializer=self.createInitializer('zeros')
                              )(x)
            if mixed_batchnorm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            if mixed_dropout_cnn > 0.0:
                x = layers.Dropout(mixed_dropout_cnn, seed=self._rnd_gen.integers(9999999))(x)

        # lstm stack
        for layer_id, units in enumerate(lstm_layers):
            x = layers.LSTM(units, 
                            activation='tanh',
                              return_sequences=(layer_id!=(len(lstm_layers)-1)),
                              kernel_regularizer=regularizers.l2(l2_reg),
                              kernel_initializer=self.createInitializer('glorot_uniform'),
                              recurrent_initializer=self.createInitializer('orthogonal'),
                              bias_initializer=self.createInitializer('zeros'),
                              dropout=mixed_dropout_lstm, # Adding dropout here reduces reproducability, but we dont have that anyway due to GPU computing
                              recurrent_dropout=mixed_dropout_lstm
                              )(x)
            if mixed_batchnorm:
                x = layers.BatchNormalization()(x)
            # if mixed_dropout_lstm > 0.0:
            #     x = layers.Dropout(mixed_dropout_lstm, seed=self._rnd_gen.integers(9999999))(x)

        # dense stack
        x = layers.Flatten()(x)
        for units in dense_layers:
            x = layers.Dense(units=units,
                               activation=None,
                               kernel_regularizer=regularizers.l2(l2_reg),
                               kernel_initializer=self.createInitializer('glorot_uniform'),
                               bias_initializer=self.createInitializer('zeros'))(x)
            x = layers.Activation('relu')(x)
            if mixed_dropout_dense > 0.0:
                x = layers.Dropout(mixed_dropout_dense, seed=self._rnd_gen.integers(9999999))(x)
        
        # create outputs
        outputs = []
        for out_idx, out_feature in enumerate(ds.element_spec[1]):
            # adapt number of neurons to match number of classes
            n_units = 16
            if output_type == 'binary' or output_type == 'regression' or '_Location' in out_feature:
                n_units = 1
            elif '_Node' in out_feature:
                n_units = 4
            elif '_Type' in out_feature:
                n_units = 4

            output_activation = final_activation if final_activation is not None else ('sigmoid' if output_type=='binary'
                                                                                       else ('softmax' if output_type=='classification'
                                                                                       else 'linear'))
            
            output = layers.Dense(units=n_units,
                                activation=output_activation,
                                kernel_regularizer=regularizers.l2(l2_reg),
                                kernel_initializer=self.createInitializer('glorot_uniform'),
                                bias_initializer=self.createInitializer('zeros'),
                                name=out_feature)(x)
            outputs.append(output)
        self._model = keras.Model(inputs=inputs, outputs=outputs[0] if len(outputs)==1 else outputs)

        # create optimizer
        optimizer=keras.optimizers.Adam()
        if lr_scheduler:
            if len(lr_scheduler) == 2:
                lr_scheduler = [0.001] + lr_scheduler # add learning rate
            lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_scheduler[0], decay_steps=lr_scheduler[1], decay_rate=lr_scheduler[2], staircase=True)
            optimizer=keras.optimizers.Adam(lr_schedule)

        # select losses and metrics
        loss_functions = {
            'binary' : [tf.losses.BinaryCrossentropy()],
            'classification' : [tf.losses.SparseCategoricalCrossentropy() for _ in range(len(ds.element_spec[1]))],
            'regression' : [AsymmetricMSE(alpha=asymmetric_loss) for _ in range(len(ds.element_spec[1]))]
        }
        metrics = {
            'binary' : [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            'classification' : ['accuracy'],
            'regression' : ['mse', 'mae']
        }

        self.compile(optimizer=optimizer, loss_fn=loss_functions[output_type], metrics=metrics[output_type])

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
    def __init__(self, ds, input_dropout=0.0, mixed_dropout=0.0, lstm_layers=[32,32], dense_layers=[], l2_reg=0.0, lr_scheduler=[], seed=None):
        "Create a model with lstm and dense layers, meant to predict a single output feature at one timestep. Support future-knowledge inputs."
        super().__init__(seed)
        
        # determine input shape
        in_shape = ds.element_spec[0].shape.as_list()
        in_shape = in_shape[1:] if in_shape[0] is None else in_shape # remove batch dimension

        inputs = layers.Input(shape=in_shape, name='Input')
        x = inputs

        # TODO: add convolutional layers

        ### HISTORY LSTM STACK ##########
        x_h = x
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

        # DENSE LAYERS
        for units in dense_layers:
            x = layers.Dense(units=units,
                               activation="relu",
                               kernel_regularizer=regularizers.l2(l2_reg),
                               kernel_initializer=self.createInitializer('glorot_uniform'),
                               bias_initializer=self.createInitializer('zeros'))(x)
            
        # OUTPUT LAYER
        outputs = []
        for out_idx, out_feature in enumerate(ds.element_spec[1]):
            # adapt number of neurons to match number of classes... not 20 for _Node and _Type
            n_units = 16
            if '_Location' in out_feature:
                n_units = 2
            elif '_Node' in out_feature:
                n_units = 4
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


class STATEFUL_LSTM(Prediction_Model):
    def __init__(self, train_ds, val_ds,
                 input_dropout=0.0,
                 mixed_dropout_lstm=0.0,
                 mixed_dropout_dense=0.0,
                 lstm_layers=[32,32],
                 dense_layers=[],
                 l2_reg=0.0,
                 lr_scheduler=[],
                 seed=None):
        super().__init__(seed)


        # The input sizes need to match the dimensions of each dataset
        self._train_model = self.create_model(train_ds, batch_size=train_ds._batch_size.numpy(), input_dropout=input_dropout, mixed_dropout_lstm=mixed_dropout_lstm,
                     mixed_dropout_dense=mixed_dropout_dense,
                     lstm_layers=lstm_layers, dense_layers=dense_layers, l2_reg=l2_reg,
                     lr_scheduler=lr_scheduler)
        
        self._val_model = self.create_model(val_ds, batch_size=val_ds._batch_size.numpy(), input_dropout=input_dropout, mixed_dropout_lstm=mixed_dropout_lstm,
                     mixed_dropout_dense=mixed_dropout_dense,
                     lstm_layers=lstm_layers, dense_layers=dense_layers, l2_reg=l2_reg,
                     lr_scheduler=lr_scheduler)
        
        self._pred_model = self.create_model(val_ds, batch_size=1, input_dropout=input_dropout, mixed_dropout_lstm=mixed_dropout_lstm,
                     mixed_dropout_dense=mixed_dropout_dense,
                     lstm_layers=lstm_layers, dense_layers=dense_layers, l2_reg=l2_reg,
                     lr_scheduler=lr_scheduler)
        
    def save_weights(self, dir):
        self._train_model.save_weights(dir)

    def load_weights(self, dir):
        self._train_model.load_weights(dir)
        self._val_model.load_weights(dir)
        self._pred_model.load_weights(dir)

    def create_model(self, ds, batch_size, 
                     input_dropout=0.0,
                     mixed_dropout_lstm=0.0,
                     mixed_dropout_dense=0.0,
                     lstm_layers=[32,32], dense_layers=[], l2_reg=0.0, lr_scheduler=[], seed=None):
        # determine input shape(s) and setup input layer(s)
        in_shape_hist = ds.element_spec[0].shape.as_list()
        in_shape_hist = in_shape_hist[1:] if in_shape_hist[0] is None else in_shape_hist # remove batch dimension
        input_layer = layers.Input(shape=in_shape_hist, batch_size=batch_size, name="Inputs_History")

        ### HISTORY LSTM STACK ##########
        x = input_layer
        if input_dropout > 0.0:
            x = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x)
        for layer_id, layer_units in enumerate(lstm_layers):
            x = layers.LSTM(layer_units, 
                              return_sequences=(layer_id!=(len(lstm_layers)-1)),
                              kernel_regularizer=regularizers.l2(l2_reg),
                              stateful=True,
                              kernel_initializer=self.createInitializer('glorot_uniform'),
                              recurrent_initializer=self.createInitializer('orthogonal'),
                              bias_initializer=self.createInitializer('zeros')
                              )(x)
            if mixed_dropout_lstm > 0.0:
                x = layers.Dropout(mixed_dropout_lstm, seed=self._rnd_gen.integers(9999999))(x)
        x = layers.Flatten()(x)
        #################################

        # DENSE LAYERS
        for units in dense_layers:
            x = layers.Dense(units=units,
                               activation="relu",
                               kernel_regularizer=regularizers.l2(l2_reg),
                               kernel_initializer=self.createInitializer('glorot_uniform'),
                               bias_initializer=self.createInitializer('zeros'))(x)
            if mixed_dropout_dense > 0.0:
                x = layers.Dropout(mixed_dropout_dense, seed=self._rnd_gen.integers(9999999))(x)
            
        # OUTPUT LAYER
        outputs = []
        for out_idx, out_feature in enumerate(ds.element_spec[1]):
            # adapt number of neurons to match number of classes... not 20 for _Node and _Type
            n_units = 16
            if '_Location' in out_feature:
                n_units = 1
            elif '_Node' in out_feature:
                n_units = 4
            elif '_Type' in out_feature:
                n_units = 4
            
            output = layers.Dense(units=n_units,
                                activation='linear',
                                kernel_regularizer=regularizers.l2(l2_reg),
                                kernel_initializer=self.createInitializer('glorot_uniform'),
                                bias_initializer=self.createInitializer('zeros'),
                                name=out_feature)(x)
            outputs.append(output)
        
        model = tf.keras.Model(inputs=input_layer, outputs=output)

        optimizer=tf.keras.optimizers.Adam()
        if lr_scheduler:
            if len(lr_scheduler) == 2:
                lr_scheduler = [0.001] + lr_scheduler # add learning rate
            lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_scheduler[0], decay_steps=lr_scheduler[1], decay_rate=lr_scheduler[2], staircase=True)
            optimizer=tf.keras.optimizers.Adam(lr_schedule)

        model.compile(optimizer=optimizer, loss=[tf.losses.MeanSquaredError() for _ in range(len(ds.element_spec[1]))], metrics=['mse', 'mae'])
        return model

    def summary(self):
        self._train_model.summary()

    def fit(self, train_ds, val_ds, epochs=100, eval_frequency=5, save_best_only=False, early_stopping=0, target_metric='val_mse', plot_hist=False, verbose=2):
        
        training_hist = {}
        training_hist['loss'] = []
        training_hist['mse'] = []
        training_hist['epoch'] = []
        if val_ds is not None:
            training_hist['val_loss'] = []
            training_hist['val_mse'] = []
        best_weights = None
        best_eval_epoch = 0
        n_actual_epochs = 0
        for epoch in range(epochs):
            n_actual_epochs+=1
            self._train_model.fit(train_ds, epochs=1, verbose=verbose, shuffle=False)
            self._train_model.reset_states()
            if epoch%eval_frequency==0:
                train_loss, train_mse, train_mae = self._train_model.evaluate(train_ds, verbose=0)
                self._train_model.reset_states()
                if val_ds is not None:
                    self._val_model.set_weights(self._train_model.get_weights())
                    val_loss, val_mse, val_mae = self._val_model.evaluate(val_ds, verbose=0)
                    self._val_model.reset_states()
                new_best=False
                if save_best_only and (len(training_hist[target_metric]) == 0 or val_mse < np.min(training_hist[target_metric])):
                    new_best=True
                    best_eval_epoch = epoch
                    best_weights = self._train_model.get_weights()
                training_hist['loss'].append(train_loss)
                training_hist['mse'].append(train_mse)
                training_hist['epoch'].append(epoch)
                if val_ds is not None:
                    training_hist['val_loss'].append(val_loss)
                    training_hist['val_mse'].append(val_mse)
                if verbose > 0:
                    print(f"Epoch {epoch}/{epochs}: Train:[{train_loss:.4f}/{train_mse:.4f}]" +
                        (f" Val:[{val_loss:.4f}/{val_mse:.4f}]" if val_ds is not None else "") + 
                        " (loss/mse)" + 
                        (" -> New best!" if new_best else ""))
                if early_stopping > 0 and  (epoch > (best_eval_epoch + early_stopping)) and val_mse > np.min(training_hist[target_metric]):
                    if verbose > 0:
                        print(f"Val_mse did not improve since epoch {best_eval_epoch}, early-stopping")
                        break
        if save_best_only:
            self._train_model.set_weights(best_weights)
        self._val_model.set_weights(self._train_model.get_weights())
        self._pred_model.set_weights(self._train_model.get_weights())

        if verbose>0:
            train_loss, train_mse, train_mae = self._train_model.evaluate(train_ds, verbose=verbose)
            val_loss, val_mse, val_mae = self._val_model.evaluate(val_ds, verbose=verbose)
            print(f"Finished Training after {n_actual_epochs} epochs. Final Metrics:\n" +
                    f"Train:[{train_loss:.4f}/{train_mse:.4f}]" + 
                    (f"Val:[{val_loss:.4f}/{val_mse:.4f}]" if val_ds is not None else "") + 
                    "(loss/mse)")
            
        if plot_hist:
            plot_keys = list(training_hist.keys())
            plot_keys.remove('epoch')
            print(plot_keys)
            self.plot_hist(training_hist)

        return training_hist

    def predict(self, ds, verbose=1):
        assert(self._pred_model is not None)
        self._pred_model.reset_states()
        return self._pred_model.predict(ds, verbose=verbose)
    
    def evaluate(self, ds, verbose=1):
        assert(self._val_model is not None)
        self._val_model.reset_states()
        return self._val_model.predict(ds, verbose=verbose)
