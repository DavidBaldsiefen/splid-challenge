import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers, optimizers
import matplotlib.pyplot as plt
import numpy as np


class Prediction_Model():
    """Tensorflow model wrapper that covers some boilerplate code."""
    def __init__(self, seed=None):
        self._seed=seed
        self._model = None
        self._rnd_gen = np.random.default_rng(self._seed)

    def compile(self, optimizer=keras.optimizers.legacy.Adam(), loss_fn=keras.losses.MeanSquaredError(), metrics=['mse']):
        assert(self._model is not None)
        self._model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    def evaluate(self, ds, verbose=1):
        assert(self._model is not None)
        return self._model.evaluate(ds, verbose=verbose)
    
    def predict(self, ds, verbose=1):
        assert(self._model is not None)
        return self._model.predict(ds, verbose=verbose)
    
    def fit(self, train_ds, val_ds=None, epochs=100, early_stopping=0, save_best_only=False, target_metric='val_accuracy', callbacks=[], class_weight=None, plot_hist=False, verbose=1):
        """Train the model using the specified parameters"""
        assert(self._model is not None)

        callbacks=callbacks

        # add callbacks
        if early_stopping > 0:
            callbacks.append(keras.callbacks.EarlyStopping(monitor=target_metric,
                                           mode='auto',
                                           patience=early_stopping,
                                           restore_best_weights=save_best_only,
                                           verbose=verbose))
            
        if verbose>0:
            print(f"Starting training. Optimizing \"{target_metric}\"")

        self._hist = self._model.fit(train_ds, validation_data=val_ds, verbose=verbose, epochs=epochs, class_weight=class_weight, callbacks=callbacks)

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
        """Plot training curves"""
        hist_keys = list(hist.history.keys())
        loss_keys = [k for k in hist_keys if 'loss' in k]
        acc_keys = [k for k in hist_keys if 'accuracy' in k]
        other_keys = custom_keys
        fig, axes = plt.subplots(nrows=1, ncols=2 if not other_keys else 3, figsize=(10 if not other_keys else 14,3))
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
        """Wrapper to create seeded initializers, aiding in reproducibility"""
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

    def save(self, path):
        self._model.save(path)
    
    @property
    def model(self):
        return self._model
    
    @property
    def layers(self):
        return self._model.layers
    
    @property
    def seed(self):
        return self._seed
    
    @property
    def hist(self):
        return self._hist

class AsymmetricMSE(tf.keras.losses.Loss):
    """Loss function that penalizes over- (a>0) or underestimation (a<0). alpha has to be in range ]-1,1["""
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
                 mixed_batchnorm_cnn=False,
                 mixed_batchnorm_dense=False,
                 mixed_batchnorm_lstm=False,
                 split_cnn=False,
                 split_dense=False,
                 split_lstm=False,
                 cnn_lstm_order='cnn_lstm', # 'lstm_cnn'
                 mixed_batchnorm_before_relu=False,
                 conv1d_layers=[], # [filters, kernel_size, kernel_stride, dilation_rate, maxpool]
                 convlstm1d_layers=[], # deprecated
                 conv2d_layers=[], # deprecated
                 lstm_layers=[], # [units, return_sequence, maxpool, avgpool]
                 dense_layers=[32,32],
                 deep_layer_in_output=False, # deprecated
                 l1_reg=0.0,
                 l2_reg=0.0,
                 lr_scheduler=[],
                 output_type='classification', # 'binary', 'regression', 'oneshot'
                 final_activation=None,
                 final_activation_bias_initializer=None,
                 asymmetric_loss=0.0,
                 optimizer='adam',
                 seed=None):
        """Create a model consisting of LSTM, CNN, Dense and Regularization/Pooling layers.
        Supports multiple in- and outputs based on the provided datset specifications."""
        super().__init__(seed)

        assert(output_type in ['classification', 'binary', 'regression', 'oneshot'])

        # determine input shape
        in_shape_local = ds.element_spec[0]['local_in'].shape.as_list()
        in_shape_local = in_shape_local[1:] if in_shape_local[0] is None else in_shape_local # remove batch dimension
        if 'global_in' in ds.element_spec[0].keys():
            global_input = True
            in_shape_global = ds.element_spec[0]['global_in'].shape.as_list()
        else:
            global_input=False
            in_shape_global = (1024,1024)
        in_shape_global = in_shape_global[1:] if in_shape_global[0] is None else in_shape_global # remove batch dimension

        if len(ds.element_spec[1]) != 2:
            split_cnn=False
            split_dense=False
            split_lstm=False

        # input layer
        inputs_local = layers.Input(shape=in_shape_local, name='local_in')
        x__L_local = inputs_local

        inputs_global = layers.Input(shape=in_shape_global, name='global_in')
        x__L_global = inputs_global

        if input_dropout > 0.0:
            x__L_local = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x__L_local)
            x__L_global = layers.Dropout(input_dropout, seed=self._rnd_gen.integers(9999999))(x__L_global)

        # STACKS
        if cnn_lstm_order == 'cnn_lstm':
            x_L_local, x_R_local, x_L_global, x_R_global = self.create_parallel_stack(split_cnn, x__L_local, None, x__L_global, None, 
                                                                self.create_cnn_stack, 
                                                                cnn_layers=conv1d_layers,
                                                                dropout=mixed_dropout_cnn,
                                                                batchnorm=mixed_batchnorm_cnn,
                                                                batchnorm_before_relu=mixed_batchnorm_before_relu,
                                                                l1=l1_reg,
                                                                l2=l2_reg)
            
            x_L_local, x_R_local, x_L_global, x_R_global = self.create_parallel_stack(split_lstm, x_L_local, x_R_local, x_L_global, x_R_global, 
                                                                self.create_lstm_stack, 
                                                                lstm_layers=lstm_layers,
                                                                dropout=mixed_dropout_lstm,
                                                                batchnorm=mixed_batchnorm_lstm,
                                                                l1=l1_reg,
                                                                l2=l2_reg,
                                                                return_sequences=False)
        else:
            x_L_local, x_R_local, x_L_global, x_R_global = self.create_parallel_stack(split_lstm, x__L_local, None, x__L_global, None, 
                                                                self.create_lstm_stack, 
                                                                lstm_layers=lstm_layers,
                                                                dropout=mixed_dropout_lstm,
                                                                batchnorm=mixed_batchnorm_lstm,
                                                                l1=l1_reg,
                                                                l2=l2_reg,
                                                                return_sequences=False)
            
            x_L_local, x_R_local, x_L_global, x_R_global = self.create_parallel_stack(split_cnn, x_L_local, x_R_local, x_L_global, x_R_global, 
                                                                self.create_cnn_stack, 
                                                                cnn_layers=conv1d_layers,
                                                                dropout=mixed_dropout_cnn,
                                                                batchnorm=mixed_batchnorm_cnn,
                                                                batchnorm_before_relu=mixed_batchnorm_before_relu,
                                                                l1=l1_reg,
                                                                l2=l2_reg)
             
        # concatenate the local and global pathways here if applicable
        if global_input:
            if x_R_local is not None:
                x_L_local = layers.Concatenate(axis=-1)([x_L_local, x_L_global])
                x_R_local = layers.Concatenate(axis=-1)([x_R_local, x_R_global])
            else:
                x_L_local = layers.Concatenate(axis=-1)([x_L_local, x_L_global])
        
        x_L_local, x_R_local, x_L_global, x_R_global = self.create_parallel_stack(split_dense, x_L_local, x_R_local, x_L_global, x_R_global, 
                                                                self.create_dense_stack, 
                                                                dense_layers=dense_layers,
                                                                dropout=mixed_dropout_dense,
                                                                batchnorm=mixed_batchnorm_dense,
                                                                batchnorm_before_relu=mixed_batchnorm_before_relu,
                                                                l1=l1_reg,
                                                                l2=l2_reg)
        
        # create outputs
        outputs = []
        for out_idx, out_feature in enumerate(ds.element_spec[1]):
            # adapt number of neurons to match number of classes
            n_units = 16
            if output_type == 'oneshot':
                n_units = ds.element_spec[1][out_feature].shape[1]
            elif output_type == 'binary' or output_type == 'regression' or '_Location' in out_feature:
                n_units = 1
            elif '_Node' in out_feature:
                n_units = 4
            elif '_Type' in out_feature:
                n_units = 4

            output_activation = final_activation if final_activation is not None else ('sigmoid' if output_type=='binary'
                                                                                       else ('softmax' if output_type=='classification'
                                                                                       else 'linear'))
            output_input = x_L_local if ((out_idx==0) or (x_R_local is None)) else x_R_local
            output = layers.Dense(units=n_units,
                                activation=output_activation,
                                kernel_regularizer=regularizers.L1L2(l1=l1_reg, l2=l2_reg), # TODO: regularizing the last layer is probably not ideal
                                kernel_initializer=self.createInitializer('glorot_uniform'), 
                                bias_initializer=self.createInitializer('zeros') if final_activation_bias_initializer is None else final_activation_bias_initializer,
                                name=out_feature)(output_input)
            outputs.append(output)
        self._model = keras.Model(inputs=[inputs_local, inputs_global] if global_input else [inputs_local], outputs=outputs[0] if len(outputs)==1 else outputs)

        # create optimizer
        if optimizer == 'adam':
            optimizer=keras.optimizers.Adam(learning_rate=0.001 if not lr_scheduler else
                                        lr_scheduler[0] if len(lr_scheduler)==1 else
                                        tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=lr_scheduler[0], decay_rate=lr_scheduler[1], staircase=True) if len(lr_scheduler)==2 else
                                        tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_scheduler[0], decay_steps=lr_scheduler[1], decay_rate=lr_scheduler[2], staircase=True))
        elif optimizer == 'SGD':
            optimizer=keras.optimizers.experimental.SGD(learning_rate=0.001 if not lr_scheduler else
                                        lr_scheduler[0] if len(lr_scheduler)==1 else
                                        tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=lr_scheduler[0], decay_rate=lr_scheduler[1], staircase=True) if len(lr_scheduler)==2 else
                                        tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_scheduler[0], decay_steps=lr_scheduler[1], decay_rate=lr_scheduler[2], staircase=True))
        elif optimizer == 'RMSprop':
            optimizer=keras.optimizers.experimental.RMSprop(learning_rate=0.001 if not lr_scheduler else
                                        lr_scheduler[0] if len(lr_scheduler)==1 else
                                        tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=lr_scheduler[0], decay_rate=lr_scheduler[1], staircase=True) if len(lr_scheduler)==2 else
                                        tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_scheduler[0], decay_steps=lr_scheduler[1], decay_rate=lr_scheduler[2], staircase=True))
        else:
            print('Unknown optimizer. defaulting to Adam')
            optimizer=keras.optimizers.Adam(learning_rate=0.001 if not lr_scheduler else
                                        lr_scheduler[0] if len(lr_scheduler)==1 else
                                        tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=lr_scheduler[0], decay_rate=lr_scheduler[1], staircase=True) if len(lr_scheduler)==2 else
                                        tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_scheduler[0], decay_steps=lr_scheduler[1], decay_rate=lr_scheduler[2], staircase=True))

        # select losses and metrics
        loss_functions = {
            'binary' : [tf.losses.BinaryCrossentropy() for _ in range(len(ds.element_spec[1]))],
            'oneshot' : [[AsymmetricMSE(alpha=asymmetric_loss) for _ in range(len(ds.element_spec[1]))] if asymmetric_loss != 0.0 else
                            [tf.losses.MeanSquaredError() for _ in range(len(ds.element_spec[1]))]],
            'classification' : [tf.losses.SparseCategoricalCrossentropy() for _ in range(len(ds.element_spec[1]))],
            'regression' : [[AsymmetricMSE(alpha=asymmetric_loss) for _ in range(len(ds.element_spec[1]))] if asymmetric_loss != 0.0 else
                            [tf.losses.MeanSquaredError() for _ in range(len(ds.element_spec[1]))]]
        }
        metrics = {
            'binary' : [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            'oneshot' : ['mse'],
            'classification' : ['accuracy'],
            'regression' : ['mse'] # mae
        }

        self.compile(optimizer=optimizer, loss_fn=loss_functions[output_type], metrics=metrics[output_type])
    
    def create_dense_stack(self, input_layer, dense_layers, dropout, batchnorm, batchnorm_before_relu, l1, l2, name):
        """Create a stack of dense layers, optionally followed by batchnorm and dropout layers"""
        if not dense_layers:
            return input_layer
        x = layers.Flatten(name=f'dense_flatten_{name}')(input_layer)
        for layer_idx, units in enumerate(dense_layers):
            x = layers.Dense(units=units,
                                activation=None,
                                kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
                                kernel_initializer=self.createInitializer('glorot_uniform'),
                                bias_initializer=self.createInitializer('zeros'),
                                name=f'dense_{layer_idx}_{name}')(x)
            if batchnorm and batchnorm_before_relu:
                x = layers.BatchNormalization(name=f'dense_BN_{layer_idx}_{name}')(x)
            x = layers.Activation('relu', name=f'dense_relu_{layer_idx}_{name}')(x)
            if batchnorm and not batchnorm_before_relu:
                x = layers.BatchNormalization(name=f'dense_BN_{layer_idx}_{name}')(x)
            if dropout > 0.0:
                x = layers.Dropout(dropout, seed=self._rnd_gen.integers(9999999), name=f'dense_DO_{layer_idx}_{name}')(x)
        return x
    
    def create_cnn_stack(self, input_layer, cnn_layers, dropout, batchnorm, batchnorm_before_relu, l1, l2, name=''):
        """Create a stack of Conv1D layers, optionally followed by batchnorm, MaxPool and dropout layers"""
        if not cnn_layers:
            return input_layer
        x = input_layer
        for layer_idx, (filters, kernel_size, kernel_stride, dilation_rate, maxpool) in enumerate(cnn_layers):
            x = layers.Conv1D(filters,
                              kernel_size,
                              strides=kernel_stride,
                              dilation_rate=dilation_rate,
                              activation=None,
                              kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
                              kernel_initializer=self.createInitializer('glorot_uniform'),
                              bias_initializer=self.createInitializer('zeros'),
                              name=f'conv1d_{layer_idx}_{name}')(x)
            if batchnorm and batchnorm_before_relu:
                x = layers.BatchNormalization(name=f'conv1d_BN_{layer_idx}_{name}')(x)
            x = layers.Activation('relu', name=f'conv1d_relu_{layer_idx}_{name}')(x)
            if batchnorm and not batchnorm_before_relu:
                x = layers.BatchNormalization(name=f'conv1d_BN_{layer_idx}_{name}')(x)
            if maxpool>1:
                x = layers.MaxPool1D(maxpool, name=f'conv1d_MP_{layer_idx}_{name}')(x)
            if dropout > 0.0:
                x = layers.Dropout(dropout, seed=self._rnd_gen.integers(9999999), name=f'conv1d_DO_{layer_idx}_{name}')(x)
        return x
    
    def create_lstm_stack(self, input_layer, lstm_layers, dropout, batchnorm, l1, l2, return_sequences, name=''):
        """Create a stack of LSTM layers, optionally followed by batchnorm, MaxPool, AvgPool and dropout layers"""
        if not lstm_layers:
            return input_layer
        x = input_layer
        for layer_idx, (units, layer_return_sequences, maxpool, avgpool) in enumerate(lstm_layers):
            x = layers.LSTM(units, 
                            activation='tanh',
                              return_sequences=(return_sequences or layer_return_sequences),
                              kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
                              kernel_initializer=self.createInitializer('glorot_uniform'),
                              recurrent_initializer=self.createInitializer('orthogonal'),
                              bias_initializer=self.createInitializer('zeros'),
                              dropout=0.0, # Adding dropout here reduces reproducability, but we dont have that anyway due to GPU computing
                              recurrent_dropout=0.0, # adding recurrent dropout prevents CUDNN computation!
                              name=f'lstm_{layer_idx}_{name}')(x)
            if batchnorm:
                x = layers.BatchNormalization(name=f'lstm_BN_{layer_idx}_{name}')(x)
            if maxpool>1:
                x = layers.MaxPool1D(maxpool, name=f'lstm_MP_{layer_idx}_{name}')(x)
            if avgpool>1:
                x = layers.AvgPool1D(avgpool, name=f'lstm_MP_{layer_idx}_{name}')(x)
            if dropout > 0.0:
                x = layers.Dropout(dropout, seed=self._rnd_gen.integers(9999999), name=f'lstm_DO_{layer_idx}_{name}')(x)
        return x

    def create_parallel_stack(self, split, x_L_local, x_R_local, x_L_global, x_R_global, stack_func, **args):
        """Code that handles the creation of parallel stacks for multi-input models"""
        split_input= x_R_local is not None

        if split:
            x_L_local_new = stack_func(input_layer=x_L_local, 
                                   name='L_local',
                                   **args)
            x_R_local_new = stack_func(input_layer=(x_R_local if split_input else x_L_local), 
                                   name='R_local',
                                   **args)
            x_L_global_new = stack_func(input_layer=x_L_global, 
                                   name='L_global',
                                   **args)
            x_R_global_new = stack_func(input_layer=(x_R_global if split_input else x_L_global), 
                                   name='R_global',
                                   **args)
        else:
            if split_input: 
                x_L_local = layers.Concatenate(axis=-1)([x_L_local, x_R_local])
                x_L_global = layers.Concatenate(axis=-1)([x_L_global, x_R_global])
            
            x_L_local_new = stack_func(input_layer=x_L_local, 
                                   name='local',
                                   **args)
            x_R_local_new = None
            x_L_global_new = stack_func(input_layer=x_L_global, 
                                   name='global',
                                   **args)
            x_R_global_new = None
        return x_L_local_new, x_R_local_new, x_L_global_new, x_R_global_new