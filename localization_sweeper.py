import wandb
import tensorflow as tf
from wandb.keras import WandbMetricsLogger
from pathlib import Path
#from pympler.tracker import SummaryTracker
#import tracemalloc
import gc
import numpy as np
import pickle

from base import datahandler, prediction_models, utils, localizer




class ClearMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

def parameter_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # for key in ['Precision', 'Recall', 'F2', 'TP', 'RMSE', 'FP', 'FN']:
        #     wandb.define_metric(f'{key}', summary="max" if key in ['Precision', 'Recall', 'F2', 'TP'] else 'min')
        #     wandb.define_metric(f'val/{key}', summary="max" if key in ['Precision', 'Recall', 'F2', 'TP'] else 'min')
        #     wandb.define_metric(f'train/{key}', summary="max" if key in ['Precision', 'Recall', 'F2', 'TP'] else 'min')

        # =================================Data Loading & Preprocessing================================================

        # Load data
        challenge_data_dir = Path('dataset/phase_1_v3/')
        data_dir = challenge_data_dir / "train"
        labels_dir = challenge_data_dir / 'train_labels.csv'
        split_dataframes = datahandler.load_and_prepare_dataframes(data_dir, labels_dir)

        directions=config.training['directions']
        print(f"Directions: {directions}")

        # Create Dataset
        utils.set_random_seed(42)

        non_transform_features =[]
        diff_transform_features=[]
        sin_transform_features = []
        sin_cos_transform_features = []

        for key, value in config.input_features.items():
            ft_name = key
            if key == 'Eccentricity': ft_name = 'Eccentricity'
            elif key == 'Semimajor_Axis': ft_name = 'Semimajor Axis (m)'
            elif key == 'Inclination': ft_name = 'Inclination (deg)'
            elif key == 'RAAN': ft_name = 'RAAN (deg)'
            elif key == 'Argument_of_Periapsis': ft_name = 'Argument of Periapsis (deg)'
            elif key == 'True_Anomaly': ft_name = 'True Anomaly (deg)'
            elif key == 'Longitude': ft_name = 'Longitude (deg)'
            elif key == 'Latitude': ft_name = 'Latitude (deg)'
            else: print(f"WARNING! UNKNOWN INPUT FEATURE KEY: {key}")
            if value == True:
                non_transform_features += [ft_name]

        for key, value in config.feature_engineering.items():
            ft_name = key.replace('_', ' ') + ' (deg)'
            if value == 'non': non_transform_features += [ft_name]
            elif value == 'diff': diff_transform_features += [ft_name]
            elif value == 'sin': sin_transform_features += [ft_name]
            elif value == 'sin_cos': sin_cos_transform_features += [ft_name]
            else: print(f"Warning: unknown feature_engineering attribute \'{value}\' for feature {ft_name}")

        ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,#{df_k : split_dataframes[df_k] for df_k in list(split_dataframes.keys())[:900]},
                                                non_transform_features=non_transform_features,
                                                diff_transform_features=diff_transform_features,
                                                sin_transform_features=sin_transform_features,
                                                sin_cos_transform_features=sin_cos_transform_features,
                                                overview_features_mean=config.ds_gen['overview_features_mean'],
                                                overview_features_std=config.ds_gen['overview_features_std'],
                                                add_daytime_feature=config.ds_gen['add_daytime_feature'],
                                                add_yeartime_feature=config.ds_gen['add_yeartime_feature'],
                                                add_linear_timeindex=config.ds_gen['add_linear_timeindex'],
                                                with_labels=True,
                                                pad_location_labels=config.ds_gen['pad_location_labels'],
                                                nonbinary_padding=config.ds_gen['nonbinary_padding'],
                                                train_val_split=0.8,
                                                input_stride=config.ds_gen['input_stride'],
                                                padding='zero', #!
                                                scale=True,
                                                unify_value_ranges=True,
                                                per_object_scaling=config.ds_gen['per_object_scaling'],
                                                node_class_multipliers={'ID':config.ds_gen['class_multiplier_ID'],
                                                                        'IK':config.ds_gen['class_multiplier_IK'],
                                                                        'AD':config.ds_gen['class_multiplier_AD'],
                                                                        'SS':1.0},
                                                input_history_steps=config.ds_gen['input_history_steps'],
                                                input_future_steps=config.ds_gen['input_future_steps'],
                                                input_dtype=np.float32,
                                                sort_inputs=True,
                                                seed=11,
                                                deepcopy=False)
        print('Trn-keys:', ds_gen.train_keys[:10])
        print('Val-keys:', ds_gen.val_keys[:10])
        train_ds, val_ds = ds_gen.get_datasets(2048,
                                               label_features=[f'{dir}_Node_Location_nb' for dir in directions],
                                               shuffle=True,
                                               stride=30,
                                               keep_label_stride=1)

        print(train_ds.element_spec)

        # =================================Model Creation & Training================================================

        model = prediction_models.Dense_NN(val_ds, 
                                           conv1d_layers=config.model['conv1d_layers'],
                                           dense_layers=config.model['dense_layers'],
                                           deep_layer_in_output=config.model['deep_layer_in_output'],
                                           lstm_layers=config.model['lstm_layers'],
                                           l2_reg=config.model['l2_reg'],
                                           input_dropout=config.model['input_dropout'],
                                           mixed_dropout_dense=config.model['mixed_dropout_dense'],
                                           mixed_dropout_cnn=config.model['mixed_dropout_cnn'],
                                           mixed_dropout_lstm=config.model['mixed_dropout_lstm'],
                                           mixed_batchnorm_cnn=config.model['mixed_batchnorm_cnn'],
                                           mixed_batchnorm_dense=config.model['mixed_batchnorm_dense'],
                                           mixed_batchnorm_before_relu=config.model['mixed_batchnorm_before_relu'],
                                           lr_scheduler=config.model['lr_scheduler'],
                                           output_type='regression',
                                           final_activation='linear',
                                           seed=0)
        model.summary()

        del train_ds
        del val_ds
        gc.collect()


        best_results = {'foo' : 'bar'}
        best_f2 = 0.0
        best_f2_threshold = -1.0
        best_f2_step = 0
        global_wandb_step = 0
        for strides, offset, keep_label, epochs in config.training['training_sets']:
            print(f"Strides: {strides} Offset: {offset} Keeping Label: {keep_label} Epochs: {epochs}")
            train_ds, val_ds = ds_gen.get_datasets(config.training['batch_size'],
                                                label_features=[f'{dir}_Node_Location_nb' for dir in directions],
                                                shuffle=True,
                                                only_ew_sk=False,
                                                stride=1 if keep_label else strides,
                                                keep_label_stride=1 if not keep_label else strides,
                                                stride_offset=offset,
                                                verbose=0)
        
            # train
            hist = model.fit(train_ds,
                            val_ds=val_ds,
                            epochs=epochs,
                            early_stopping=0,
                            target_metric='val_mse',
                            plot_hist=False,
                            callbacks=[WandbMetricsLogger(initial_global_step=global_wandb_step), ClearMemoryCallback()],
                            verbose=2)
            
            del train_ds
            del val_ds
        
            global_wandb_step += epochs
            print(f"----------------------Step: {global_wandb_step}-----------------------------")
            
            scores = localizer.perform_evaluation_pipeline(ds_gen,
                                        model,
                                        'val',
                                        gt_path = challenge_data_dir / 'train_labels.csv',
                                        output_dirs=directions,
                                        prediction_batches=5,
                                        thresholds = np.linspace(30.0, 85.0, 12),
                                        object_limit=None,
                                        with_initial_node=False,
                                        nodes_to_consider=config.training['nodes_to_consider'],
                                        verbose=0)
            print(f"--------------------------------------------------------------------")
            gc.collect()
            
            f2s = [score['F2'] for score in scores]
            best_local_f2_idx = np.argmax(f2s)
            dict_to_log = {}
            for key in scores[best_local_f2_idx].keys():
                dict_to_log[f'val/{key}'] = scores[best_local_f2_idx][key]
            if f2s[best_local_f2_idx] > best_f2:
                best_f2 = f2s[best_local_f2_idx]
                best_f2_threshold = scores[best_local_f2_idx]['Threshold']
                best_f2_step = global_wandb_step
                best_results = dict_to_log
            wandb.log(dict_to_log, commit=False)

        file_path = wandb.run.dir+"/model_" + wandb.run.id + ".hdf5"
        print(f"Saving model to \"{file_path}\"")
        model.model.save(file_path)
        wandb.save(file_path)
        if config.ds_gen['per_object_scaling'] == False:
            scaler_path = wandb.run.dir+"/scaler_" + wandb.run.id + ".pkl"
            print(f"Saving scaler to \"{scaler_path}\"")
            pickle.dump(ds_gen.scaler, open(scaler_path, 'wb'))
            wandb.save(scaler_path)

        # ====================================Evaluation===================================================

        print("-------------------------------")
        print('Best Results: ', best_results)
        wandb.run.summary['best_F2'] = best_f2
        wandb.run.summary['best_F2_threshold'] = best_f2_threshold
        wandb.run.summary['best_F2_step'] = best_f2_step

        train_object_limit = 500
        n_batches = 10
        print(f"Running train-evaluation on a subset of {train_object_limit} objects ({n_batches} batches) with threshold {60.0}:")
        evaluation_results_t = localizer.perform_evaluation_pipeline(ds_gen,
                                        model,
                                        'train',
                                        gt_path = challenge_data_dir / 'train_labels.csv',
                                        output_dirs=directions,
                                        prediction_batches=5,
                                        thresholds = [50.0],
                                        object_limit=train_object_limit,
                                        with_initial_node=False,
                                        nodes_to_consider=config.training['nodes_to_consider'],
                                        verbose=0)
        for key,value in evaluation_results_t[0].items():
            wandb.run.summary[f'train/{key}'] = value

        split_dataframes = None
        ds_gen = None
        del split_dataframes
        del ds_gen
        gc.collect()

        print("Done.")

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "best_F2"},
    "parameters": {
        "input_features" : {
            "parameters" : {
                'Eccentricity' : {"values": [True]},
                'Semimajor_Axis' : {"values": [True]},
                'Inclination' : {"values": [True]},
                'RAAN' : {"values": [False]},
                'Argument_of_Periapsis' : {"values": [True]},
                'True_Anomaly' : {"values": [False]},
                'Longitude' : {"values": [False]},
                'Latitude' : {"values": [True]},
            }
        },
        "feature_engineering" : {
            "parameters" : {
                'RAAN' : {"values": ['non']},
                'Argument_of_Periapsis' : {"values": ['sin']},
                'True_Anomaly' : {"values": ['sin']},
                'Longitude' : {"values": ['sin']},
                'Latitude' : {"values": ['non']},
            }
        },
        "ds_gen" : {
            "parameters" : {
            'overview_features_mean' : {"values" : [[]]},
            'overview_features_std' : {"values" : [[]]},
            "pad_location_labels" : {"values": [0]},
            "nonbinary_padding" : {"values": [
                                              [110.0, 70.0, 49.0, 34.0, 24.0, 12.0]
                                              ]},
            "input_stride" : {"values": [2]},
            "per_object_scaling" : {"values" : [False]},
            "add_daytime_feature" : {"values": [False]},
            "add_yeartime_feature" : {"values": [False]},
            "add_linear_timeindex" : {"values": [False]},
            "class_multiplier_ID" : {"values": [0.0]},
            "class_multiplier_IK" : {"values": [1.0]},
            "class_multiplier_AD" : {"values": [1.0]},
            "input_history_steps" : {"values": [128]},
            "input_future_steps" : {"values": [32]},
            }
        },
        "model" : {
            "parameters" : {
            "conv1d_layers" : {"values": [#[]
                                          #[[64,15,1,1,1],[64,15,1,1,1],[48,15,2,1,1]],
                                          [[64,7,1,1,1],[64,7,1,1,1],[48,7,2,1,1]],
                                          #[[48,8,2]],
                                          #[[48,8,3,1,1],[48,4,1,1,1],[48,3,1,1,1]],
                                          #[[48,4,1,1,1],[48,4,2,1,1],[48,3,1,1,1]],
                                          ]},
            "conv2d_layers" : {"values": [[]]},
            "dense_layers" : {"values": [[64,32]]},
            "deep_layer_in_output" : {"values": [False]},
            "lstm_layers" : {"values": [[]]},
            "l2_reg" : {"values": [0.00025]},
            "input_dropout" : {"values": [0.0]},
            "mixed_batchnorm_cnn" : {"values": [True]},
            "mixed_batchnorm_dense" : {"values": [True]},
            "mixed_batchnorm_before_relu" : {"values": [False]},
            "mixed_dropout_dense" : {"values": [0.05]},
            "mixed_dropout_cnn" : {"values": [0.05]},
            "mixed_dropout_lstm" : {"values": [0.0]},
            "lr_scheduler" : {"values": [[0.005]]},
            "seed" : {"values": [0]},
            }
        },
         "training" : {
            "parameters" : {
            "training_sets" : {"values": [
                                    #[[5,0,True,2],[5,0,True,2]],
                                    #[[8,0,True,20],[8,1,True,20],[8,2,True,20]],
                                    #[[7,0,True,20],[7,1,True,20],[7,2,True,20]],
                                    #[[6,0,True,20],[6,1,True,20],[6,2,True,20]],
                                    [[5,0,True,25],[5,1,True,25],[5,2,True,25]],
                                    ]
                                },
            "nodes_to_consider" : {"values": [['AD', 'IK']]
                                },
            "batch_size" : {"values": [2048]},
            "directions" : {"values" : [['EW', 'NS']]}
            }
        },
    },
    
}

##########################################################
# Start the actual sweep
wandb.login()
project="splid-challenge-localizer"
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
wandb.agent(sweep_id, project=project, function=parameter_sweep)