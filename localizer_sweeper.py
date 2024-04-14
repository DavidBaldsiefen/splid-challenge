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


# Callback which helps with memory issues when working on slow computers
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
        challenge_data_dir = Path('dataset/phase_2/')
        data_dir_train_val = challenge_data_dir / "training"
        data_dir_test = challenge_data_dir / "test"
        labels_dir_train_val = challenge_data_dir / 'train_label.csv'
        labels_dir_test = challenge_data_dir / 'test_label.csv'
        split_dataframes_train_val = datahandler.load_and_prepare_dataframes(data_dir_train_val, labels_dir_train_val)
        split_dataframes_test = datahandler.load_and_prepare_dataframes(data_dir_test, labels_dir_test)

        directions=config.training['directions']
        print(f"Directions: {directions}")

        # Create Dataset
        utils.set_random_seed(42)

        non_transform_features =[]
        lowpass_features = []
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

        for key, value in config.lowpass_features.items():
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
                print(f"Replacing normal ft with lowpass: {ft_name}")
                lowpass_features += [ft_name]
                if ft_name in non_transform_features:
                    non_transform_features.remove(ft_name)

        for key, value in config.feature_engineering.items():
            ft_name = key.replace('_', ' ') + ' (deg)'
            if value == 'diff': diff_transform_features += [ft_name]
            elif value == 'sin': sin_transform_features += [ft_name]
            elif value == 'sin_cos': sin_cos_transform_features += [ft_name]
            elif value == 'non': idontknowwhatelsetodo=1# do nothing
            else: print(f"Warning: unknown feature_engineering attribute \'{value}\' for feature {ft_name}")

        ds_gen = datahandler.DatasetGenerator(train_val_df_dict=split_dataframes_train_val,#{df_k : split_dataframes[df_k] for df_k in list(split_dataframes.keys())[:900]},
                                              test_df_dict=split_dataframes_test,
                                              exclude_objects=[30, 113, 1012, 1383, 1385, 1386, 1471, 1473, 1474],
                                                non_transform_features=non_transform_features,
                                                diff_transform_features=diff_transform_features,
                                                legacy_diff_transform=config.ds_gen['legacy_diff_transform'],
                                                sin_transform_features=sin_transform_features,
                                                sin_cos_transform_features=sin_cos_transform_features,
                                                lowpass_features=lowpass_features,
                                                lowpass_filter_order=15,
                                                overview_features_mean=config.ds_gen['overview_features_mean'],
                                                overview_features_std=config.ds_gen['overview_features_std'],
                                                add_daytime_feature=config.ds_gen['add_daytime_feature'],
                                                add_yeartime_feature=config.ds_gen['add_yeartime_feature'],
                                                add_linear_timeindex=config.ds_gen['add_linear_timeindex'],
                                                linear_timeindex_as_overview=config.ds_gen['linear_timeindex_as_overview'],
                                                with_labels=True,
                                                pad_location_labels=config.ds_gen['pad_location_labels'],
                                                nonbinary_padding=config.ds_gen['nonbinary_padding'],
                                                train_val_split=0.95,
                                                input_stride=config.ds_gen['input_stride'],
                                                padding='zero', #!
                                                scale=True,
                                                unify_value_ranges=True,
                                                per_object_scaling=config.ds_gen['per_object_scaling'],
                                                node_class_multipliers={'ID':config.ds_gen['class_multiplier_ID'],
                                                                        'IK':config.ds_gen['class_multiplier_IK'],
                                                                        'AD':config.ds_gen['class_multiplier_AD'],
                                                                        'SS':0.0},
                                                input_history_steps=config.ds_gen['input_history_steps'],
                                                input_future_steps=config.ds_gen['input_future_steps'],
                                                input_dtype=np.float32,
                                                sort_input_features=True,
                                                seed=11,
                                                deepcopy=False)
        print('Trn-keys:', ds_gen.train_keys[:10])
        print('Val-keys:', ds_gen.val_keys[:10])
        datasets = ds_gen.get_datasets(2048,
                                        label_features=[f'{dir}_Node_Location_nb' for dir in directions],
                                        shuffle=True,
                                        stride=100,
                                        keep_label_stride=1)

        print(datasets['train'].element_spec)

        # =================================Model Creation & Training================================================

        model = prediction_models.Dense_NN(datasets['val'], 
                                           conv1d_layers=config.model['conv1d_layers'],
                                           dense_layers=config.model['dense_layers'],
                                           lstm_layers=config.model['lstm_layers'],
                                           cnn_lstm_order=config.model['cnn_lstm_order'],
                                           split_cnn=config.model['split_cnn'],
                                           split_dense=config.model['split_dense'],
                                           split_lstm=config.model['split_lstm'],
                                           l2_reg=config.model['l2_reg'],
                                           input_dropout=config.model['input_dropout'],
                                           mixed_dropout_dense=config.model['mixed_dropout_dense'],
                                           mixed_dropout_cnn=config.model['mixed_dropout_cnn'],
                                           mixed_dropout_lstm=config.model['mixed_dropout_lstm'],
                                           mixed_batchnorm_cnn=config.model['mixed_batchnorm_cnn'],
                                           mixed_batchnorm_dense=config.model['mixed_batchnorm_dense'],
                                           mixed_batchnorm_lstm=config.model['mixed_batchnorm_lstm'],
                                           mixed_batchnorm_before_relu=config.model['mixed_batchnorm_before_relu'],
                                           optimizer=config.model['optimizer'],
                                           lr_scheduler=config.model['lr_scheduler'],
                                           output_type='regression',
                                           final_activation='linear',
                                           seed=0)
        model.summary()

        del datasets
        gc.collect()


        best_results = {'foo' : 'bar'}
        best_f2 = 0.0
        best_f2_threshold = -1.0
        best_f2_step = 0
        global_wandb_step = 0
        for strides, offset, keep_label, epochs in config.training['training_sets']:
            print(f"Strides: {strides} Offset: {offset} Keeping Label: {keep_label} Epochs: {epochs}")
            datasets = ds_gen.get_datasets(config.training['batch_size'],
                                                label_features=[f'{dir}_Node_Location_nb' for dir in directions],
                                                convolve_input_stride=config.ds_gen['convolve_input_stride'],
                                                shuffle=True,
                                                only_ew_sk=False,
                                                stride=1 if keep_label else strides,
                                                keep_label_stride=1 if not keep_label else strides,
                                                stride_offset=offset,
                                                test_keys=[],
                                                verbose=0)
        
            # train
            hist = model.fit(datasets['train'],
                            val_ds=datasets['val'],
                            epochs=epochs,
                            early_stopping=0,
                            target_metric='val_loss',
                            save_best_only=False,
                            plot_hist=False,
                            callbacks=[WandbMetricsLogger(initial_global_step=global_wandb_step), ClearMemoryCallback()],
                            verbose=2)
            
            del datasets
        
            global_wandb_step += epochs
            print(f"----------------------Step: {global_wandb_step}-----------------------------")
            
            scores = localizer.perform_evaluation_pipeline(ds_gen,
                                        model.model,
                                        'val',
                                        gt_path = labels_dir_train_val,
                                        convolve_input_stride=config.ds_gen['convolve_input_stride'],
                                        output_dirs=directions,
                                        prediction_batch_size=96,
                                        thresholds = np.linspace(25.0, 70.0, 10),
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
            print(f"Scaler means & scale: {ds_gen.scaler.mean_} {ds_gen.scaler.scale_}")

        # ====================================Evaluation===================================================

        print("-------------------------------")
        print('Best Results: ', best_results)
        wandb.run.summary['best_F2'] = best_f2
        wandb.run.summary['best_F2_threshold'] = best_f2_threshold
        wandb.run.summary['best_F2_step'] = best_f2_step

        train_object_limit = 200
        print(f"Running train-evaluation on a subset of {train_object_limit} objects:")
        evaluation_results_train = localizer.perform_evaluation_pipeline(ds_gen,
                                        model.model,
                                        'train',
                                        gt_path = labels_dir_train_val,
                                        convolve_input_stride=config.ds_gen['convolve_input_stride'],
                                        output_dirs=directions,
                                        prediction_batch_size=96,
                                        thresholds = [50.0],
                                        object_limit=train_object_limit,
                                        with_initial_node=False,
                                        nodes_to_consider=config.training['nodes_to_consider'],
                                        verbose=0)
        for key,value in evaluation_results_train[0].items():
            wandb.run.summary[f'train/{key}'] = value

        train_object_limit = 96
        print(f"Running test-evaluation:")
        evaluation_results_test = localizer.perform_evaluation_pipeline(ds_gen,
                                        model.model,
                                        'test',
                                        gt_path = labels_dir_test,
                                        convolve_input_stride=config.ds_gen['convolve_input_stride'],
                                        output_dirs=directions,
                                        prediction_batch_size=96,
                                        thresholds = [50.0],
                                        object_limit=None,
                                        with_initial_node=False,
                                        nodes_to_consider=config.training['nodes_to_consider'],
                                        verbose=0)
        for key,value in evaluation_results_test[0].items():
            wandb.run.summary[f'test/{key}'] = value


        # Garbage collection which helps on low power machines
        split_dataframes_train_val = None
        split_dataframes_test = None
        ds_gen = None
        del split_dataframes_train_val
        del split_dataframes_test
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
                'RAAN' : {"values": [True]},
                'Argument_of_Periapsis' : {"values": [False]},
                'True_Anomaly' : {"values": [False]},
                'Longitude' : {"values": [False]},
                'Latitude' : {"values": [True]},
            }
        },
        "lowpass_features" : {
            "parameters" : {
                'Eccentricity' : {"values": [False]},
                'Semimajor_Axis' : {"values": [False]},
                'Inclination' : {"values": [False]},
                'RAAN' : {"values": [False]},
                'Argument_of_Periapsis' : {"values": [False]},
                'True_Anomaly' : {"values": [False]},
                'Longitude' : {"values": [False]},
                'Latitude' : {"values": [False]},
            }
        },
        "feature_engineering" : {
            "parameters" : {
                'Inclination' : {"values": ['diff']},
                'RAAN' : {"values": ['non']},
                'Argument_of_Periapsis' : {"values": ['sin']},
                'True_Anomaly' : {"values": ['diff']},
                'Longitude' : {"values": ['diff']},
                'Latitude' : {"values": ['non']},
            }
        },
        "ds_gen" : {
            "parameters" : {
            'overview_features_mean' : {"values" : [[]]},
            'overview_features_std' : {"values" : [#['Inclination (deg)'], 
                                                   []
                                                   ]},
            "pad_location_labels" : {"values": [0]},
            "nonbinary_padding" : {"values": [
                                              [100.0, 70.0, 49.0, 34.0, 24.0, 16.0],
                                              #[11.0, 7.0, 4.9, 3.4, 2.4, 1.2]
                                              ]},
            "input_stride" : {"values": [2,3,4]},
            "per_object_scaling" : {"values" : [False]},
            "add_daytime_feature" : {"values": [False]},
            "add_yeartime_feature" : {"values": [False]},
            "add_linear_timeindex" : {"values": [False]},
            "linear_timeindex_as_overview" : {"values": [True]},
            "convolve_input_stride" : {"values": [True]},
            "legacy_diff_transform" : {"values": [True, False]},
            "class_multiplier_ID" : {"values": [0.0]},
            "class_multiplier_IK" : {"values": [1.0]},
            "class_multiplier_AD" : {"values": [1.0]},
            "input_history_steps" : {"values": [128]},
            "input_future_steps" : {"values": [32]},
            }
        },
        "model" : {
            "parameters" : {
            "conv1d_layers" : {"values": [#[],
                                          #[[64,11,1,1,1],[64,11,1,1,1],[48,11,2,1,1]],
                                          #[[64,9,1,1,1],[64,9,1,1,1],[48,9,2,1,1]],
                                          #[[64,7,1,1,1],[64,7,1,1,1],[48,7,2,1,1]],
                                          [[64,7,1,1,1],[64,7,1,1,1],[48,7,2,1,1]],
                                          #[[64,7,1,1,1],[64,7,1,1,1],[48,7,1,1,1]],
                                          #[[64,13,12,1,1]],
                                          #[[64,23,20,1,1]],
                                          #[[64,7,2,1,1],[64,7,3,1,1]],
                                          #[[64,6,2,1,1]],
                                          #[[64,7,6,1,1]],
                                          #[[64,15,6,1,1]],
                                          #[[64,13,12,1,1]],
                                          #[[64,13,12,1,1]],
                                          #[[64,5,4,1,1]],
                                          #[[96,3,2,1,1]],
                                          #[[48,8,3,1,1],[48,4,1,1,1],[48,3,1,1,1]],
                                          #[[48,4,1,1,1],[48,4,2,1,1],[48,3,1,1,1]],
                                          ]},
            "dense_layers" : {"values": [[64,32]]},
            "lstm_layers" : {"values": [#[[48, True, 2, 1]],
                                        #[[32, True, 4, 1]],
                                        #[[96, True, 1, 1]],
                                        #[[64, True, 1, 1]],
                                        #[[48, True, 1, 1]],
                                        #[[128, True, 2, 1]],
                                        []
                                        ]},
            "cnn_lstm_order" : {"values" : ['lstm_cnn']},
            "split_cnn" : {"values" : [True]},
            "split_dense" : {"values" : [False]},
            "split_lstm" : {"values" : [True]},
            "l2_reg" : {"values": [0.00025]},
            "input_dropout" : {"values": [0.0]},
            "mixed_batchnorm_cnn" : {"values": [True]},
            "mixed_batchnorm_dense" : {"values": [True]},
            "mixed_batchnorm_lstm" : {"values": [True]},
            "mixed_batchnorm_before_relu" : {"values": [False]},
            "mixed_dropout_dense" : {"values": [0.05]},
            "mixed_dropout_cnn" : {"values": [0.05]},
            "mixed_dropout_lstm" : {"values": [0.0]},
            "lr_scheduler" : {"values": [[0.005],
                                         #[0.001],
                                         #[0.005, 2000, 0.9],
                                         #[0.002, 2000, 0.9],
                                         #[0.005, 1000, 0.9]
                                         ]},
            "optimizer" : {"values" : ['adam']},
            "seed" : {"values": [42]},
            }
        },
         "training" : {
            "parameters" : {
            "training_sets" : {"values": [
                                    #[[5,0,True,2],[5,0,True,2]],
                                    #[[8,0,True,20],[8,1,True,20],[8,2,True,20]],
                                    #[[7,0,True,20],[7,1,True,20],[7,2,True,20]],
                                    #[[6,0,True,28],[6,1,True,28],[6,2,True,28]],
                                    #[[5,0,True,30],[5,1,True,30],[5,2,True,30]],
                                    #[[5,1,True,80]],
                                    #[[4,0,True,50],[4,3,True,50]],
                                    #[[6,0,True,50],[6,3,True,50]],
                                    [[5,0,True,45],[5,3,True,45]],
                                    #[[4,0,True,28],[4,1,True,28],[4,2,True,28]],
                                    ]
                                },
            "nodes_to_consider" : {"values": [['AD', 'IK']]
                                },
            "batch_size" : {"values": [2048]},
            "directions" : {"values" : [['EW', 'NS'],
                                        #['EW'],
                                        #['NS']
                                        ]}
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