import pandas as pd
import numpy as np
import wandb
from wandb.keras import WandbMetricsLogger
from pathlib import Path
import gc
import pickle
import copy

from base import datahandler, prediction_models, evaluation, utils, classifier



def parameter_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # =================================Data Loading & Preprocessing================================================

        # Load data
        challenge_data_dir = Path('dataset/phase_1_v3/')
        data_dir = challenge_data_dir / "train"
        labels_dir = challenge_data_dir / 'train_labels.csv'
        split_dataframes = datahandler.load_and_prepare_dataframes(data_dir, labels_dir)

        dirs=config.training['directions']
        print(f"Directions: {dirs}")

        # Create Dataset
        utils.set_random_seed(42)

        non_transform_features =[]
        highpass_features = []
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

        for key, value in config.highpass_features.items():
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
                print(f"Replacing normal ft with highpass: {ft_name}")
                highpass_features += [ft_name]
                if ft_name in non_transform_features:
                    non_transform_features.remove(ft_name)

        for key, value in config.feature_engineering.items():
            ft_name = key
            if key == 'Eccentricity': ft_name = 'Eccentricity'
            elif key == 'Semimajor_Axis': ft_name = 'Semimajor Axis (m)'
            elif key == 'Inclination': ft_name = 'Inclination (deg)'
            elif key == 'RAAN': ft_name = 'RAAN (deg)'
            elif key == 'Argument_of_Periapsis': ft_name = 'Argument of Periapsis (deg)'
            elif key == 'True_Anomaly': ft_name = 'True Anomaly (deg)'
            elif key == 'Longitude': ft_name = 'Longitude (deg)'
            elif key == 'Latitude': ft_name = 'Latitude (deg)'
            if value == 'diff': diff_transform_features += [ft_name]
            elif value == 'sin': sin_transform_features += [ft_name]
            elif value == 'sin_cos': sin_cos_transform_features += [ft_name]
            elif value == 'non': idontknowwhatelsetodo=1# do nothing
            else: print(f"Warning: unknown feature_engineering attribute \'{value}\' for feature {ft_name}")
        
        ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                              exclude_objects=[9, 10, 13, 19, 30, 113, 1012, 1383, 1385, 1386, 1471, 1473, 1474],
                                                non_transform_features=non_transform_features,
                                                diff_transform_features=diff_transform_features,
                                                sin_transform_features=sin_transform_features,
                                                sin_cos_transform_features=sin_cos_transform_features,
                                                highpass_features=highpass_features,
                                                overview_features_mean=config.ds_gen['overview_features_mean'],
                                                overview_features_std=config.ds_gen['overview_features_std'],
                                                add_daytime_feature=config.ds_gen['add_daytime_feature'],
                                                add_yeartime_feature=config.ds_gen['add_yeartime_feature'],
                                                add_linear_timeindex=config.ds_gen['add_linear_timeindex'],
                                                linear_timeindex_as_overview=config.ds_gen['linear_timeindex_as_overview'],
                                                with_labels=True,
                                                pad_location_labels=config.ds_gen['pad_location_labels'],
                                                nonbinary_padding=[],
                                                train_val_split=0.8,
                                                input_stride=config.ds_gen['input_stride'],
                                                padding='zero',
                                                scale=True,
                                                unify_value_ranges=True,
                                                per_object_scaling=config.ds_gen['per_object_scaling'],
                                                node_class_multipliers={'ID':1.0,
                                                                        'IK':1.0,
                                                                        'AD':1.0,
                                                                        'SS':1.0},
                                                nodes_to_include_as_locations=config.ds_gen['nodes_to_include_as_locations'],
                                                input_history_steps=config.ds_gen['input_history_steps'],
                                                input_future_steps=config.ds_gen['input_future_steps'],
                                                input_dtype=np.float32,
                                                sort_inputs=True,
                                                seed=11,
                                                deepcopy=False)
        
        print('Trn-keys:', ds_gen.train_keys[:10])
        print('Val-keys:', ds_gen.val_keys[:10])
        

        train_ds, val_ds = ds_gen.get_datasets(batch_size=config.training['batch_size'],
                                               label_features=[f'{dir}_Type' for dir in dirs],
                                               with_identifier=False,
                                               overview_as_second_input=config.model['overview_as_second_input'],
                                               shuffle=True,
                                               only_nodes=True if config.ds_gen['keep_label_stride'] <= 1 else False,
                                               stride=config.ds_gen['stride'],
                                               keep_label_stride=config.ds_gen['keep_label_stride'],
                                               stride_offset=0 if config.ds_gen['keep_label_stride'] <= 1 else 250,
                                               verbose=0)
        
        print(train_ds.element_spec)

        model = prediction_models.Dense_NN(val_ds, 
                                           conv1d_layers=config.model['conv1d_layers'],
                                           dense_layers=config.model['dense_layers'],
                                           lstm_layers=config.model['lstm_layers'],
                                           cnn_lstm_order=config.model['cnn_lstm_order'],
                                           split_cnn=config.model['split_cnn'],
                                           split_dense=config.model['split_dense'],
                                           split_lstm=config.model['split_lstm'],
                                           l1_reg=config.model['l1_reg'],
                                           l2_reg=config.model['l2_reg'],
                                           input_dropout=config.model['input_dropout'],
                                           mixed_dropout_dense=config.model['mixed_dropout_dense'],
                                           mixed_dropout_cnn=config.model['mixed_dropout_cnn'],
                                           mixed_dropout_lstm=config.model['mixed_dropout_lstm'],
                                           mixed_batchnorm_cnn=config.model['mixed_batchnorm_cnn'],
                                           mixed_batchnorm_lstm=config.model['mixed_batchnorm_lstm'],
                                           mixed_batchnorm_dense=config.model['mixed_batchnorm_dense'],
                                           mixed_batchnorm_before_relu=config.model['mixed_batchnorm_before_relu'],
                                           lr_scheduler=config.model['lr_scheduler'],
                                           output_type='classification',
                                           seed=config.model['seed'])
        model.summary()

        # class_weights ={
        #     0: config.class_weights['CK'],
        #     1: config.class_weights['EK'],
        #     2: config.class_weights['HK'],
        #     3: config.class_weights['NK']}
        
        # fix to allow class weights
        # train_ds= train_ds.map(lambda x,y:(x,y[f'NS_Type']))
        # val_ds = val_ds.map(lambda x,y:(x,y[f'NS_Type']))

        # train
        hist = model.fit(train_ds,
                         val_ds=val_ds,
                         epochs=500,
                         plot_hist=False,
                         early_stopping=69,
                         save_best_only=True,
                         target_metric=config.training['target_metric'] if len(dirs) > 1 else 'val_accuracy',
                         #class_weight=class_weights, 
                         callbacks=[WandbMetricsLogger()],
                         verbose=2)

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

        train_ds = None
        val_ds = None
        del train_ds
        del val_ds

        # ====================================Evaluation===================================================

        # perform final evaluation
        print("Running val-evaluation:")
        pred_df = classifier.create_prediction_df(ds_gen=ds_gen,
                                model=model.model,
                                train=False,
                                test=False,
                                model_outputs=[f'{dir}_Type' for dir in dirs],
                                object_limit=None,
                                only_nodes=True,
                                confusion_matrix=False,
                                prediction_batches=4,
                                verbose=2)
        ground_truth_df = pd.read_csv(challenge_data_dir / 'train_labels.csv')#.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
        ground_truth_eval_df = pd.read_csv(challenge_data_dir / 'train_labels.csv')
        ground_truth_df.loc[ground_truth_df['Node'] != 'ID', 'Node'] = 'UNKNOWN'
        ground_truth_df.loc[ground_truth_df['Node'] != 'ID', 'Type'] = 'UNKNOWN'
        typed_df = classifier.fill_unknown_types_based_on_preds(pred_df, ground_truth_df, dirs=dirs)
        classified_df = classifier.fill_unknwon_nodes_based_on_type(typed_df, dirs=dirs)
        evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_eval_df, participant=classified_df, ignore_nodes=True)


        precision, recall, f2, rmse, total_tp, total_fp, total_fn, total_df = evaluator.score()
        wandb.define_metric('Precision', summary="max")
        wandb.define_metric('TP', summary="max")
        wandb.define_metric('FP', summary="min")
        wandb.run.summary['Precision'] = precision
        wandb.run.summary['TP'] = total_tp
        wandb.run.summary['FP'] = total_fp
        print(f"VAL: P: {precision:.3f} TP: {total_tp} FP: {total_fp}")

        print("Running train-evaluation:")
        pred_df = classifier.create_prediction_df(ds_gen=ds_gen,
                                model=model.model,
                                train=True,
                                test=False,
                                model_outputs=[f'{dir}_Type' for dir in dirs],
                                object_limit=None,
                                only_nodes=True,
                                prediction_batches=5,
                                verbose=2)
        ground_truth_df = pd.read_csv(challenge_data_dir / 'train_labels.csv')#.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
        ground_truth_eval_df = pd.read_csv(challenge_data_dir / 'train_labels.csv')
        ground_truth_df.loc[ground_truth_df['Node'] != 'ID', 'Type'] = 'UNKNOWN'
        ground_truth_df.loc[ground_truth_df['Node'] != 'ID', 'Node'] = 'UNKNOWN'
        typed_df = classifier.fill_unknown_types_based_on_preds(pred_df, ground_truth_df, dirs=dirs)
        classified_df = classifier.fill_unknwon_nodes_based_on_type(typed_df, dirs=dirs)
        evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_eval_df, participant=classified_df, ignore_nodes=True)
        precision, recall, f2, rmse, total_tp, total_fp, total_fn, total_df = evaluator.score()
        wandb.define_metric('train_Precision', summary="max")
        wandb.define_metric('train_TP', summary="max")
        wandb.define_metric('train_FP', summary="min")
        wandb.run.summary['train_Precision'] = precision
        wandb.run.summary['train_TP'] = total_tp
        wandb.run.summary['train_FP'] = total_fp
        print(f"TRN: P: {precision:.3f} TP: {total_tp} FP: {total_fp}")
        
        split_dataframes = None
        ds_gen = None
        del typed_df
        del classified_df
        del evaluator
        del split_dataframes
        del ds_gen
        gc.collect()

        print("Done.")

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "Precision"},
    #"run_cap":60,
    "parameters": {
       "input_features" : {
            "parameters" : {
                'Eccentricity' : {"values": [False]},
                'Semimajor_Axis' : {"values": [False]},
                'Inclination' : {"values": [False]},
                'RAAN' : {"values": [True]},
                'Argument_of_Periapsis' : {"values": [False]},
                'True_Anomaly' : {"values": [False]},
                'Longitude' : {"values": [False]},
                'Latitude' : {"values": [True]},
            }
        },
        "highpass_features" : {
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
                'Eccentricity' : {"values": ['diff']},
                'Semimajor_Axis' : {"values": ['diff']},
                'Inclination' : {"values": ['diff']},
                'RAAN' : {"values": ['non']},
                'Argument_of_Periapsis' : {"values": ['sin']},
                'True_Anomaly' : {"values": ['diff']},
                'Longitude' : {"values": ['sin']},
                'Latitude' : {"values": ['non']},
            }
        },
        "ds_gen" : {
            "parameters" : {
            'overview_features_mean' : {"values" : [[],
                                                   #['Longitude (sin)', 'RAAN (deg)', 'Eccentricity']
                                                   ]},
            'overview_features_std' : {"values" : [#[],
                                                   ['Longitude (sin)', 'Latitude (deg)', 'Argument of Periapsis (sin)', 'Inclination (deg)'],
                                                   #[],
                                                   #['Longitude (sin)']
                                                   ]},
            "pad_location_labels" : {"values": [0]},
            "nodes_to_include_as_locations" : {"values": [#['SS', 'AD', 'IK', 'ID'],
                                                          ['SS', 'AD', 'IK'],
                                                          ]},
            "stride" : {"values": [1]},
            "keep_label_stride" : {"values": [1000]}, # if 1, keep only labels
            "input_stride" : {"values": [2]},
            "per_object_scaling" : {"values" : [False]},
            "add_daytime_feature" : {"values": [False]},
            "add_yeartime_feature" : {"values": [False]},
            "add_linear_timeindex" : {"values": [True]},
            "linear_timeindex_as_overview" : {"values": [True, False]},
            "input_history_steps" : {"values": [32]},
            "input_future_steps" : {"values": [256]},
            }
        },
        "model" : {
            "parameters" : {
            "conv1d_layers" : {"values": [[],
                                          #[[128,9,1,2,1],[128,9,1,1,1],[64,9,3,1,1]],
                                          #[[128,7,1,1,1],[128,7,1,1,1],[64,7,2,1,1]],
                                          #[[64,6,6,1,1]],
                                          #[[64,3,1,1,1]],
                                          #[[64,11,6,1,1]],
                                          #[[64,7,1,1,1],[64,7,2,1,1]],
                                          #[[128,7,1,1,1],[128,7,1,1,1],[64,14,4,1,1]],
                                          #[[128,9,1,2,1],[128,9,1,1,1],[64,9,3,1,1]],
                                          #[[64,16,1,2,1],[64,13,1,1,1],[48,8,1,1,1]],
                                          ]},
            "dense_layers" : {"values": [[64,48]]},
            "lstm_layers" : {"values": [#[[32, True, 4]],
                                        #[[48, True, 4]],
                                        #[[64, True, 1]],
                                        #[[64, True, 5]],
                                        #[[96, True, 6]],
                                        #[[64, True, 8, 1]],
                                        #[[64, True, 1, 8]],
                                        #[[96, True, 1, 8]],
                                        [[96, True, 8, 1]],
                                        #[[128, True, 10]],
                                        ]},
            "cnn_lstm_order" : {"values" : ['lstm_cnn']},
            "split_cnn" : {"values" : [False]},
            "split_dense" : {"values" : [False]},
            "split_lstm" : {"values" : [True]},
            "overview_as_second_input" : {"values" : [False]},
            "l1_reg" : {"values": [0.0]},
            "l2_reg" : {"values": [0.00015]},
            "input_dropout" : {"values": [0.0]},
            "mixed_batchnorm_cnn" : {"values": [False]},
            "mixed_batchnorm_dense" : {"values": [True]},
            "mixed_batchnorm_lstm" : {"values": [False]},
            "mixed_batchnorm_before_relu" : {"values": [False]},
            "mixed_dropout_dense" : {"values": [0.15]},
            "mixed_dropout_cnn" : {"values": [0.0]},
            "mixed_dropout_lstm" : {"values": [0.2]},
            "lr_scheduler" : {"values": [[0.0035, 300, 0.9]]},
            "optimizer" : {"values": ['adam']},
            "seed" : {"values": [42,69]},
            }
        },
        "training" : {
            "parameters" : {
            "batch_size" : {"values": [128]},
            "directions" : {"values" : [['EW', 'NS'],
                                        #['EW'],
                                        #['NS']
                                        ]},
            "target_metric" : {"values" : ['val_EW_Type_accuracy']}
            }
        },
        # "class_weights" : {
        #     "parameters" : {
        #         "CK" : {"distribution": "uniform",
        #                     "min": 0.25,
        #                     "max": 3.0},
        #         "EK" : {"distribution": "uniform",
        #                     "min": 0.25,
        #                     "max": 3.0},
        #         "HK" : {"distribution": "uniform",
        #                     "min": 0.25,
        #                     "max": 3.0},
        #         "NK" : {"distribution": "uniform",
        #                     "min": 0.25,
        #                     "max": 3.0},
        #     }
        # }
    },
    
}

# TODO: Somehow collect information on where the FP/FN occur, which nodes etc
##########################################################
# Start the actual sweep
wandb.login()
project="splid-challenge-v2"
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
wandb.agent(sweep_id, project=project, function=parameter_sweep)