import pandas as pd
import numpy as np
import wandb
from wandb.keras import WandbMetricsLogger
from pathlib import Path
import gc

from base import datahandler, prediction_models, evaluation, utils, classifier



dirs=['EW', 'NS']

def parameter_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # =================================Data Loading & Preprocessing================================================

        # Load data
        challenge_data_dir = Path('dataset/phase_1_v2/')
        data_dir = challenge_data_dir / "train"
        labels_dir = challenge_data_dir / 'train_labels.csv'
        split_dataframes = datahandler.load_and_prepare_dataframes(data_dir, labels_dir)

        # Create Dataset
        utils.set_random_seed(42)

        non_transform_features = ['Eccentricity',
                                'Semimajor Axis (m)',
                                'Inclination (deg)',
                                'RAAN (deg)',
                                #'Argument of Periapsis (deg)',
                                #'True Anomaly (deg)',
                                'Latitude (deg)']
        diff_transform_features=[]
        sin_transform_features = []
        sin_cos_transform_features = []

        for key, value in config.feature_engineering.items():
            ft_name = key.replace('_', ' ') + ' (deg)'
            if value == 'non': non_transform_features += [ft_name]
            elif value == 'diff': diff_transform_features += [ft_name]
            elif value == 'sin': sin_transform_features += [ft_name]
            elif value == 'sin_cos': sin_cos_transform_features += [ft_name]
            else: print(f"Warning: unknown feature_engineering attribute \'{value}\' for feature {ft_name}")
        
        ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                                non_transform_features=non_transform_features,
                                                diff_transform_features=diff_transform_features,
                                                sin_transform_features=['Argument of Periapsis (deg)', 'True Anomaly (deg)', 'Longitude (deg)'], # TEMPORARY
                                                sin_cos_transform_features=sin_cos_transform_features,
                                                overview_features_mean=config.ds_gen['overview_features_mean'],
                                                overview_features_std=config.ds_gen['overview_features_std'],
                                                add_daytime_feature=config.ds_gen['add_daytime_feature'],
                                                add_yeartime_feature=config.ds_gen['add_yeartime_feature'],
                                                add_linear_timeindex=config.ds_gen['add_linear_timeindex'],
                                                with_labels=True,
                                                pad_location_labels=config.ds_gen['pad_location_labels'],
                                                nonbinary_padding=[100.0, 70.0, 49.0, 34.0, 24.0],
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
                                                input_history_steps=config.ds_gen['input_history_steps'],
                                                input_future_steps=config.ds_gen['input_future_steps'],
                                                input_dtype=np.float32,
                                                seed=181,
                                                deepcopy=False)
        
        print('Trn-keys:', ds_gen.train_keys)
        print('Val-keys:', ds_gen.val_keys)        
        

        train_ds, val_ds = ds_gen.get_datasets(batch_size=512,
                                               label_features=[f'{dir}_Type' for dir in dirs],
                                               with_identifier=False,
                                               shuffle=True,
                                               only_nodes=True,
                                               stride=config.ds_gen['stride'])
        
        print(train_ds.element_spec)

        model = prediction_models.Dense_NN(val_ds, 
                                           conv1d_layers=config.model['conv1d_layers'],
                                           dense_layers=config.model['dense_layers'],
                                           lstm_layers=config.model['lstm_layers'],
                                           l2_reg=config.model['l2_reg'],
                                           input_dropout=config.model['input_dropout'],
                                           mixed_dropout_dense=config.model['mixed_dropout_dense'],
                                           mixed_dropout_cnn=config.model['mixed_dropout_cnn'],
                                           mixed_dropout_lstm=config.model['mixed_dropout_lstm'],
                                           mixed_batchnorm=config.model['mixed_batchnorm'],
                                           lr_scheduler=config.model['lr_scheduler'],
                                           output_type='classification',
                                           seed=0)
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
                         epochs=350,
                         plot_hist=False,
                         early_stopping=40,
                         target_metric='val_EW_Type_accuracy' if len(dirs) > 1 else 'val_accuracy',
                         #class_weight=class_weights, 
                         callbacks=[WandbMetricsLogger()],
                         verbose=2)

        file_path = wandb.run.dir+"\\model_" + wandb.run.id + ".hdf5"
        print(f"Saving model to \"{file_path}\"")
        model.model.save(file_path)
        wandb.save(file_path)

        train_ds = None
        val_ds = None
        del train_ds
        del val_ds

        # ====================================Evaluation===================================================

        # perform final evaluation
        print("Running val-evaluation:")
        pred_df = classifier.create_prediction_df(ds_gen=ds_gen,
                                model=model,
                                train=False,
                                test=False,
                                model_outputs=[f'{dir}_Type' for dir in dirs],
                                object_limit=None,
                                only_nodes=True,
                                confusion_matrix=False,
                                prediction_batches=3,
                                verbose=2)
        ground_truth_df = pd.read_csv(challenge_data_dir / 'train_labels.csv')#.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
        oneshot_df = classifier.apply_one_shot_method(preds_df=pred_df, location_df=ground_truth_df, dirs=dirs)
        evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=oneshot_df)
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
                                model=model,
                                train=True,
                                test=False,
                                model_outputs=[f'{dir}_Type' for dir in dirs],
                                object_limit=None,
                                only_nodes=True,
                                verbose=2)
        ground_truth_df = pd.read_csv(challenge_data_dir / 'train_labels.csv')#.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
        oneshot_df = classifier.apply_one_shot_method(preds_df=pred_df, location_df=ground_truth_df, dirs=dirs)
        evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=oneshot_df)
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
        del split_dataframes
        del ds_gen
        gc.collect()

        print("Done.")

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "Precision"},
    #"run_cap":60,
    "parameters": {
        "feature_engineering" : {
            "parameters" : {
                'RAAN' : {"values": ['non']},
                'Argument_of_Periapsis' : {"values": ['sin']},
                'True_Anomaly' : {"values": ['sin']},
                'Longitude' : {"values": ['sin']},
            }
        },
        "ds_gen" : {
            "parameters" : {
            'overview_features_mean' : {"values" : [[],
                                                   ['Longitude (sin)', 'RAAN (deg)', 'Eccentricity']
                                                   ]},
            'overview_features_std' : {"values" : [[],
                                                   ['Latitude (deg)', 'Argument of Periapsis (sin)']
                                                   ]},
            "pad_location_labels" : {"values": [0]},
            "stride" : {"values": [1]},
            "input_stride" : {"values": [2]},
            "per_object_scaling" : {"values" : [False]},
            "add_daytime_feature" : {"values": [False]},
            "add_yeartime_feature" : {"values": [False]},
            "add_linear_timeindex" : {"values": [True, False]},
            "input_history_steps" : {"values": [32,16,1]},
            "input_future_steps" : {"values": [128]},
            }
        },
        "model" : {
            "parameters" : {
            "conv1d_layers" : {"values": [#[]
                                          [[32,6,1,1,1],[32,6,1,1,1],[32,6,1,1,1]]
                                          ]},
            "conv2d_layers" : {"values": [[]]},
            "dense_layers" : {"values": [[64,32]]},
            "lstm_layers" : {"values": [[]]},
            "l2_reg" : {"values": [0.001]},
            "input_dropout" : {"values": [0.0]},
            "mixed_batchnorm" : {"values": [False]},
            "mixed_dropout_dense" : {"values": [0.05]},
            "mixed_dropout_cnn" : {"values": [0.0, 0.05, 0.1]},
            "mixed_dropout_lstm" : {"values": [0.0]},
            "lr_scheduler" : {"values": [[0.008,300,0.9]]},
            "seed" : {"values": [0]},
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