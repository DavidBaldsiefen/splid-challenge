import wandb
from wandb.keras import WandbMetricsLogger
from pathlib import Path
#from pympler.tracker import SummaryTracker
#import tracemalloc
import gc
import numpy as np

from base import datahandler, prediction_models, utils, localizer


direction='NS'

def parameter_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # =================================Data Loading & Preprocessing================================================

        # Load data
        challenge_data_dir = Path('dataset/phase_1_v2/')
        data_dir = challenge_data_dir / "train"
        labels_dir = challenge_data_dir / 'train_labels.csv'
        split_dataframes = datahandler.load_and_prepare_dataframes(data_dir, labels_dir)

        print(f"Direction: {direction}")

        # Create Dataset
        utils.set_random_seed(42)

        non_transform_features = ['Eccentricity',
                                    'Semimajor Axis (m)',
                                    'Inclination (deg)',
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
                                                sin_transform_features=sin_transform_features,
                                                sin_cos_transform_features=sin_cos_transform_features,
                                                overview_features=config.ds_gen['overview_features'],
                                                add_daytime_feature=config.ds_gen['add_daytime_feature'],
                                                add_yeartime_feature=config.ds_gen['add_yeartime_feature'],
                                                with_labels=True,
                                                pad_location_labels=config.ds_gen['pad_location_labels'],
                                                nonbinary_padding=[100.0, 70.0, 49.0, 34.0, 24.0],
                                                train_val_split=0.8,
                                                input_stride=config.ds_gen['input_stride'],
                                                padding='zero', #!
                                                per_object_scaling=config.ds_gen['per_object_scaling'],
                                                node_class_multipliers={'ID':config.ds_gen['class_multiplier_ID'],
                                                                        'IK':config.ds_gen['class_multiplier_IK'],
                                                                        'AD':1.0,
                                                                        'SS':1.0},
                                                input_history_steps=config.ds_gen['input_history_steps'],
                                                input_future_steps=config.ds_gen['input_future_steps'],
                                                input_dtype=np.float32,
                                                seed=181,
                                                deepcopy=False)
        # print('Trn-keys:', ds_gen.train_keys)
        # print('Val-keys:', ds_gen.val_keys)
        train_ds, val_ds = ds_gen.get_datasets(1024,
                                               label_features=[f'{direction}_Node_Location_nb'],
                                               shuffle=True,
                                               stride=config.ds_gen['stride'],
                                               keep_label_stride=config.ds_gen['keep_label_stride'])

        print(train_ds.element_spec)

        # =================================Model Creation & Training================================================

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
                                           output_type='regression',
                                           final_activation='linear',
                                           seed=0)
        model.summary()
        
        # train
        hist = model.fit(train_ds,
                         val_ds=val_ds,
                         epochs=100,
                         early_stopping=15,
                         target_metric='val_loss',
                         plot_hist=False,
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
        n_batches = 10
        print(f"Running val-evaluation ({n_batches} batches):")

        preds_df = localizer.create_prediction_df(ds_gen=ds_gen,
                                model=model,
                                train=False,
                                test=False,
                                output_dirs=[direction],
                                object_limit=None,
                                prediction_batches=n_batches,
                                verbose=2)
        subm_df = localizer.postprocess_predictions(preds_df=preds_df,
                                            dirs=[direction],
                                            threshold=60.0,
                                            add_initial_node=True,
                                            clean_consecutives=True,
                                            deepcopy=False)

        evaluation_results_v = localizer.evaluate_localizer(subm_df=subm_df,
                                                        gt_path=challenge_data_dir / 'train_labels.csv',
                                                        object_ids=list(map(int, ds_gen.val_keys)),
                                                        dirs=[direction],
                                                        with_initial_node=False,
                                                        return_scores=True,
                                                        verbose=2)
        
        for key,value in evaluation_results_v.items():
            wandb.define_metric(key, summary="max" if key in ['Precision', 'Recall', 'F2', 'TP'] else 'min')
            wandb.run.summary[f'{key}'] = value

        train_object_limit = 500

        print(f"Running train-evaluation on a subset of {train_object_limit} objects ({n_batches} batches):")
        preds_df = localizer.create_prediction_df(ds_gen=ds_gen,
                                model=model,
                                train=True,
                                test=False,
                                output_dirs=[direction],
                                object_limit=train_object_limit,
                                prediction_batches=n_batches,
                                verbose=2)
        
        subm_df = localizer.postprocess_predictions(preds_df=preds_df,
                                            dirs=[direction],
                                            threshold=60.0,
                                            add_initial_node=True,
                                            clean_consecutives=True,
                                            deepcopy=False)
        

        evaluation_results_t = localizer.evaluate_localizer(subm_df=subm_df,
                                                        gt_path=challenge_data_dir / 'train_labels.csv',
                                                        object_ids=list(map(int, ds_gen.train_keys))[:train_object_limit],
                                                        dirs=[direction],
                                                        with_initial_node=False,
                                                        return_scores=True,
                                                        verbose=2)
        for key,value in evaluation_results_t.items():
            wandb.define_metric(f'train_{key}', summary="max" if key in ['Precision', 'Recall', 'F2', 'TP'] else 'min')
            wandb.run.summary[f'train_{key}'] = value

        split_dataframes = None
        ds_gen = None
        del split_dataframes
        del ds_gen
        gc.collect()

        print("Done.")

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "F2"},
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
            'overview_features' : {"values" : [['Longitude (sin)', 'RAAN (deg)', 'Eccentricity'],
                                                   ['Longitude (sin)', 'RAAN (deg)', 'Argument of Periapsis (sin)'],
                                                   ['Longitude (sin)', 'RAAN (deg)', 'Inclination (deg)'],
                                                   ['Longitude (sin)', 'RAAN (deg)', 'Semimajor Axis (m)']]},
            "pad_location_labels" : {"values": [0]},
            "stride" : {"values": [1]},
            "keep_label_stride" : {"values": [5]},
            "input_stride" : {"values": [4]},
            "per_object_scaling" : {"values" : [False]},
            "add_daytime_feature" : {"values": [False]},
            "add_yeartime_feature" : {"values": [False]},
            "class_multiplier_ID" : {"values": [1.5]}, #!!
            "class_multiplier_IK" : {"values": [1.0]},
            "input_history_steps" : {"values": [128]},
            "input_future_steps" : {"values": [128]},
            }
        },
        "model" : {
            "parameters" : {
            "conv1d_layers" : {"values": [#[]
                                          [[64,6],[64,6],[48,6]]
                                          ]},
            "conv2d_layers" : {"values": [[]]},
            "dense_layers" : {"values": [[64,64]]},
            "lstm_layers" : {"values": [[]]},
            "l2_reg" : {"values": [0.00025]},
            "input_dropout" : {"values": [0.0]},
            "mixed_batchnorm" : {"values": [True]},
            "mixed_dropout_dense" : {"values": [0.05]},
            "mixed_dropout_cnn" : {"values": [0.05]},
            "mixed_dropout_lstm" : {"values": [0.0]},
            "lr_scheduler" : {"values": [[0.003,3000,0.9]]},
            "seed" : {"values": [0]},
            }
        },
    },
    
}

# TODO: there used to be (?) a memory leak somewhere here
##########################################################
# Start the actual sweep
wandb.login()
project="splid-challenge-localizer"
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
wandb.agent(sweep_id, project=project, function=parameter_sweep)