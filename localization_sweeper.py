import pandas as pd
import numpy as np
import wandb
from wandb.keras import WandbMetricsLogger
from pathlib import Path

from base import datahandler, prediction_models, evaluation, utils

# Load data
challenge_data_dir = Path('dataset/phase_1/')
data_dir = challenge_data_dir / "train"
labels_dir = challenge_data_dir / 'train_labels.csv'
split_dataframes = datahandler.load_and_prepare_dataframes(data_dir, labels_dir)

input_features = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)',
       'Argument of Periapsis (deg)', 'True Anomaly (deg)', 'Latitude (deg)',
       'Longitude (deg)', 'Altitude (m)']

ew_input_features = ['Eccentricity', 'Longitude (deg)', 'Latitude (deg)']
ns_input_features = ['Eccentricity', 'Semimajor Axis (m)', 'RAAN (deg)', 'Inclination (deg)', 'Latitude (deg)', 'Altitude (m)']

direction='EW'

label_features=[f'{direction}_Node_Location']


def parameter_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Create Dataset
        utils.set_random_seed(42)
        ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                                input_features=ew_input_features if direction=='EW' else ns_input_features,
                                                with_labels=True,
                                                pad_location_labels=config.ds_gen['pad_location_labels'],
                                                train_val_split=0.8,
                                                input_stride=config.ds_gen['input_stride'],
                                                padding=config.ds_gen['padding'],
                                                transform_features=config.ds_gen['transform_features'],
                                                input_history_steps=config.ds_gen['input_history_steps'], 
                                                input_future_steps=config.ds_gen['input_future_steps'], 
                                                seed=69)
        print('Input Features: ', ew_input_features if direction=='EW' else ns_input_features)
        print('Trn-keys:', ds_gen.train_keys)
        print('Val-keys:', ds_gen.val_keys)
        
        train_ds, val_ds = ds_gen.get_datasets(512, label_features=[f'{direction}_Node_Location'],
                                               shuffle=True,
                                               stride=config.ds_gen['stride'])

        print(train_ds.element_spec)

        model = prediction_models.Dense_NN(val_ds, 
                                           conv1d_layers=config.model['conv1d_layers'],
                                           dense_layers=config.model['dense_layers'],
                                           l2_reg=config.model['l2_reg'],
                                           input_dropout=config.model['input_dropout'],
                                           mixed_dropout=config.model['mixed_dropout'],
                                           lr_scheduler=config.model['lr_scheduler'],
                                           seed=0)
        model.summary()

        train_ds = train_ds.map(lambda x,y:(x,y[f'{direction}_Node_Location']))
        val_ds = val_ds.map(lambda x,y:(x,y[f'{direction}_Node_Location'])) # temporary fix to allow class weights
        
        # train
        hist = model.fit(train_ds, val_ds=val_ds,
                         epochs=30,
                         plot_hist=False,
                         class_weight={0: config.training['class_weight_0'], 1: config.training['class_weight_1']},
                         callbacks=[WandbMetricsLogger()],
                         verbose=2)

        file_path = wandb.run.dir+"\\model_" + wandb.run.id + ".hdf5"
        model.model.save(file_path)
        wandb.save(file_path)

        # perform final evaluation
        print("Running val-evaluation:")
        evaluation_results_v = evaluation.evaluate_localizer(ds_gen, split_dataframes=split_dataframes,
                                                             gt_path=challenge_data_dir / 'train_labels.csv',
                                                             model=model.model, train=False,
                                                             with_initial_node=False,
                                                             remove_consecutives=True,
                                                             direction=direction,
                                                             return_scores=True)
        for key,value in evaluation_results_v.items():
            wandb.define_metric(key, summary="max" if key in ['Precision', 'Recall', 'F2', 'TP'] else 'min')
            wandb.run.summary[f'{key}'] = value
        print("Running train-evaluation:")
        evaluation_results_t = evaluation.evaluate_localizer(ds_gen, split_dataframes=split_dataframes,
                                                             gt_path=challenge_data_dir / 'train_labels.csv',
                                                             model=model.model, train=True,
                                                             with_initial_node=False,
                                                             remove_consecutives=True,
                                                             direction=direction,
                                                             return_scores=True)
        for key,value in evaluation_results_t.items():
            wandb.define_metric(f'train_{key}', summary="max" if key in ['Precision', 'Recall', 'F2', 'TP'] else 'min')
            wandb.run.summary[f'train_{key}'] = value
        
        print("Done.")

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "Recall"},
    "parameters": {
        "ds_gen" : {
            "parameters" : {
            "pad_location_labels" : {"values": [0, 1, 2]},
            "stride" : {"values": [1]},
            "input_stride" : {"values": [1]},
            "padding" : {"values": ['none']},
            "transform_features" : {"values": [True]},
            "input_history_steps" : {"values": [24]},
            "input_future_steps" : {"values": [24]},
            }
        },
        "model" : {
            "parameters" : {
            "conv1d_layers" : {"values": [[],
                                          #[[64,16],[64,8]],
                                          ]},
            "dense_layers" : {"values": [[128,64,32]]},
            "l2_reg" : {"values": [0.0]},
            "input_dropout" : {"values": [0.0]},
            "mixed_dropout" : {"values": [0.0, 0.1]},
            "lr_scheduler" : {"values": [[]]},
            "seed" : {"values": [0]},
            }
        },
        "training" : {
            "parameters" : {
            "class_weight_0" : {"values": [1.0]},
            "class_weight_1" : {"values": [100.0, 150.0, 200.0]}
            }
        },
    },
    
}

# TODO: there used to be (?) a memory leak somewhere here
# TODO: deterministic training seems to be not so deterministic... possibly related to ds shuffle?
##########################################################
# Start the actual sweep
wandb.login()
project="splid-challenge"
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
wandb.agent(sweep_id, project=project, function=parameter_sweep)