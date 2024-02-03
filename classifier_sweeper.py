import pandas as pd
import numpy as np
import wandb
from wandb.keras import WandbMetricsLogger
from pathlib import Path
import gc

from base import datahandler, prediction_models, evaluation, utils, classifier

# Load data
challenge_data_dir = Path('dataset/phase_1_v2/')
data_dir = challenge_data_dir / "train"
labels_dir = challenge_data_dir / 'train_labels.csv'
split_dataframes = datahandler.load_and_prepare_dataframes(data_dir, labels_dir)

input_features = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)',
       'Argument of Periapsis (deg)', 'True Anomaly (deg)', 'Latitude (deg)',
       'Longitude (deg)', 'Altitude (m)']

input_features = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)', 'Latitude (deg)', 'Longitude (deg)']

label_features=['EW_Type', 'NS_Type']

def parameter_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Create Dataset
        utils.set_random_seed(42)
        
        ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      input_features=input_features,
                                      with_labels=True,
                                      train_val_split=0.8,
                                      input_stride=config.ds_gen['input_stride'],
                                      transform_features=config.ds_gen['transform_features'],
                                      padding='zero',
                                      scale=True,
                                      per_object_scaling=config.ds_gen['per_object_scaling'],
                                      input_history_steps=1,
                                      input_future_steps=config.ds_gen['input_future_steps'],
                                      seed=69)
        
        print('Trn-keys:', ds_gen.train_keys)
        print('Val-keys:', ds_gen.val_keys)        
        

        train_ds, val_ds = ds_gen.get_datasets(batch_size=256,
                                               label_features=['EW_Type', 'NS_Type'],
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
                                           lr_scheduler=config.model['lr_scheduler'],
                                           mixed_batchnorm=config.model['mixed_batchnorm'],
                                           seed=0)
        model.summary()

        # train
        hist = model.fit(train_ds, val_ds=val_ds, epochs=300, plot_hist=False, callbacks=[WandbMetricsLogger()], verbose=2)

        file_path = wandb.run.dir+"\\model_" + wandb.run.id + ".hdf5"
        model.model.save(file_path)
        wandb.save(file_path)

        # perform final evaluation
        print("Running val-evaluation:")
        pred_df = classifier.create_prediction_df(ds_gen=ds_gen,
                                model=model,
                                train=False,
                                test=False,
                                model_outputs=['EW_Type', 'NS_Type'],
                                object_limit=None,
                                only_nodes=True,
                                verbose=2)
        ground_truth_df = pd.read_csv(challenge_data_dir / 'train_labels.csv')#.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
        oneshot_df = classifier.apply_one_shot_method(preds_df=pred_df, location_df=ground_truth_df)
        evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=oneshot_df)
        precision, recall, f2, rmse, total_tp, total_fp, total_fn = evaluator.score()
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
                                model_outputs=['EW_Type', 'NS_Type'],
                                object_limit=None,
                                only_nodes=True,
                                verbose=2)
        ground_truth_df = pd.read_csv(challenge_data_dir / 'train_labels.csv')#.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
        oneshot_df = classifier.apply_one_shot_method(preds_df=pred_df, location_df=ground_truth_df)
        evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=oneshot_df)
        precision, recall, f2, rmse, total_tp, total_fp, total_fn = evaluator.score()
        wandb.define_metric('train_Precision', summary="max")
        wandb.define_metric('train_TP', summary="max")
        wandb.define_metric('train_FP', summary="min")
        wandb.run.summary['train_Precision'] = precision
        wandb.run.summary['train_TP'] = total_tp
        wandb.run.summary['train_FP'] = total_fp
        print(f"TRN: P: {precision:.3f} TP: {total_tp} FP: {total_fp}")
        
        del ds_gen
        del train_ds
        del val_ds
        gc.collect()
        print("Done.")

sweep_configuration = {
    "method": "grid",
    "metric": {"goal": "maximize", "name": "Precision"},
    "parameters": {
        "ds_gen" : {
            "parameters" : {
            "stride" : {"values": [1]},
            "input_stride" : {"values": [1,2]},
            "transform_features" : {"values": [True]},
            "input_future_steps" : {"values": [128,64,32]},
            "per_object_scaling" : {"values": [True, False]}
            }
        },
        "model" : {
            "parameters" : {
            "conv1d_layers" : {"values": [
                                          [[32,6],[32,6],[32,6]],
                                          
                                          ]},
            "lstm_layers" : {"values": [[],
                                        #[24,24],
                                        #[48,48],
                                          ]},
            "dense_layers" : {"values": [[16,8]]},
            "l2_reg" : {"values": [0.0002]},
            "input_dropout" : {"values": [0.0]},
            "mixed_dropout_dense" : {"values": [0.5]},
            "mixed_dropout_cnn" : {"values": [0.2]},
            "mixed_dropout_lstm" : {"values": [0.0]},
            "mixed_batchnorm" : {"values": [False]},
            "lr_scheduler" : {"values": [[0.005, 600, 0.9],
                                         #[0.005, 400, 0.8],
                                         ]},
            "seed" : {"values": [42]},
            }
        },
    },
    
}

# TODO: Somehow collect information on where the FP/FN occur, which nodes etc
##########################################################
# Start the actual sweep
wandb.login()
project="splid-challenge-v2"
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
wandb.agent(sweep_id, project=project, function=parameter_sweep)