import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import pickle
import time
import sys

from base import utils, datahandler, classifier, localizer

DEBUG_MODE = False

if DEBUG_MODE:
    from base import evaluation
    print("Warning: Running in debug-mode, disable before submitting!")

LOCALIZER_EW_DIR = Path('models/ew_localizer.hdf5') if DEBUG_MODE else Path('/models/ew_localizer.hdf5')
SCALER_EW_DIR = Path('models/EW_localizer_scaler.pkl') if DEBUG_MODE else Path('/models/EW_localizer_scaler.pkl')

LOCALIZER_NS_DIR = Path('models/ns_localizer.hdf5') if DEBUG_MODE else Path('/models/ns_localizer.hdf5')
SCALER_NS_DIR = Path('models/NS_localizer_scaler.pkl') if DEBUG_MODE else Path('/models/NS_localizer_scaler.pkl')

CLASSIFIER_DIR = Path('models/ew_ns_classifier_oneshot.hdf5') if DEBUG_MODE else Path('/models/ew_ns_classifier_oneshot.hdf5')
SCALER_CLASSIFIER_DIR = Path('models/ew_ns_classifier_scaler_oneshot.pkl') if DEBUG_MODE else Path('/models/ew_ns_classifier_scaler_oneshot.pkl')

TEST_DATA_DIR = Path('dataset/val/') if DEBUG_MODE else Path('/dataset/test/')
TEST_PREDS_FP = Path('submission/submission.csv') if DEBUG_MODE else Path('/submission/submission.csv')

# Load Data
split_dataframes = datahandler.load_and_prepare_dataframes(TEST_DATA_DIR, labels_dir=None)
print(f"Loaded {len(split_dataframes.keys())} dataset files from \"{TEST_DATA_DIR}\". Creating dataset")

# ============================================================================================
# EW-Localization
ew_input_features = ['Eccentricity', 'Semimajor Axis (m)', 'Argument of Periapsis (deg)', 'Longitude (deg)', 'Altitude (m)']
ew_localizer_scaler = pickle.load(open(SCALER_EW_DIR, 'rb'))
ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      input_features=ew_input_features,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=1,
                                      padding='none',
                                      input_history_steps=12,
                                      input_future_steps=12,
                                      custom_scaler=ew_localizer_scaler,
                                      seed=69)

print(f"Predicting EW locations using model \"{LOCALIZER_EW_DIR}\"")
ew_localizer = tf.keras.models.load_model(LOCALIZER_EW_DIR)

ew_preds_df = localizer.create_prediction_df(ds_gen=ds_gen,
                                model=ew_localizer,
                                train=False,
                                test=True,
                                output_dirs=['EW'],
                                verbose=2)

ew_subm_df = localizer.postprocess_predictions(preds_df=ew_preds_df,
                                            dirs=['EW'],
                                            threshold=50.0,
                                            add_initial_node=True,
                                            clean_consecutives=True)

# NS-Localization
ns_input_features = ['Eccentricity', 'Semimajor Axis (m)', 'RAAN (deg)', 'Inclination (deg)', 'Latitude (deg)']
ns_localizer_scaler = pickle.load(open(SCALER_NS_DIR, 'rb'))
ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      input_features=ns_input_features,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=1,
                                      padding='none',
                                      input_history_steps=12,
                                      input_future_steps=12,
                                      custom_scaler=ns_localizer_scaler,
                                      seed=69)

print(f"Predicting NS locations using model \"{LOCALIZER_NS_DIR}\"")
ns_localizer = tf.keras.models.load_model(LOCALIZER_NS_DIR)

ns_preds_df = localizer.create_prediction_df(ds_gen=ds_gen,
                                model=ns_localizer,
                                train=False,
                                test=True,
                                output_dirs=['NS'],
                                verbose=2)

ns_subm_df = localizer.postprocess_predictions(preds_df=ns_preds_df,
                                            dirs=['NS'],
                                            threshold=90.0,
                                            add_initial_node=True,
                                            clean_consecutives=True)
print(ns_subm_df.loc[ns_subm_df['TimeIndex']!=0].head(10))

# Combine locations
df_locs = pd.concat([ew_subm_df, ns_subm_df]).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

print(f"#EW_Preds: {len(df_locs.loc[(df_locs['Direction'] == 'EW')])}")
print(f"#NS_Preds: {len(df_locs.loc[(df_locs['Direction'] == 'NS')])}")

# ============================================================================================
# Classification
classifier_scaler = pickle.load(open(SCALER_CLASSIFIER_DIR, 'rb'))
input_features_reduced_further = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)', 'Latitude (deg)', 'Longitude (deg)']

ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      input_features=input_features_reduced_further,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=2, #!
                                      padding='none',
                                      input_history_steps=1,
                                      input_future_steps=124,
                                      custom_scaler=classifier_scaler,
                                      seed=69)
print(f"Classifying using model \"{CLASSIFIER_DIR}\"")
classifier_model = tf.keras.models.load_model(CLASSIFIER_DIR)

pred_df = classifier.create_prediction_df(ds_gen=ds_gen,
                                model=classifier_model,
                                train=False,
                                test=True,
                                model_outputs=['EW_Type', 'NS_Type'],
                                object_limit=None,
                                verbose=2)

#majority_df = classifier.apply_majority_method(preds_df=pred_df, location_df=df_locs)
majority_df = classifier.apply_one_shot_method(preds_df=pred_df, location_df=df_locs)

# =====================================================================================================

# TEMPORARY: Remove NS detections, because they probably add way more FP than TP
df_reduced = majority_df.loc[(majority_df['TimeIndex'] == 0) | (majority_df['Direction'] == 'EW') | (majority_df['Direction'] == 'NS')]
# Only EW: 0.56 0.57 0.57?
# Save final results
results = df_reduced
print(f"Finished predictions, saving to \"{TEST_PREDS_FP}\"")
print(results.head(10))
results.to_csv(TEST_PREDS_FP, index=False)

if not DEBUG_MODE:
    print("Done. Sleeping for 6 minutes.")
    time.sleep(360) # TEMPORARY FIX TO OVERCOME EVALAI BUG
    print("Finished sleeping")
else:
    print("Evaluating...")
    ground_truth_df = pd.read_csv(Path('dataset/val_labels.csv'))
    evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=results)
    precision, recall, f2, rmse, total_tp, total_fp, total_fn = evaluator.score()
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F2: {f2:.2f}')
    print(f'RMSE: {float(rmse):.4}')
    print(f'TP: {total_tp} FP: {total_fp} FN: {total_fn}')
    print("Done.")