import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import pickle
import time
import gc

from base import utils, datahandler, classifier, localizer

DEBUG_MODE = False

if DEBUG_MODE:
    from base import evaluation
    print("Warning: Running in debug-mode, disable before submitting!")

# TODO: move if/else into the Path
LOCALIZER_EW_DIR = Path(('' if DEBUG_MODE else '/') + 'models/ew_localizer_cnn.hdf5')
SCALER_EW_DIR = Path(('' if DEBUG_MODE else '/') + 'models/EW_localizer_scaler_cnn.pkl')

LOCALIZER_NS_DIR = Path(('' if DEBUG_MODE else '/') + 'models/ns_localizer_cnn.hdf5')
SCALER_NS_DIR = Path(('' if DEBUG_MODE else '/') + 'models/NS_localizer_scaler_cnn.pkl')

CLASSIFIER_DIR = Path(('' if DEBUG_MODE else '/') + 'models/ew_ns_classifier_oneshot_cnn.hdf5')
SCALER_CLASSIFIER_DIR = Path(('' if DEBUG_MODE else '/') + 'models/ew_ns_classifier_scaler_oneshot_cnn.pkl')

TEST_DATA_DIR = Path(('' if DEBUG_MODE else '/') + 'dataset/test/')
TEST_PREDS_FP = Path(('' if DEBUG_MODE else '/') + 'submission/submission.csv')

# Load Data
split_dataframes = datahandler.load_and_prepare_dataframes(TEST_DATA_DIR, labels_dir=None)
print(f"Loaded {len(split_dataframes.keys())} dataset files from \"{TEST_DATA_DIR}\". Creating dataset")

# =================================LOCALIZATION==========================================
# TODO: add some safeguard in case there are too many detections in one object
#-----------------------------------EW-------------------------------
ew_input_features = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)',
                        'Argument of Periapsis (deg)', 'True Anomaly (deg)', 'Latitude (deg)',
                        'Longitude (deg)', 'Altitude (m)']
ew_localizer_scaler = pickle.load(open(SCALER_EW_DIR, 'rb'))
ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      input_features=ew_input_features,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=4,
                                      padding='none',
                                      input_history_steps=64,
                                      input_future_steps=24,
                                      per_object_scaling=True,
                                      custom_scaler=None,
                                      seed=69)

print(f"Predicting EW locations using model \"{LOCALIZER_EW_DIR}\" and scaler \"{SCALER_EW_DIR}\"")
ew_localizer = tf.keras.models.load_model(LOCALIZER_EW_DIR)

ew_preds_df = localizer.create_prediction_df(ds_gen=ds_gen,
                                model=ew_localizer,
                                train=False,
                                test=True,
                                output_dirs=['EW'],
                                verbose=2)

ew_subm_df = localizer.postprocess_predictions(preds_df=ew_preds_df,
                                            dirs=['EW'],
                                            threshold=65.0,
                                            add_initial_node=True,
                                            clean_consecutives=True)
gc.collect()
#-----------------------------------NS-------------------------------
ns_input_features = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)',
                        'Argument of Periapsis (deg)', 'True Anomaly (deg)', 'Latitude (deg)',
                        'Longitude (deg)', 'Altitude (m)']
ns_localizer_scaler = pickle.load(open(SCALER_NS_DIR, 'rb'))
ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      input_features=ns_input_features,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=4,
                                      padding='none',
                                      input_history_steps=64,
                                      input_future_steps=24,
                                      per_object_scaling=True,
                                      custom_scaler=None,
                                      seed=69)

print(f"Predicting NS locations using model \"{LOCALIZER_NS_DIR}\" and scaler \"{SCALER_NS_DIR}\"")
ns_localizer = tf.keras.models.load_model(LOCALIZER_NS_DIR)

ns_preds_df = localizer.create_prediction_df(ds_gen=ds_gen,
                                model=ns_localizer,
                                train=False,
                                test=True,
                                output_dirs=['NS'],
                                verbose=2)

ns_subm_df = localizer.postprocess_predictions(preds_df=ns_preds_df,
                                            dirs=['NS'],
                                            threshold=55.0,
                                            add_initial_node=True,
                                            clean_consecutives=True)
gc.collect()
#--------------------------------COMBINE-------------------------------
df_locs = pd.concat([ew_subm_df, ns_subm_df]).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

print(f"#EW_Preds: {len(df_locs.loc[(df_locs['Direction'] == 'EW')])}")
print(f"#NS_Preds: {len(df_locs.loc[(df_locs['Direction'] == 'NS')])}")

# =================================CLASSIFICATION==========================================

classifier_scaler = pickle.load(open(SCALER_CLASSIFIER_DIR, 'rb'))
input_features_reduced_further = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)', 'Latitude (deg)', 'Longitude (deg)']

ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      input_features=input_features_reduced_further,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=2,
                                      padding='zero',
                                      input_history_steps=1,
                                      input_future_steps=128,
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

# Use this to (temporarily!) remove certain parts
df_reduced = majority_df.loc[(majority_df['TimeIndex'] == 0) | (majority_df['Direction'] == 'EW') | (majority_df['Direction'] == 'NS')]

# Save final results
results = df_reduced
print(results.head(5))

if not DEBUG_MODE:
    print(f"Finished predictions, saving to \"{TEST_PREDS_FP}\"")
    results.to_csv(TEST_PREDS_FP, index=False)
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