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

LOCALIZER_EW_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/EW_localizer_cnn.hdf5')
SCALER_EW_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/EW_localizer_scaler_cnn.pkl')

LOCALIZER_NS_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/NS_localizer_cnn.hdf5')
SCALER_NS_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/NS_localizer_scaler_cnn.pkl')

CLASSIFIER_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/ew_ns_classifier_oneshot_cnn.hdf5')
SCALER_CLASSIFIER_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/ew_ns_classifier_scaler_oneshot_cnn.pkl')

TEST_DATA_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'dataset/test/') #!!!
TEST_PREDS_FP = Path(('submission/' if DEBUG_MODE else '/') + 'submission/submission.csv')

# Load Data
split_dataframes = datahandler.load_and_prepare_dataframes(TEST_DATA_DIR, labels_dir=None)
print(f"Loaded {len(split_dataframes.keys())} dataset files from \"{TEST_DATA_DIR}\". Creating dataset")

# =================================LOCALIZATION==========================================
#-----------------------------------EW-------------------------------

ew_localizer_scaler = pickle.load(open(SCALER_EW_DIR, 'rb'))
ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      non_transform_features=['Eccentricity',
                                                              'Semimajor Axis (m)',
                                                              'Inclination (deg)',
                                                              'RAAN (deg)',
                                                              'Argument of Periapsis (deg)',
                                                              'True Anomaly (deg)',
                                                              'Longitude (deg)',
                                                              'Latitude (deg)'],
                                      diff_transform_features=['Eccentricity',
                                                              'Semimajor Axis (m)',
                                                              'Inclination (deg)',
                                                              'RAAN (deg)',
                                                              'Argument of Periapsis (deg)',
                                                              'True Anomaly (deg)',
                                                              'Longitude (deg)',
                                                              'Latitude (deg)'],
                                      sin_transform_features=[],
                                      sin_cos_transform_features=[],
                                      overview_features_mean=[],
                                      overview_features_std=[],
                                      add_daytime_feature=False,
                                      add_yeartime_feature=False,
                                      add_linear_timeindex=False,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=2,
                                      padding='zero',
                                      input_history_steps=48,
                                      input_future_steps=24,
                                      per_object_scaling=False,
                                      custom_scaler=ew_localizer_scaler,
                                      input_dtype=np.float32,
                                      sort_inputs=True,
                                      seed=69)

print(f"Predicting EW locations using model \"{LOCALIZER_EW_DIR}\" and scaler \"{SCALER_EW_DIR}\"")
ew_localizer = tf.keras.models.load_model(LOCALIZER_EW_DIR, compile=False)

ew_preds_df = localizer.create_prediction_df(ds_gen=ds_gen,
                                model=ew_localizer,
                                train=False,
                                test=True,
                                output_dirs=['EW'],
                                prediction_batches=5,
                                verbose=2)

ew_subm_df = localizer.postprocess_predictions(preds_df=ew_preds_df,
                                            dirs=['EW'],
                                            threshold=60.0,
                                            add_initial_node=True,
                                            clean_consecutives=True)
gc.collect()
#-----------------------------------NS-------------------------------

ns_localizer_scaler = pickle.load(open(SCALER_NS_DIR, 'rb'))
ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      non_transform_features=['Eccentricity',
                                                              'Semimajor Axis (m)',
                                                              'Inclination (deg)',
                                                              'RAAN (deg)',
                                                              #'Argument of Periapsis (deg)',
                                                              #'True Anomaly (deg)',
                                                              #'Longitude (deg)',
                                                              'Latitude (deg)'],
                                      diff_transform_features=[#'Eccentricity',
                                                              #'Semimajor Axis (m)',
                                                              #'Inclination (deg)',
                                                              #'RAAN (deg)',
                                                              #'Argument of Periapsis (deg)',
                                                              'True Anomaly (deg)',
                                                              #'Longitude (deg)',
                                                              #'Latitude (deg)'
                                                              ],
                                      sin_transform_features=[#'Eccentricity',
                                                              #'Semimajor Axis (m)',
                                                              #'Inclination (deg)',
                                                              #'RAAN (deg)',
                                                              'Argument of Periapsis (deg)',
                                                              #'True Anomaly (deg)',
                                                              'Longitude (deg)',
                                                              #'Latitude (deg)'
                                                              ],
                                      sin_cos_transform_features=[],
                                      overview_features_mean=['Longitude (sin)',
                                                              'RAAN (deg)'
                                                               ],
                                      overview_features_std=['Latitude (deg)'
                                                             ],
                                      add_daytime_feature=False,
                                      add_yeartime_feature=False,
                                      add_linear_timeindex=True,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=8,
                                      padding='zero', #!
                                      input_history_steps=256,
                                      input_future_steps=256,
                                      per_object_scaling=False,
                                      custom_scaler=ns_localizer_scaler,
                                      input_dtype=np.float32,
                                      sort_inputs=True,
                                      seed=69)

print(f"Predicting NS locations using model \"{LOCALIZER_NS_DIR}\" and scaler \"{SCALER_NS_DIR}\"")
ns_localizer = tf.keras.models.load_model(LOCALIZER_NS_DIR, compile=False)
ns_preds_df = localizer.create_prediction_df(ds_gen=ds_gen,
                                model=ns_localizer,
                                train=False,
                                test=True,
                                output_dirs=['NS'],
                                prediction_batches=5,
                                verbose=2)

ns_subm_df = localizer.postprocess_predictions(preds_df=ns_preds_df,
                                            dirs=['NS'],
                                            threshold=42.0,
                                            add_initial_node=True,
                                            clean_consecutives=True)
gc.collect()
#--------------------------------COMBINE-------------------------------
df_locs = pd.concat([ew_subm_df, ns_subm_df]).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

print(f"#EW_Preds: {len(df_locs.loc[(df_locs['Direction'] == 'EW')])}")
print(f"#NS_Preds: {len(df_locs.loc[(df_locs['Direction'] == 'NS')])}")

# =================================CLASSIFICATION==========================================

classifier_scaler = pickle.load(open(SCALER_CLASSIFIER_DIR, 'rb'))

ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      non_transform_features=['Eccentricity',
                                                              'Semimajor Axis (m)',
                                                              'Inclination (deg)',
                                                              'RAAN (deg)',
                                                              'Argument of Periapsis (deg)',
                                                              'True Anomaly (deg)',
                                                              'Latitude (deg)',
                                                              'Longitude (deg)',
                                                              ],
                                      diff_transform_features=['Eccentricity',
                                                               'Semimajor Axis (m)',
                                                               'Inclination (deg)',
                                                               'RAAN (deg)',
                                                               'Argument of Periapsis (deg)',
                                                               'True Anomaly (deg)',
                                                               'Longitude (deg)',
                                                               'Latitude (deg)'],
                                      sin_transform_features=[#'Inclination (deg)',
                                                              #'RAAN (deg)',
                                                              #'Argument of Periapsis (deg)',
                                                              #'True Anomaly (deg)',
                                                              #'Longitude (deg)'
                                                                ],
                                      sin_cos_transform_features=[],
                                      overview_features_mean=[#'Inclination (sin)',
                                                              #'RAAN (sin)'
                                                                ],
                                      overview_features_std=[#'Latitude (deg)'
                                                             ],
                                      add_daytime_feature=False,
                                      add_yeartime_feature=False,
                                      add_linear_timeindex=False,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=1,
                                      padding='zero',
                                      input_history_steps=16,
                                      input_future_steps=128,
                                      custom_scaler=classifier_scaler,
                                      input_dtype=np.float32,
                                      sort_inputs=True,
                                      seed=69)
print(f"Classifying using model \"{CLASSIFIER_DIR}\"")
classifier_model = tf.keras.models.load_model(CLASSIFIER_DIR, compile=False)

pred_df = classifier.create_prediction_df(ds_gen=ds_gen,
                                model=classifier_model,
                                train=False,
                                test=True,
                                only_nodes=False,
                                model_outputs=['EW_Type', 'NS_Type'],
                                object_limit=None,
                                prediction_batches=5,
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
    ground_truth_df = pd.read_csv(Path('submission/dataset/test_labels.csv'))
    results.to_csv('submission/submission/debug_submission.csv', index=False)
    evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=results)
    precision, recall, f2, rmse, total_tp, total_fp, total_fn, total_df = evaluator.score()
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F2: {f2:.3f}')
    print(f'RMSE: {float(rmse):.4}')
    print(f'TP: {total_tp} FP: {total_fp} FN: {total_fn}')
    print("Done.")