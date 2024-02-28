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

LOCALIZER_ADIK_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/ADIK_localizer_cnn.hdf5')
SCALER_ADIK_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/ADIK_localizer_scaler_cnn.pkl')

LOCALIZER_ID_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/ID_localizer_cnn.hdf5')
SCALER_ID_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/ID_localizer_scaler_cnn.pkl')

CLASSIFIER_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/ew_ns_classifier_oneshot_cnn.hdf5')
SCALER_CLASSIFIER_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/ew_ns_classifier_scaler_oneshot_cnn.pkl')

TEST_DATA_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'dataset/test/') #!!!
TEST_PREDS_FP = Path(('submission/' if DEBUG_MODE else '/') + 'submission/submission.csv')

# Load Data
split_dataframes = datahandler.load_and_prepare_dataframes(TEST_DATA_DIR, labels_dir=None)
print(f"Loaded {len(split_dataframes.keys())} dataset files from \"{TEST_DATA_DIR}\". Creating dataset")

# =================================LOCALIZATION==========================================
#-----------------------------------AD+IK-------------------------------

adik_localizer_scaler = pickle.load(open(SCALER_ADIK_DIR, 'rb'))
ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      non_transform_features=['Eccentricity',
                                                              'Semimajor Axis (m)',
                                                              'Inclination (deg)',
                                                              'RAAN (deg)',
                                                              'Argument of Periapsis (deg)',
                                                              #'True Anomaly (deg)',
                                                              #'Longitude (deg)',
                                                              'Latitude (deg)'],
                                      diff_transform_features=[#'Eccentricity',
                                                              #'Semimajor Axis (m)',
                                                              #'Inclination (deg)',
                                                              'RAAN (deg)',
                                                              'Argument of Periapsis (deg)',
                                                              'True Anomaly (deg)',
                                                              'Longitude (deg)',
                                                              'Latitude (deg)'
                                                              ],
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
                                      input_history_steps=128,
                                      input_future_steps=32,
                                      per_object_scaling=False,
                                      custom_scaler=adik_localizer_scaler,
                                      unify_value_ranges=True,
                                      input_dtype=np.float32,
                                      sort_inputs=True,
                                      seed=69)

print(f"Predicting AD+IK locations using model \"{LOCALIZER_ADIK_DIR}\" and scaler \"{SCALER_ADIK_DIR}\"")
adik_localizer = tf.keras.models.load_model(LOCALIZER_ADIK_DIR, compile=False)

adik_preds_df = localizer.create_prediction_df(ds_gen=ds_gen,
                                model=adik_localizer,
                                train=False,
                                test=True,
                                output_dirs=['EW', 'NS'],
                                prediction_batches=5,
                                verbose=2)

adik_subm_df = localizer.postprocess_predictions(preds_df=adik_preds_df,
                                            dirs=['EW', 'NS'],
                                            threshold=65.0, # tendency would be to set this value even higher
                                            add_initial_node=False, # Do not add initial nodes just yet
                                            clean_consecutives=True)
gc.collect()
#-----------------------------------ID-------------------------------

id_localizer_scaler = pickle.load(open(SCALER_ID_DIR, 'rb'))
ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      non_transform_features=['Eccentricity',
                                                              'Semimajor Axis (m)',
                                                              'Inclination (deg)',
                                                              'RAAN (deg)',
                                                              'Argument of Periapsis (deg)',
                                                              #'True Anomaly (deg)',
                                                              'Longitude (deg)',
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
                                                              #'Argument of Periapsis (deg)',
                                                              #'True Anomaly (deg)',
                                                              #'Longitude (deg)',
                                                              #'Latitude (deg)'
                                                              ],
                                      sin_cos_transform_features=[],
                                      overview_features_mean=[#'Longitude (sin)',
                                                              #'RAAN (deg)'
                                                               ],
                                      overview_features_std=[#'Latitude (deg)'
                                                             ],
                                      add_daytime_feature=False,
                                      add_yeartime_feature=False,
                                      add_linear_timeindex=False,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=4,
                                      padding='zero', #!
                                      input_history_steps=256,
                                      input_future_steps=256,
                                      per_object_scaling=False,
                                      custom_scaler=id_localizer_scaler,
                                      unify_value_ranges=True,
                                      input_dtype=np.float32,
                                      sort_inputs=True,
                                      seed=69)

print(f"Predicting ID locations using model \"{LOCALIZER_ID_DIR}\" and scaler \"{SCALER_ID_DIR}\"")
id_localizer = tf.keras.models.load_model(LOCALIZER_ID_DIR, compile=False)
id_preds_df = localizer.create_prediction_df(ds_gen=ds_gen,
                                model=id_localizer,
                                train=False,
                                test=True,
                                output_dirs=['EW', 'NS'], # ! Check orders
                                prediction_batches=5,
                                verbose=2)

id_subm_df = localizer.postprocess_predictions(preds_df=id_preds_df,
                                            dirs=['EW', 'NS'],
                                            threshold=42.0,
                                            add_initial_node=False,
                                            clean_consecutives=True)

# For ID, we know the node and type already. In theory there could be FPs where other nodes are, but for duplicates, the nodes&types are reset again
id_subm_df['Node'] = 'ID'
id_subm_df['Type'] = 'NK'

gc.collect()
#--------------------------------COMBINE-------------------------------
df_locs = pd.concat([adik_subm_df, id_subm_df]).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
# TODO: remove duplicates - which are possible now
duplicate_indices = df_locs.duplicated(subset=['ObjectID', 'TimeIndex', 'Direction'], keep=False) # returns index of all duplicates
df_locs.loc[duplicate_indices==True, 'Node'] = 'UNKNOWN'
df_locs.loc[duplicate_indices==True, 'Type'] = 'UNKNOWN'
duplicate_indices_keep_first = df_locs.duplicated(subset=['ObjectID', 'TimeIndex', 'Direction'], keep='first') # returns index of all duplicates except the first
df_locs = df_locs[duplicate_indices_keep_first==False]
print(f"Removed {duplicate_indices_keep_first.sum()} duplicate entries, keeping the first occurence and setting Type&Node to UNKNOWN")

print(f"#ADIK_Preds: {len(adik_subm_df)}")
print(f"#ID_Preds: {len(id_subm_df)}")
print(f"#EW_Preds: {len(df_locs.loc[(df_locs['Direction'] == 'EW')])}")
print(f"#NS_Preds: {len(df_locs.loc[(df_locs['Direction'] == 'NS')])}")

# add initial nodes
initial_node_dfs = []
for dir in ['NS', 'EW']:
    initial_node_df = pd.DataFrame(columns=df_locs.columns)
    initial_node_df['ObjectID'] = list(map(int, ds_gen.train_keys))
    initial_node_df['TimeIndex'] = 0
    initial_node_df['Direction'] = dir
    initial_node_df['Node'] = 'SS'
    initial_node_df['Type'] = 'UNKNOWN'
    initial_node_dfs.append(initial_node_df)
print(f"Adding {len(initial_node_dfs[0]) + len(initial_node_dfs[1])} initial nodes.")
df_locs = pd.concat([df_locs] + initial_node_dfs)

# =================================CLASSIFICATION==========================================

classifier_scaler = pickle.load(open(SCALER_CLASSIFIER_DIR, 'rb'))

ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      non_transform_features=['Eccentricity',
                                                              'Semimajor Axis (m)',
                                                              'Inclination (deg)',
                                                              'RAAN (deg)',
                                                              #'Argument of Periapsis (deg)',
                                                              'True Anomaly (deg)',
                                                              'Latitude (deg)',
                                                              #'Longitude (deg)',
                                                              ],
                                      diff_transform_features=['Eccentricity',
                                                               'Semimajor Axis (m)',
                                                               'Inclination (deg)',
                                                               'RAAN (deg)',
                                                               'Argument of Periapsis (deg)',
                                                               'True Anomaly (deg)',
                                                               #'Longitude (deg)',
                                                               'Latitude (deg)'
                                                               ],
                                      sin_transform_features=[ #'Inclination (deg)',
                                                               #'RAAN (deg)',
                                                               'Argument of Periapsis (deg)',
                                                               #'True Anomaly (deg)',
                                                               'Longitude (deg)',
                                                               #'Latitude (deg)'
                                                              ],
                                      sin_cos_transform_features=[
                                                               #'Inclination (deg)',
                                                               #'RAAN (deg)',
                                                               #'Argument of Periapsis (deg)',
                                                               #'True Anomaly (deg)',
                                                               #'Longitude (deg)',
                                                               #'Latitude (deg)'
                                                               ],
                                      overview_features_mean=['Eccentricity',
                                                              'Semimajor Axis (m)',
                                                              'Inclination (deg)',
                                                              'RAAN (deg)',
                                                              'Argument of Periapsis (sin)',
                                                              #'True Anomaly (deg)',
                                                              #'Latitude (deg)',
                                                              'Longitude (sin)',
                                                              ],
                                      overview_features_std=['Latitude (deg)',
                                                             'Argument of Periapsis (sin)'
                                                             ],
                                      add_daytime_feature=False,
                                      add_yeartime_feature=False,
                                      add_linear_timeindex=True,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=1,
                                      padding='zero',
                                      input_history_steps=16,
                                      input_future_steps=128,
                                      custom_scaler=classifier_scaler,
                                      unify_value_ranges=True,
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
#majority_df = classifier.apply_one_shot_method(preds_df=pred_df, location_df=df_locs)
print(df_locs.head(10))
typed_df = classifier.fill_unknown_types_based_on_preds(pred_df, df_locs, dirs=['EW', 'NS'])
print(typed_df.head(10))
classified_df = classifier.fill_unknwon_nodes_based_on_type(typed_df, dirs=['EW', 'NS'])
print(classified_df.head(10))

# =====================================================================================================

# Use this to (temporarily!) remove certain parts
df_reduced = classified_df.loc[(classified_df['TimeIndex'] == 0) | (classified_df['Direction'] == 'EW') | (classified_df['Direction'] == 'NS')]

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
    print("------------------------------------------------------")
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

    if total_df is not None:
        tp_ID = len(total_df.loc[(total_df['Node'] == 'ID') & (total_df['classification'] == 'TP')])
        fn_ID = len(total_df.loc[(total_df['Node'] == 'ID') & (total_df['classification'] == 'FN')])
        tp_IK = len(total_df.loc[(total_df['Node'] == 'IK') & (total_df['classification'] == 'TP')])
        fn_IK = len(total_df.loc[(total_df['Node'] == 'IK') & (total_df['classification'] == 'FN')])
        tp_AD = len(total_df.loc[(total_df['Node'] == 'AD') & (total_df['classification'] == 'TP')])
        fn_AD = len(total_df.loc[(total_df['Node'] == 'AD') & (total_df['classification'] == 'FN')])
        print(f"TP/FN based on Node:")
        print(f"ID: {tp_ID}|{fn_ID}")
        print(f"IK: {tp_IK}|{fn_IK}")
        print(f"AD: {tp_AD}|{fn_AD}")
    # perform no-class evaluation as well
    evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=results, ignore_classes=True)
    precision, recall, f2, rmse, total_tp, total_fp, total_fn, total_df = evaluator.score()
    print(f"Scores when ignoring classification:\n\tPrecision: {precision:.2f} Recall: {recall:.2f} F2: {f2:.3f} | TP: {total_tp} FP: {total_fp} FN: {total_fn}")
    print("------------------------------------------------------")

    print("Done.")