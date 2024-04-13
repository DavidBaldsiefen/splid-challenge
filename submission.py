import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import pickle
import time
import gc

from base import datahandler, classifier, localizer

# Use DEBUG mode when running outside of docker container
DEBUG_MODE = True
if DEBUG_MODE:
    from base import evaluation
    print("Warning: Running in debug-mode, disable before submitting!")

LOCALIZER_ADIK_DIR = Path(('' if DEBUG_MODE else '/') + 'models/localizer_adik_model.hdf5')
SCALER_ADIK_DIR = Path(('' if DEBUG_MODE else '/') + 'models/localizer_adik_scaler.pkl')

#LOCALIZER_ADIK_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/model_jrlrqj4g.hdf5')
#SCALER_ADIK_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/scaler_jrlrqj4g.pkl')

LOCALIZER_ID_DIR = Path(('' if DEBUG_MODE else '/') + 'models/localizer_id_model.hdf5')
SCALER_ID_DIR = Path(('' if DEBUG_MODE else '/') + 'models/localizer_id_scaler.pkl')

#LOCALIZER_ID_DIR = Path("/home/david/Code/splid-challenge/submission/models/model_x9fwigu5.hdf5")
#SCALER_ID_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/scaler_x9fwigu5.pkl')

CLASSIFIER_DIR = Path(('' if DEBUG_MODE else '/') + 'models/classifier_model.hdf5')
SCALER_CLASSIFIER_DIR = Path(('' if DEBUG_MODE else '/') + 'models/classifier_scaler.pkl')
#CLASSIFIER_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/model_zhkytvx1.hdf5')
#SCALER_CLASSIFIER_DIR = Path(('submission/' if DEBUG_MODE else '/') + 'models/scaler_zhkytvx1.pkl')

TEST_DATA_DIR = Path(('dataset/phase_2/' if DEBUG_MODE else '/dataset/') + 'test') #!!!
DEBUG_LABELS_DIR = Path('dataset/phase_2/test_label.csv')

TEST_PREDS_FP = Path(('' if DEBUG_MODE else '/submission/') + 'submission.csv')

# Load Data
split_dataframes = datahandler.load_and_prepare_dataframes(TEST_DATA_DIR, labels_dir=None)
print(f"Loaded {len(split_dataframes.keys())} dataset files from \"{TEST_DATA_DIR}\". Creating dataset")

# =================================LOCALIZATION==========================================
#-----------------------------------AD+IK-------------------------------

adik_subm_df = localizer.perform_submission_pipeline(localizer_dir=LOCALIZER_ADIK_DIR,
                                                    scaler_dir=SCALER_ADIK_DIR,
                                                    split_dataframes=split_dataframes,
                                                    output_dirs=['EW', 'NS'],
                                                    thresholds=[45.0, 50.0],
                                                    convolve_input_stride=True,
                                                    merge_neighbors_below_distance=2,
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
                                                                            'Inclination (deg)',
                                                                            #'RAAN (deg)',
                                                                            #'Argument of Periapsis (deg)',
                                                                            'True Anomaly (deg)',
                                                                            'Longitude (deg)',
                                                                            #'Latitude (deg)'
                                                                            ],
                                                    legacy_diff_transform=True,
                                                    sin_transform_features=[#'Eccentricity',
                                                                            #'Semimajor Axis (m)',
                                                                            #'Inclination (deg)',
                                                                            #'RAAN (deg)',
                                                                            'Argument of Periapsis (deg)',
                                                                            #'True Anomaly (deg)',
                                                                            #'Longitude (deg)',
                                                                            #'Latitude (deg)'
                                                                            ],
                                                    sin_cos_transform_features=[],
                                                    overview_features_mean=[],
                                                    overview_features_std=[],
                                                    add_daytime_feature=False,
                                                    add_yeartime_feature=False,
                                                    add_linear_timeindex=False,
                                                    input_history_steps=128,
                                                    input_future_steps=32,
                                                    input_stride=2,
                                                    padding='zero')

gc.collect()
#-----------------------------------ID-------------------------------

id_subm_df = localizer.perform_submission_pipeline(localizer_dir=LOCALIZER_ID_DIR,
                                                    scaler_dir=SCALER_ID_DIR,
                                                    split_dataframes=split_dataframes,
                                                    output_dirs=['EW', 'NS'],
                                                    thresholds=[40.0, 50.0],
                                                    convolve_input_stride=True,
                                                    merge_neighbors_below_distance=2,
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
                                                                            'Inclination (deg)',
                                                                            #'RAAN (deg)',
                                                                            #'Argument of Periapsis (deg)',
                                                                            'True Anomaly (deg)',
                                                                            'Longitude (deg)',
                                                                            #'Latitude (deg)'
                                                                            ],
                                                    legacy_diff_transform=True,
                                                    sin_transform_features=[#'Eccentricity',
                                                                            #'Semimajor Axis (m)',
                                                                            #'Inclination (deg)',
                                                                            #'RAAN (deg)',
                                                                            'Argument of Periapsis (deg)',
                                                                            #'True Anomaly (deg)',
                                                                            #'Longitude (deg)',
                                                                            #'Latitude (deg)'
                                                                            ],
                                                    sin_cos_transform_features=[],
                                                    overview_features_mean=[#'Longitude (sin)',
                                                                            #'RAAN (deg)'
                                                                            ],
                                                    overview_features_std=['Inclination (deg)'
                                                                            ],
                                                    add_daytime_feature=False,
                                                    add_yeartime_feature=False,
                                                    add_linear_timeindex=True,
                                                    linear_timeindex_as_overview=True,
                                                    input_history_steps=320,
                                                    input_future_steps=256,
                                                    input_stride=4,
                                                    padding='zero')

# For ID, we know the node and type already. In theory there could be FPs where other nodes are, but for duplicates, the nodes&types are reset again
id_subm_df['Node'] = 'ID'
id_subm_df['Type'] = 'NK'

gc.collect()
#--------------------------------COMBINE-------------------------------
df_locs = pd.concat([adik_subm_df, id_subm_df]).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
# remove duplicates - which are possible now
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
    initial_node_df['ObjectID'] = list(map(int, split_dataframes.keys()))
    initial_node_df['TimeIndex'] = 0
    initial_node_df['Direction'] = dir
    initial_node_df['Node'] = 'SS'
    initial_node_df['Type'] = 'UNKNOWN'
    initial_node_dfs.append(initial_node_df)
print(f"Adding {len(initial_node_dfs[0]) + len(initial_node_dfs[1])} initial nodes.")
df_locs = pd.concat([df_locs] + initial_node_dfs)

# =================================CLASSIFICATION==========================================

classified_df = classifier.perform_submission_pipeline(classifier_dir=CLASSIFIER_DIR,
                                                    scaler_dir=SCALER_CLASSIFIER_DIR,
                                                    split_dataframes=split_dataframes,
                                                    loc_preds=df_locs,
                                                    convolve_input_stride=False,
                                                    remove_ns_during_ew_nk=True,
                                                    remove_consecutive_ID_IK=False,
                                                    output_dirs=['EW', 'NS'],
                                                    non_transform_features=['Eccentricity',
                                                              'Semimajor Axis (m)',
                                                              #'Inclination (deg)',
                                                              'RAAN (deg)',
                                                              #'Argument of Periapsis (deg)',
                                                              'True Anomaly (deg)',
                                                              'Latitude (deg)',
                                                              #'Longitude (deg)',
                                                              ],
                                                    diff_transform_features=['Eccentricity',
                                                                            'Semimajor Axis (m)',
                                                                            'Inclination (deg)',
                                                                            #'RAAN (deg)',
                                                                            #'Argument of Periapsis (deg)',
                                                                            'True Anomaly (deg)',
                                                                            #'Longitude (deg)',
                                                                            #'Latitude (deg)'
                                                                            ],
                                                    legacy_diff_transform=True,
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
                                                    overview_features_mean=[#'Eccentricity',
                                                                            #'Semimajor Axis (m)',
                                                                            #'Inclination (deg)',
                                                                            #'RAAN (deg)',
                                                                            #'Argument of Periapsis (sin)',
                                                                            #'True Anomaly (deg)',
                                                                            #'Latitude (deg)',
                                                                            #'Longitude (sin)',
                                                                            ],
                                                    overview_features_std=['Argument of Periapsis (sin)', 'Latitude (deg)', 'Longitude (sin)'],
                                                    add_daytime_feature=False,
                                                    add_yeartime_feature=False,
                                                    add_linear_timeindex=True,
                                                    linear_timeindex_as_overview=True,
                                                    input_history_steps=32,
                                                    input_future_steps=256,
                                                    input_stride=1,
                                                    padding='zero')

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
    ground_truth_df = pd.read_csv(DEBUG_LABELS_DIR)
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