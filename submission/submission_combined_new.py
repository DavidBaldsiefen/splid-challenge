import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import pickle
import time
import sys

from base import utils, datahandler, classifier

DEBUG_MODE = False

if DEBUG_MODE:
    print("Warnin: Running in debug-mode, disable before submitting!")

LOCALIZER_EW_DIR = Path('models/ew_localizer.hdf5') if DEBUG_MODE else Path('/models/ew_localizer.hdf5')
SCALER_EW_DIR = Path('models/EW_localizer_scaler.pkl') if DEBUG_MODE else Path('/models/EW_localizer_scaler.pkl')
LOCALIZER_NS_DIR = Path('models/ns_localizer.hdf5') if DEBUG_MODE else Path('/models/ns_localizer.hdf5')
SCALER_NS_DIR = Path('models/NS_localizer_scaler.pkl') if DEBUG_MODE else Path('/models/NS_localizer_scaler.pkl')
CLASSIFIER_DIR = Path('models/ew_ns_classifier_new.hdf5') if DEBUG_MODE else Path('/models/ew_ns_classifier_new.hdf5')
SCALER_CLASSIFIER_DIR = Path('models/ew_ns_classifier_scaler_new.pkl') if DEBUG_MODE else Path('/models/ew_ns_classifier_scaler_new.pkl')
TEST_DATA_DIR = Path('dataset/test/') if DEBUG_MODE else Path('/dataset/test/')
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
test_ds = ds_gen.get_datasets(128, label_features=[], shuffle=False, with_identifier=True, stride=1)

inputs = np.concatenate([element for element in test_ds.map(lambda x,z: x).as_numpy_iterator()])
identifiers_ew = np.concatenate([element for element in test_ds.map(lambda x,z: z).as_numpy_iterator()])

print(f"Predicting EW locations using model \"{LOCALIZER_EW_DIR}\"")
ew_localizer = tf.keras.models.load_model(LOCALIZER_EW_DIR)
preds_ew = ew_localizer.predict(inputs, verbose=2)
preds_ew_argmax = (preds_ew>=50.0).astype(int)

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
                                      custom_scaler=ew_localizer_scaler,
                                      seed=69)
test_ds = ds_gen.get_datasets(128, label_features=[], shuffle=False, with_identifier=True, stride=1)

inputs = np.concatenate([element for element in test_ds.map(lambda x,z: x).as_numpy_iterator()])
identifiers_ns = np.concatenate([element for element in test_ds.map(lambda x,z: z).as_numpy_iterator()])

print(f"Predicting NS locations using model \"{LOCALIZER_NS_DIR}\"")
ns_localizer = tf.keras.models.load_model(LOCALIZER_NS_DIR)
preds_ns = ns_localizer.predict(inputs, verbose=2)
preds_ns_argmax = (preds_ns>=50.0).astype(int)

# Localization cleanup
df_ew = pd.DataFrame(np.concatenate([identifiers_ew.reshape(-1,2)], axis=1), columns=['ObjectID', 'TimeIndex'], dtype=np.int32)
df_ew[f'Location_EW'] = preds_ew_argmax
df_ns = pd.DataFrame(np.concatenate([identifiers_ns.reshape(-1,2)], axis=1), columns=['ObjectID', 'TimeIndex'], dtype=np.int32)
df_ns[f'Location_NS'] = preds_ns_argmax # for binary preds

# clean consecutive detections
df_ew = df_ew.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
df_ew = df_ew.loc[(df_ew['Location_EW']==1)]
df_ew['consecutive'] = (df_ew['TimeIndex'] - df_ew['TimeIndex'].shift(1) != 1).cumsum()
df_ew=df_ew.groupby('consecutive').apply(lambda sub_df: sub_df.iloc[int(len(sub_df)/2), :]).reset_index(drop=True).drop(columns=['consecutive'])

df_ns = df_ns.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
df_ns = df_ns.loc[(df_ns['Location_NS']==1)]
df_ns['consecutive'] = (df_ns['TimeIndex'] - df_ns['TimeIndex'].shift(1) != 1).cumsum()
df_ns=df_ns.groupby('consecutive').apply(lambda sub_df: sub_df.iloc[int(len(sub_df)/2), :]).reset_index(drop=True).drop(columns=['consecutive'])

df = df_ew.merge(df_ns, how='outer', on = ['ObjectID', 'TimeIndex']).fillna(0).astype(np.int32)

# add initial node prediction for each object
object_ids = list(map(int, list(split_dataframes.keys())))
for obj_id in object_ids:
    df = df.sort_index()
    df.loc[-1] = [int(obj_id), 0, 1, 1] # objid, timeindex, Location_EW, Location_NS
    df.index = df.index + 1
    df = df.sort_index()

df_locs = df.loc[(df['Location_EW'] == 1) | (df['Location_NS'] == 1)]

print(f"ObjectIDs ({len(object_ids)}): {object_ids}")
print(f"#EW_Preds: {len(df.loc[(df['Location_EW'] == 1)])}")
print(f"#NS_Preds: {len(df.loc[(df['Location_NS'] == 1)])}")

sub_dfs = []
for dir in ['EW', 'NS']:
    sub_df = df_locs.copy()
    sub_df = sub_df.loc[(sub_df[f'Location_{dir}'] == 1)]
    sub_df['Direction'] = dir
    sub_df = sub_df.drop([f'Location_{dir}'], axis='columns')
    sub_dfs.append(sub_df)
df_locs = pd.concat(sub_dfs)
df_locs = df_locs.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
print(df_locs.head(5))

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
                                      input_future_steps=50,
                                      custom_scaler=classifier_scaler,
                                      seed=69)
classifier_model = tf.keras.models.load_model(CLASSIFIER_DIR)

pred_df = classifier.create_prediction_df(ds_gen=ds_gen,
                                model=classifier_model,
                                train=False,
                                test=True,
                                model_outputs=['EW_Type', 'NS_Type'],
                                verbose=2)
majority_df = classifier.apply_majority_method(preds_df=pred_df, location_df=df_locs)

# =====================================================================================================

# TEMPORARY: Remove NS detections, because they probably add way more FP than TP
df_reduced = majority_df.loc[(majority_df['TimeIndex'] == 0) | (majority_df['Direction'] == 'EW')]
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