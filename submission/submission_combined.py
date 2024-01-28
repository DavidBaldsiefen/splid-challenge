import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import pickle
import time
import sys

from base import utils, datahandler

LOCALIZER_EW_DIR = Path('/models/ew_localizer.hdf5')
SCALER_EW_DIR = Path('/models/EW_localizer_scaler.pkl')
LOCALIZER_NS_DIR = Path('/models/ns_localizer.hdf5')
SCALER_NS_DIR = Path('/models/NS_localizer_scaler.pkl')
CLASSIFIER_DIR = Path('/models/ew_ns_classifier.hdf5')
TYPE_CLASSIFIER_DIR = Path('/models/ew_ns_type_classifier.hdf5')
SCALER_CLASSIFIER_DIR = Path('/models/classifier_scaler.pkl')
SCALER_TYPE_CLASSIFIER_DIR = Path('/models/type_classifier_scaler.pkl')
TEST_DATA_DIR = Path('/dataset/test/')
TEST_PREDS_FP = Path('/submission/submission.csv')

# Load Data
split_dataframes = datahandler.load_and_prepare_dataframes(TEST_DATA_DIR, labels_dir=None)
print(f"Loaded {len(split_dataframes.keys())} dataset files from \"{TEST_DATA_DIR}\". Creating dataset")

reduced_input_features = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)',
       'Argument of Periapsis (deg)', 'True Anomaly (deg)', 'Latitude (deg)',
       'Longitude (deg)', 'Altitude (m)']
ew_input_features = ['Eccentricity', 'Semimajor Axis (m)', 'Argument of Periapsis (deg)', 'Longitude (deg)', 'Altitude (m)']
ns_input_features = ['Eccentricity', 'Semimajor Axis (m)', 'RAAN (deg)', 'Inclination (deg)', 'Latitude (deg)', 'Altitude (m)']

# ============================================================================================
# EW-Localization
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
ns_localizer_scaler = pickle.load(open(SCALER_NS_DIR, 'rb'))
ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      input_features=ns_input_features,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=4,
                                      padding='none',
                                      input_history_steps=80,
                                      input_future_steps=80,
                                      custom_scaler=ns_localizer_scaler,
                                      seed=69)
test_ds = ds_gen.get_datasets(128, label_features=[], shuffle=False, with_identifier=True, stride=1)

inputs = np.concatenate([element for element in test_ds.map(lambda x,z: x).as_numpy_iterator()])
identifiers_ns = np.concatenate([element for element in test_ds.map(lambda x,z: z).as_numpy_iterator()])

print(f"Predicting NS locations using model \"{LOCALIZER_NS_DIR}\"")
ns_localizer = tf.keras.models.load_model(LOCALIZER_NS_DIR)
preds_ns = ns_localizer.predict(inputs, verbose=2)
preds_ns_argmax = (preds_ns>=0.5).astype(int)

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
df_locs = df_locs.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

print(f"ObjectIDs ({len(object_ids)}): {object_ids}")
print(f"#EW_Preds: {len(df.loc[(df['Location_EW'] == 1)])}")
print(f"#NS_Preds: {len(df.loc[(df['Location_NS'] == 1)])}")

# ============================================================================================
# Classification
classifier_scaler = pickle.load(open(SCALER_CLASSIFIER_DIR, 'rb'))
ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      input_features=reduced_input_features,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=2, #!
                                      padding='same',
                                      input_history_steps=40,
                                      input_future_steps=40,
                                      custom_scaler=classifier_scaler,
                                      seed=69)
test_ds = ds_gen.get_datasets(128, label_features=[], shuffle=False, with_identifier=True, stride=1)

inputs = np.concatenate([element for element in test_ds.map(lambda x,z: x).as_numpy_iterator()])
identifiers = np.concatenate([element for element in test_ds.map(lambda x,z: z).as_numpy_iterator()])

print(f"Classifying EW&NS locations using model \"{CLASSIFIER_DIR}\"")
ew_ns_classifier = tf.keras.models.load_model(CLASSIFIER_DIR)
preds = ew_ns_classifier.predict(inputs, verbose=2)

# Classification cleanup
sub_dfs = []
for ft_idx, feature in enumerate(['EW', 'NS']):
    sub_df = pd.DataFrame(np.concatenate([identifiers.reshape(-1,2)], axis=1), columns=['ObjectID', 'TimeIndex'], dtype=np.int32)
    sub_df['Direction'] = feature
    preds_argmax = np.argmax(preds[ft_idx], axis=1)
    sub_df[f'{feature}_Decoded'] = ds_gen.combined_label_encoder.inverse_transform(preds_argmax)
    sub_df[['Node', 'Type']] = sub_df[f'{feature}_Decoded'].str.split('-', expand=True)
    sub_df = sub_df.drop([f'{feature}_Decoded'], axis='columns')
    sub_dfs.append(sub_df)

df_classes = pd.concat(sub_dfs).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

# Lets add our background knowledge:
# 1) For timeindex 0, the node is always SS
df_classes.loc[df_classes['TimeIndex'] == 0, 'Node'] = 'SS'
# 2) AD, ID is always combined with NK
df_classes.loc[(df_classes['Node'] == 'AD') | (df_classes['Node'] == 'ID'), 'Type'] = 'NK'
# 3) IK is always combined with HK/CK/EK
df_classes.loc[(df_classes['Node'] == 'IK') & (df_classes['Type'] == 'NK'), 'Type'] = 'CK' # CK is most common

# Add the better Type classifier for the initial predictions
ew_ns_type_classifier = tf.keras.models.load_model(TYPE_CLASSIFIER_DIR)
type_classifier_scaler = pickle.load(open(SCALER_TYPE_CLASSIFIER_DIR, 'rb'))

ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      input_features=reduced_input_features,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=2,
                                      padding='none',
                                      input_history_steps=30,
                                      input_future_steps=30,
                                      custom_scaler=type_classifier_scaler,
                                      seed=69)

test_ds = ds_gen.get_datasets(512, label_features=[], shuffle=False, with_identifier=True, stride=1)

inputs = np.concatenate([element for element in test_ds.map(lambda x,z: x).as_numpy_iterator()])
identifiers = np.concatenate([element for element in test_ds.map(lambda x,z: z).as_numpy_iterator()])

print(f"Classifying SS Types using model \"{TYPE_CLASSIFIER_DIR}\"")

type_df = pd.DataFrame(np.concatenate([identifiers.reshape(-1,2)], axis=1), columns=['ObjectID', 'TimeIndex'], dtype=np.int32)

# get predictions
preds = ew_ns_type_classifier.predict(inputs, verbose=2)
for ft_idx, feature in enumerate(['EW_Type', 'NS_Type']):
    preds_argmax = np.argmax(preds[ft_idx], axis=1)
    type_df[f'{feature}_Pred'] = preds_argmax
    type_df[f'{feature}'] = ds_gen.type_label_encoder.inverse_transform(type_df[f'{feature}_Pred'])
    
# add initial nodes
if (np.min(identifiers[:,1] > 0)):
    for obj in object_ids:
        new_index = type_df.index.max()+1
        type_df.loc[new_index] = type_df.loc[0].copy() # copy  a random row
        type_df.at[new_index, 'ObjectID'] = obj
        type_df.at[new_index, 'TimeIndex'] = 0
        type_df.at[new_index, 'EW_Type'] = 'CK'
        type_df.at[new_index, 'NS_Type'] = 'CK'
        type_df.at[new_index, 'EW_Type_Pred'] = 0
        type_df.at[new_index, 'NS_Type_Pred'] = 0
type_df = type_df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True) # DO NOT REMOVE

# determine most common type in first couple steps
for obj in object_ids:
    for dir in ['EW', 'NS']:
        obj_vals = type_df.loc[(type_df['ObjectID'] == obj), f'{dir}_Type_Pred'].to_numpy()
        # get most common value in first couple elements
        most_common = [np.bincount(obj_vals[1:81]).argmax()]
        most_common_decoded = ds_gen.type_label_encoder.inverse_transform(most_common)
        first_val_index = np.min(type_df.index[(type_df['ObjectID'] == obj)].to_list())
        type_df.at[first_val_index, f'{dir}_Type'] = most_common_decoded[0]
type_df = type_df.loc[(type_df['TimeIndex'] == 0)]

type_df = type_df.loc[(type_df['TimeIndex'] == 0)]
sub_dfs = []
for dir in ['EW', 'NS']:
    sub_df = type_df.copy()
    sub_df['Node'] = 'SS'
    sub_df['Type'] = type_df[f'{dir}_Type']
    sub_df['Direction'] = dir
    sub_df = sub_df.drop([f'{dir}_Type'], axis='columns')
    sub_dfs.append(sub_df)
type_df = pd.concat(sub_dfs)
type_df = type_df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

df_classes = df_classes.loc[(df_classes['TimeIndex']>0)]
df_classes = pd.concat([df_classes, type_df]).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

# =====================================================================================================
# Combine the classifications with the node locations
df_merged = df_locs.merge(df_classes, how='left', on = ['ObjectID', 'TimeIndex'])
df_reduced = df_merged[((df_merged['Location_EW'] == 1) & (df_merged['Direction'] == 'EW') | (df_merged['Location_NS'] == 1) & (df_merged['Direction'] == 'NS'))]
df_reduced = df_reduced.drop(['Location_EW'], axis='columns')
df_reduced = df_reduced.drop(['Location_NS'], axis='columns')

# TEMPORARY: Remove NS detections, because they probably add way more FP than TP
df_reduced = df_reduced.loc[(df_reduced['TimeIndex'] == 0) | (df_reduced['Direction'] == 'EW')]
# Only EW: 0.76 0.71 0.72 0.89 99 32 41
# Only EW new: 0.74 0.68 0.69 0.20 94 33 44
# Only SS: 0.83 0.54 0.58 0.0 83 17 70
# EW + NS: 0.69 0.74 0.73 0.89 103 47 36
# Save final results
results = df_reduced
print(f"Finished predictions, saving to \"{TEST_PREDS_FP}\"")
print(results.head(10))
results.to_csv(TEST_PREDS_FP, index=False)
print("Done. Sleeping for 6 minutes.")
time.sleep(360) # TEMPORARY FIX TO OVERCOME EVALAI BUG
print("Finished sleeping")