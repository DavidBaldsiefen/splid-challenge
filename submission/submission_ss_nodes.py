import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import pickle
import time
import sys

from base import utils, datahandler

SS_TYPE_CLASSIFIER_DIR = Path('/models/ss_type_classifier.hdf5')
SS_TYPE_CLASSIFIER_SCALER_DIR = Path('/models/ss_type_classifier_scaler.pkl')

TEST_DATA_DIR = Path('/dataset/test/')
TEST_PREDS_FP = Path('/submission/submission.csv')

# TEMPORARY, REMOVE!
# TEST_DATA_DIR = Path('submission/dataset/test/')
# TEST_PREDS_FP = Path('submission/submission/submission.csv')

# Load Data
split_dataframes = datahandler.load_and_prepare_dataframes(TEST_DATA_DIR, labels_dir=None)
print(f"Loaded {len(split_dataframes.keys())} dataset files from \"{TEST_DATA_DIR}\". Creating dataset")

reduced_input_features = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)',
       'Argument of Periapsis (deg)', 'True Anomaly (deg)', 'Latitude (deg)',
       'Longitude (deg)', 'Altitude (m)']

# ============================================================================================
object_ids = list(map(int, list(split_dataframes.keys())))

# Add the Type classifier for the initial predictions
ss_type_classifier = tf.keras.models.load_model(SS_TYPE_CLASSIFIER_DIR)
ss_type_classifier_scaler = pickle.load(open(SS_TYPE_CLASSIFIER_SCALER_DIR, 'rb'))

ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                      input_features=reduced_input_features,
                                      with_labels=False,
                                      train_val_split=1.0,
                                      input_stride=2,
                                      padding='none',
                                      input_history_steps=1,
                                      input_future_steps=400,
                                      custom_scaler=ss_type_classifier_scaler,
                                      seed=69)

test_ds = ds_gen.get_datasets(256, label_features=[], shuffle=False, with_identifier=True, stride=1)

inputs = np.concatenate([element for element in test_ds.map(lambda x,z: x).as_numpy_iterator()])
identifiers = np.concatenate([element for element in test_ds.map(lambda x,z: z).as_numpy_iterator()])

print(f"Classifying SS Types using model \"{SS_TYPE_CLASSIFIER_DIR}\"")

type_df = pd.DataFrame(np.concatenate([identifiers.reshape(-1,2)], axis=1), columns=['ObjectID', 'TimeIndex'], dtype=np.int32)


# get predictions
preds = ss_type_classifier.predict(inputs, verbose=2)
for ft_idx, feature in enumerate(['EW_Type', 'NS_Type']):
    preds_argmax = np.argmax(preds[ft_idx], axis=1)
    type_df[f'{feature}_Pred'] = preds_argmax
    type_df[f'{feature}'] = ds_gen.type_label_encoder.inverse_transform(type_df[f'{feature}_Pred'])

type_df = type_df.loc[(type_df['TimeIndex'] == 0)]
sub_dfs = []
for dir in ['EW', 'NS']:
    sub_df = type_df.copy()

    sub_df['Direction'] = dir
    sub_df['Node'] = 'SS'
    sub_df['Type'] = type_df[f'{dir}_Type']
    
    sub_df = sub_df.drop([f'{dir}_Type'], axis='columns')
    sub_dfs.append(sub_df)
type_df = pd.concat(sub_dfs)
type_df = type_df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

# ================================================================================================

# Save final results
results = type_df
print(f"Finished predictions, saving to \"{TEST_PREDS_FP}\"")
print(results.head(10))
results.to_csv(TEST_PREDS_FP, index=False)
print("Done. Sleeping for 6 minutes.")
time.sleep(360) # TEMPORARY FIX TO OVERCOME EVALAI BUG
print("Finished sleeping")