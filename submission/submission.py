import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import random
import time

from base import utils, datahandler

TRAINED_MODEL_DIR = Path('/models/tmp_model.hdf5')
TEST_DATA_DIR = Path('/dataset/test/')
TEST_PREDS_FP = Path('/submission/submission.csv')

# Load Dataset without labels


split_dataframes = datahandler.load_and_prepare_dataframes(TEST_DATA_DIR, labels_dir=None)
print(f"Loaded {len(split_dataframes.keys())} dataset files from \"{TEST_DATA_DIR}\". Creating dataset")

reduced_input_features = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)',
       'Argument of Periapsis (deg)', 'True Anomaly (deg)', 'Latitude (deg)',
       'Longitude (deg)', 'Altitude (m)']
ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes, input_features=reduced_input_features,
                                      label_features=[],
                                      train_val_split=1.0, stride=1,
                                      input_history_steps=8, input_future_steps=8, seed=69)
test_ds = ds_gen.get_datasets(128, label_features=[], shuffle=False, keep_identifier=True)

# Load models
print(f"Loading Model \"{TRAINED_MODEL_DIR}\"")
model = tf.keras.models.load_model(TRAINED_MODEL_DIR)

# Get Predictions
print(f"Accumulating Predictions")
inputs = np.concatenate([element for element in test_ds.map(lambda x,z: x).as_numpy_iterator()])
identifiers = np.concatenate([element for element in test_ds.map(lambda x,z: z).as_numpy_iterator()])
preds = model.predict(inputs, verbose=2)
preds_argmax = np.argmax(preds, axis=1).reshape(-1,1)

# Create prediction dataframe
df = pd.DataFrame(np.concatenate([identifiers.reshape(-1,2)], axis=1), columns=['ObjectID', 'TimeIndex'], dtype=np.int32)
for ft_idx, feature in enumerate(['EW', 'NS']):
        preds_argmax = np.argmax(preds[ft_idx], axis=1)
        preds_decoded = ds_gen.combined_label_encoder.inverse_transform(preds_argmax)
        df[f'Predicted_{feature}'] = preds_decoded

# Smooth prediction
history_smoothing_steps=5
future_smoothing_steps=5
print(f"Smoothing for {history_smoothing_steps}-{future_smoothing_steps}")
df = utils.smooth_predictions(df, past_steps=history_smoothing_steps, fut_steps=future_smoothing_steps, verbose=0)

# Create final results
results = utils.convert_classifier_output(df).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
print(f"Finished predictions, saving to \"{TEST_PREDS_FP}\"")
print(results.head(10))
results.to_csv(TEST_PREDS_FP, index=False)
print("Done. Sleeping for 6 minutes.")
time.sleep(360) # TEMPORARY FIX TO OVERCOME EVALAI BUG
print("Finished sleeping")