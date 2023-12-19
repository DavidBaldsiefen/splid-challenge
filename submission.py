import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import random

from base import utils, datahandler

TRAINED_MODEL_EW_DIR = Path('/trained_models/dense_model_ew.h5')
TRAINED_MODEL_NS_DIR = Path('/trained_models/dense_model_ns.h5')
TEST_DATA_DIR = Path('/dataset/test/')
TEST_PREDS_FP = Path('/submission/submission.csv')

# Load Dataset without labels
split_dataframes = datahandler.load_and_prepare_dataframes(TEST_DATA_DIR, labels_dir=None)
print(f"Loaded {len(split_dataframes.keys())} dataset files from \"{TEST_DATA_DIR}\". Creating dataset")

ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes, train_val_split=1.0, stride=1, input_steps=15)
test_EW, test_NS = ds_gen.get_datasets(32, shuffle=False, with_identifiers=True)

# Load models
print(f"Loading Models \"{TRAINED_MODEL_EW_DIR}\" and \"{TRAINED_MODEL_NS_DIR}\"")
model_EW = tf.keras.models.load_model(TRAINED_MODEL_EW_DIR)
model_NS = tf.keras.models.load_model(TRAINED_MODEL_NS_DIR)

# Get Predictions
print(f"Accumulating Predictions")
inputs = np.concatenate([element for element in test_EW.map(lambda x,y,z: x).as_numpy_iterator()])
identifiers = np.concatenate([element for element in test_EW.map(lambda x,y,z: z).as_numpy_iterator()])
preds_ew = model_EW.predict(inputs)
preds_ew_argmax = np.argmax(preds_ew, axis=1).reshape(-1,1)
preds_ns = model_NS.predict(inputs)
preds_ns_argmax = np.argmax(preds_ns, axis=1).reshape(-1,1)

# Create prediction dataframe (TODO: make sure labelencoder is always the same!)
df = pd.DataFrame(np.concatenate([identifiers.reshape(-1,2), preds_ew_argmax, preds_ns_argmax], axis=1), columns=['ObjectID', 'TimeIndex', 'Predicted_EW', 'Predicted_NS'], dtype=np.int32)
df['Predicted_EW'] = ds_gen.label_encoder.inverse_transform(df['Predicted_EW'])
df['Predicted_NS'] = ds_gen.label_encoder.inverse_transform(df['Predicted_NS'])

# Smooth prediction
smoothed_df = utils.smooth_predictions(df, past_steps=3, fut_steps=4)

# Create final results
results = utils.convert_classifier_output(smoothed_df).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
print(f"Finished predictions, saving to \"{TEST_PREDS_FP}\"")
print(results.head(10))
results.to_csv(TEST_PREDS_FP, index=False)
print("Done.")