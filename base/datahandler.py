import pandas as pd
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
from tqdm import tqdm
import copy


def load_and_prepare_dataframes(data_dir, labels_dir):

    # Load the labels
    if labels_dir is None:
        print("loading data without labels")
    labels = None if labels_dir is None else pd.read_csv(labels_dir)

    # load the input data from each csv
    data_files = list(data_dir.glob('*.csv'))
    
    object_dataframes = {}

    # Check if test_data is empty
    if not data_files:
        raise ValueError(f'No csv files found in {data_dir}')
    for data_file in data_files:
        object_id = int(data_file.stem)
        object_df = pd.read_csv(data_file)
        object_df['ObjectID'] = int(data_file.stem)
        object_df['TimeIndex'] = range(len(object_df))

        # find all labels associated with the object
        if labels_dir is not None:
            object_labels = labels.loc[labels['ObjectID'] == object_id]

            # Separate the 'EW' and 'NS' types in the ground truth
            object_labels_EW = object_labels[object_labels['Direction'] == 'EW'].copy()
            object_labels_NS = object_labels[object_labels['Direction'] == 'NS'].copy()
            
            # Create 'EW' and 'NS' labels
            # TODO: "ES"-rows are just dropped here
            object_labels_EW['EW'] = object_labels_EW['Node'] + '-' + object_labels_EW['Type']
            object_labels_EW['EW_Node'] = object_labels_EW['Node']
            object_labels_EW['EW_Type'] = object_labels_EW['Type']
            object_labels_NS['NS'] = object_labels_NS['Node'] + '-' + object_labels_NS['Type']
            object_labels_NS['NS_Node'] = object_labels_NS['Node']
            object_labels_NS['NS_Type'] = object_labels_NS['Type']
            object_labels_EW.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)
            object_labels_NS.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)

            # Merge the input data with the ground truth
            object_df = pd.merge(object_df, 
                                object_labels_EW.sort_values('TimeIndex'), 
                                on=['TimeIndex', 'ObjectID'],
                                how='left')
            object_df = pd.merge_ordered(object_df, 
                                        object_labels_NS.sort_values('TimeIndex'), 
                                        on=['TimeIndex', 'ObjectID'],
                                        how='left')
            
            # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
            object_df['EW'].ffill(inplace=True)
            object_df['EW_Node'].ffill(inplace=True)
            object_df['EW_Type'].ffill(inplace=True)
            object_df['NS'].ffill(inplace=True)
            object_df['NS_Node'].ffill(inplace=True)
            object_df['NS_Type'].ffill(inplace=True)
        else:
            object_df['EW'] = 'UNKNOWN'
            object_df['EW_Node'] = 'UNKNOWN'
            object_df['EW_Type'] = 'UNKNOWN'
            object_df['NS'] = 'UNKNOWN'
            object_df['NS_Node'] = 'UNKNOWN'
            object_df['NS_Type'] = 'UNKNOWN'
        

        object_dataframes[str(object_id)] = object_df

    return object_dataframes

# now we need to create the datasets using a sliding window approach
# each window contains the input features over the last n feature steps, and tries to predict the current label (either EW or NS)
# TODO: enable use of individual label features - such as Node and Type Label
class DatasetGenerator():
    def __init__(self,
                 split_df,
                 input_steps=10, # how many input timesteps we get
                 input_features=["Eccentricity", "Semimajor Axis (m)", "Inclination (deg)", "RAAN (deg)", "Argument of Periapsis (deg)", "Mean Anomaly (deg)", "True Anomaly (deg)", "Latitude (deg)", "Longitude (deg)", "Altitude (m)", "X (m)", "Y (m)", "Z (m)", "Vx (m/s)", "Vy (m/s)", "Vz (m/s)"],
                 stride=1, # distance between datapoints
                 shuffle_train_val=True,
                 seed=42,
                 train_val_split=0.8,
                 scale=True,
                 verbose=1):

        split_df = copy.deepcopy(split_df)

        self.input_features=input_features
        self._input_feature_indices = {name:i for i, name in enumerate(input_features)}
        self.seed = seed
        
        # now, create the train and val split
        keys_list = list(split_df.keys())
        if shuffle_train_val:
            random.Random(self.seed).shuffle(keys_list) # shuffle, but with a seed for reproducability
        split_idx = int(len(keys_list) * train_val_split)
        self._train_keys = keys_list[:split_idx]
        self._val_keys = keys_list[split_idx:]
        if verbose>0:
            print(f"Creating dataset from {len(self._train_keys)} train and {len(self._val_keys)} val objects")

        #perform scaling - fit the scaler on the train data, and then scale both datasets
        if scale:
            concatenated_train_df = pd.concat([split_df[k] for k in self._train_keys], ignore_index=True)
            scaler = StandardScaler().fit(concatenated_train_df[input_features].values)
            for key in self._train_keys + self._val_keys:
                split_df[key][input_features] = scaler.transform(split_df[key][input_features].values)

        # encode labels
        possible_labels = ['UNKNOWN']
        for node_label in ['SS', 'ES', 'ID', 'AD', 'IK']:
            for type_label in ['NK', 'CK', 'EK', 'HK']:
                possible_labels.append(node_label + '-' + type_label)
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(possible_labels)
        for key, sub_df in split_df.items():
            sub_df['EW_encoded'] = self._label_encoder.transform(sub_df['EW'])
            sub_df['NS_encoded'] = self._label_encoder.transform(sub_df['NS'])

        # create datasets using timewindows
        self._train_ds_EW, self._train_ds_NS = self.create_ds_from_dataframes(split_df, self._train_keys, input_features, input_steps, stride)
        if self.val_keys:
            self._val_ds_EW, self._val_ds_NS = self.create_ds_from_dataframes(split_df, self.val_keys, input_features, input_steps, stride)
        else:
            self._val_ds_EW, self._val_ds_NS = None, None
                    
        if verbose > 0:
            print(f"Created datasets with seed {self.seed}")

    def create_ds_from_dataframes(self, split_df, keys, input_features, input_steps, stride):
        n_rows = np.sum([len(split_df[key]) for key in keys])
        inputs = np.zeros(shape=(n_rows, input_steps, len(input_features)))
        labels_EW, labels_NS = np.zeros(shape=(n_rows, 1)), np.zeros(shape=(n_rows, 1))
        element_identifiers = np.zeros(shape=(n_rows, 2))
        current_row = 0
        for key in keys:
            # to make sure that we have as many inputs as we actually have labels, we need to add rows to the beginning of the df
            # this is to ensure that we will later be "able" to predict the first entry in the labels (otherwise, we would need to skip n input_steps)
            extended_df = pd.concat([pd.DataFrame(np.nan, index=pd.RangeIndex(input_steps-1), columns=split_df[key].columns), split_df[key]]).reset_index(drop=True)
            extended_df.bfill(inplace=True) # replace NaN values with first actual value
            current_index = input_steps
            while(current_index <= extended_df.shape[0]):
                inputs[current_row] = extended_df[input_features][current_index-input_steps:current_index].to_numpy(dtype=np.float32)
                labels_EW[current_row] = extended_df['EW_encoded'][current_index-1] # -1 as input slice indexing excludes last index
                labels_NS[current_row] = extended_df['NS_encoded'][current_index-1]
                element_identifiers[current_row] = extended_df[['ObjectID', 'TimeIndex']][current_index-1:current_index].to_numpy(dtype=np.int32) # need to slice for 1 element to prevent keyerror
                current_index += stride
                current_row+=1
        ds_EW = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices((inputs)),
                                        tf.data.Dataset.from_tensor_slices((labels_EW)),
                                        tf.data.Dataset.from_tensor_slices((element_identifiers))))
        ds_NS = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices((inputs)),
                                        tf.data.Dataset.from_tensor_slices((labels_NS)),
                                        tf.data.Dataset.from_tensor_slices((element_identifiers))))
        return ds_EW, ds_NS

    def get_datasets(self, batch_size=None, shuffle=True, with_identifiers=False):
        # returns copies of the datasets
        datasets = [self._train_ds_EW, self._train_ds_NS, self._val_ds_EW, self._val_ds_NS]
        if self._val_ds_EW is None:
            datasets = [self._train_ds_EW, self._train_ds_NS]
        if not with_identifiers:
            datasets = [self.remove_identifier(ds) for ds in datasets]
        if shuffle:
            datasets = [ds.shuffle(ds.cardinality(), seed=self.seed) for ds in datasets]
        return [ds.batch(batch_size) if batch_size is not None else ds for ds in datasets]
    
    def remove_identifier(self, ds):
        def rm_id_func(x,y,z):
            return x,y
        return ds.map(rm_id_func)
        
    def get_dataset_statistics(self, train=True, labels=False):
        input_features = np.array([(ft.numpy() if not labels else lb.numpy()) for ft, lb in (self.train_ds if train else self.val_ds)])
        ds_shape = input_features.shape
        mean = tf.reduce_mean(input_features, axis=[0,1]).numpy()
        stddev = tf.math.reduce_std(input_features, axis=[0,1]).numpy()
        print(f"Dataset Shape: {ds_shape}\nMean Values: {mean} => {np.mean(mean)}\nStddev Values: {stddev} => {np.mean(stddev)}")
    
    @property
    def input_feature_indices(self):
        return self._input_feature_indices
    
    @property
    def label_encoder(self):
        return self._label_encoder
    
    @property
    def train_keys(self):
        return self._train_keys
    
    @property
    def val_keys(self):
        return self._val_keys