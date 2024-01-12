import pandas as pd
import random
import numpy as np
from tensorflow.data import Dataset
from tensorflow.math import reduce_mean, reduce_std
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
            object_labels_EW['EW_Node_Location'] = True
            object_labels_NS['NS'] = object_labels_NS['Node'] + '-' + object_labels_NS['Type']
            object_labels_NS['NS_Node'] = object_labels_NS['Node']
            object_labels_NS['NS_Type'] = object_labels_NS['Type']
            object_labels_NS['NS_Node_Location'] = True
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
            object_df['EW_Node_Location'].fillna(False, inplace=True)
            object_df['NS'].ffill(inplace=True)
            object_df['NS_Node'].ffill(inplace=True)
            object_df['NS_Type'].ffill(inplace=True)
            object_df['NS_Node_Location'].fillna(False, inplace=True)
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
                 input_history_steps=10, # how many history timesteps we get as input, including the current one
                 input_future_steps=0, # how many future timesteps we get as input
                 input_features=["Eccentricity", "Semimajor Axis (m)", "Inclination (deg)", "RAAN (deg)", "Argument of Periapsis (deg)", "Mean Anomaly (deg)", "True Anomaly (deg)", "Latitude (deg)", "Longitude (deg)", "Altitude (m)", "X (m)", "Y (m)", "Z (m)", "Vx (m/s)", "Vy (m/s)", "Vz (m/s)"],
                 with_labels=True,
                 pad_location_labels=0,
                 input_stride=1, # distance between input steps
                 padding='zero', # wether to use no/zero/same padding at the beginning and end of each df
                 transform_features=True, # wether to apply sin to certain features
                 shuffle_train_val=True,
                 seed=42,
                 train_val_split=0.8,
                 scale=True,
                 custom_scaler=None,
                 verbose=1):

        split_df = copy.deepcopy(split_df)

        self._input_features=input_features
        self._with_labels=with_labels
        self._input_feature_indices = {name:i for i, name in enumerate(input_features)}
        self._input_history_steps = input_history_steps
        self._input_future_steps = input_future_steps
        self._input_stride = input_stride
        self._padding = padding
        self._seed = seed

        if verbose>0:
            print(f"=========================Creating Generator=======================\nSeed: {self._seed}")
        
        # now, create the train and val split
        keys_list = list(split_df.keys())
        if shuffle_train_val:
            random.Random(self._seed).shuffle(keys_list) # shuffle, but with a seed for reproducability
        split_idx = int(len(keys_list) * train_val_split)
        self._train_keys = keys_list[:split_idx]
        self._val_keys = keys_list[split_idx:]
        if verbose>0:
            print(f"nTrain: {len(self._train_keys)} nVal: {len(self._val_keys)} ({len(self._train_keys)/len(keys_list):.2f})")
            print(f"Padding: {self._padding}")
            print(f"Scaling: {scale} {'(custom scaler)' if custom_scaler is not None else ''}")
            print(f"Horizons: {self._input_history_steps}-{self._input_future_steps} @ stride {self._input_stride}")

        # Make sure all val labels are also in train
        train_labels_EW, train_labels_NS, val_labels_EW, val_labels_NS = [], [], [], []
        for key in self._train_keys:
            train_labels_EW += list(split_df[key]['EW'].unique())
            train_labels_NS += list(split_df[key]['NS'].unique())
        for key in self._val_keys:
            val_labels_EW += list(split_df[key]['EW'].unique())
            val_labels_NS += list(split_df[key]['NS'].unique())
        if not (all(x in set(train_labels_EW) for x in set(val_labels_EW)) and all(x in set(train_labels_NS) for x in set(val_labels_NS))):
            print("Warning: Validation set contains labels which do not occur in training set! Maybe try different seed?")
        
        # Run sin over deg fields, to bring 0deg and 360deg next to each other (technically it would make sense to change the description, but oh my)
        # TODO: this may also be necessary for longitude
        features_to_transform = ['Mean Anomaly (deg)', 'True Anomaly (deg)', 'Argument of Periapsis (deg)'] if transform_features else []
        for key in self._train_keys + self._val_keys:
            for ft in features_to_transform:
                if ft in input_features:
                    split_df[key][ft] = np.sin(np.deg2rad(split_df[key][ft]))
        if transform_features and verbose > 0:
            print(f"Sin-Transformed features: {[ft for ft in features_to_transform if ft in input_features]}")

        #perform scaling - fit the scaler on the train data, and then scale both datasets
        if scale:
            concatenated_train_df = pd.concat([split_df[k] for k in self._train_keys], ignore_index=True)
            scaler = StandardScaler().fit(concatenated_train_df[input_features].values) if custom_scaler is None else custom_scaler
            self._scaler = scaler
            for key in self._train_keys + self._val_keys:
                split_df[key][input_features] = scaler.transform(split_df[key][input_features].values)

        # pad the location labels, making them "wider"
        if pad_location_labels>0 and with_labels:
            if verbose > 0:
                print(f"Padding node locations ({pad_location_labels})")
            for key, sub_df in split_df.items():
                timeindices = sub_df.loc[(sub_df['EW_Node_Location'] == 1)]['TimeIndex'].to_numpy()[1:] # only consider locations with timeindex > 1
                for timeindex in timeindices:
                    sub_df.loc[timeindex-pad_location_labels:timeindex+pad_location_labels, 'EW_Node_Location'] = True
                timeindices = sub_df.loc[(sub_df['NS_Node_Location'] == 1)]['TimeIndex'].to_numpy()[1:] # only consider locations with timeindex > 1
                for timeindex in timeindices:
                    sub_df.loc[timeindex-pad_location_labels:timeindex+pad_location_labels, 'NS_Node_Location'] = True

        # encode labels
        possible_node_labels = ['SS', 'ID', 'AD', 'IK']
        possible_type_labels = ['NK', 'CK', 'EK', 'HK']
        possible_combined_labels = [node_label + '-' + type_label for node_label in possible_node_labels for type_label in possible_type_labels]
        self._node_label_encoder = LabelEncoder().fit(possible_node_labels)
        self._type_label_encoder = LabelEncoder().fit(possible_type_labels)
        self._combined_label_encoder = LabelEncoder().fit(possible_combined_labels)
        if with_labels:
            for key, sub_df in split_df.items():
                sub_df['EW_Node'] = self._node_label_encoder.transform(sub_df['EW_Node'])
                sub_df['NS_Node'] = self._node_label_encoder.transform(sub_df['NS_Node'])
                sub_df['EW_Type'] = self._type_label_encoder.transform(sub_df['EW_Type'])
                sub_df['NS_Type'] = self._type_label_encoder.transform(sub_df['NS_Type'])
                sub_df['EW'] = self._combined_label_encoder.transform(sub_df['EW'])
                sub_df['NS'] = self._combined_label_encoder.transform(sub_df['NS'])
        else:
            if verbose > 0:
                print("No Labels")
        
        self._preprocessed_dataframes = split_df
  
        if verbose > 0:
            print(f"=========================Finished Generator=======================")

    def create_ds_from_dataframes(self, split_df, keys, input_features, label_features, with_identifier, input_history_steps, input_future_steps, stride, input_stride, padding):
        window_size = input_history_steps + input_future_steps
        obj_lens = {key:len(split_df[key]) - (1 if padding != 'none' else (window_size-1)) for key in keys} # the last row (ES) is removed
        strided_obj_lens = {key:int(np.ceil(obj_len/stride)) for key, obj_len in obj_lens.items()}
        n_rows = np.sum([ln for ln in strided_obj_lens.values()]) # consider stride
        inputs = np.zeros(shape=(n_rows, int(np.ceil(window_size/input_stride)), len(input_features))) # dimensions are [index, time, features]
        labels = np.zeros(shape=(n_rows, len(label_features) if label_features else 1), dtype=np.int32)
        element_identifiers = np.zeros(shape=(n_rows, 2))
        current_row = 0
        keys.sort()
        for key in keys:
            extended_df = split_df[key].copy()
            if padding != 'none':
                # to make sure that we have as many inputs as we actually have labels, we need to add rows to the beginning of the df
                # this is to ensure that we will later be "able" to predict the first entry in the labels (otherwise, we would need to skip n input_history_steps)
                nan_df_history = pd.DataFrame(np.nan, index=pd.RangeIndex(input_history_steps-1), columns=split_df[key].columns)
                nan_df_future = pd.DataFrame(np.nan, index=pd.RangeIndex(input_future_steps-1), columns=split_df[key].columns)
                extended_df = pd.concat([nan_df_history, split_df[key], nan_df_future]).reset_index(drop=True)
                if padding=='same':
                    extended_df.bfill(inplace=True) # replace NaN values in the beginning with first actual value
                    extended_df.ffill(inplace=True) # replace NaN values at the end with last actual value
                elif padding=='zero':
                    extended_df.fillna(0.0, inplace=True)
                else:
                    print(f"Warning: unknown padding method \"{padding}\"! Using zero-padding instead")
                    extended_df.fillna(0.0, inplace=True)
            
            strided_obj_len = strided_obj_lens[key]
            inputs[current_row:current_row+strided_obj_len] = np.lib.stride_tricks.sliding_window_view(extended_df[input_features].to_numpy(dtype=np.float32), window_size, axis=0).transpose(0,2,1)[::stride,::input_stride,:,]
            if label_features:
                labels[current_row:current_row+strided_obj_len] = extended_df[label_features][input_history_steps-1:-input_future_steps].to_numpy(dtype=np.int32)[::stride]
            element_identifiers[current_row:current_row+strided_obj_len] = extended_df[['ObjectID', 'TimeIndex']][input_history_steps-1:-input_future_steps].to_numpy(dtype=np.int32)[::stride,:]
            current_row+=strided_obj_len

        datasets = [Dataset.from_tensor_slices((inputs))]
        if label_features:
            datasets += [Dataset.from_tensor_slices(({feature:labels[:,ft_idx] for ft_idx, feature in enumerate(label_features)}))]
        if with_identifier:
            datasets += [Dataset.from_tensor_slices((element_identifiers))]
        
        if len(datasets)>1:
            return Dataset.zip(tuple(datasets))
        else:
            return datasets[0]

    def get_datasets(self, batch_size=None, label_features=['EW', 'EW_Node', 'EW_Type', 'NS', 'NS_Node', 'NS_Type'], with_identifier=False, shuffle=True, stride=None):
        
        # create datasets
        train_ds = self.create_ds_from_dataframes(self._preprocessed_dataframes,
                                                keys=self._train_keys,
                                                input_features=self._input_features,
                                                label_features=label_features,
                                                with_identifier=with_identifier,
                                                input_history_steps=self._input_history_steps,
                                                input_future_steps=self._input_future_steps,
                                                stride=stride,
                                                input_stride=self._input_stride,
                                                padding=self._padding)
        val_ds = self.create_ds_from_dataframes(self._preprocessed_dataframes,
                                                keys=self._val_keys,
                                                input_features=self._input_features,
                                                label_features=label_features,
                                                with_identifier=with_identifier,
                                                input_history_steps=self._input_history_steps,
                                                input_future_steps=self._input_future_steps,
                                                stride=self.stride,
                                                input_stride=self._input_stride,
                                                padding=self._padding)

        datasets = [train_ds, val_ds] if self._val_keys else [train_ds]

        if shuffle:
            datasets = [ds.shuffle(100, seed=self._seed) for ds in datasets]
        if batch_size is not None:
            datasets = [ds.batch(batch_size) for ds in datasets]
        return datasets if len(datasets)>1 else datasets[0]
    
    @property
    def input_feature_indices(self):
        return self._input_feature_indices
    
    @property
    def node_label_encoder(self):
        return self._node_label_encoder
    
    @property
    def type_label_encoder(self):
        return self._type_label_encoder
    
    @property
    def combined_label_encoder(self):
        return self._combined_label_encoder
    
    @property
    def train_keys(self):
        return self._train_keys
    
    @property
    def val_keys(self):
        return self._val_keys
    
    @property
    def scaler(self):
        return self._scaler