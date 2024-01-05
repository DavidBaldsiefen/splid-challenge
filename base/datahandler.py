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
                 label_features=['EW', 'EW_Node', 'EW_Type', 'NS', 'NS_Node', 'NS_Type'],
                 stride=1, # distance between datapoints
                 padding=True, # wether to use padding at the beginning and end of each df
                 shuffle_train_val=True,
                 seed=42,
                 train_val_split=0.8,
                 scale=True,
                 verbose=1):

        split_df = copy.deepcopy(split_df)

        self._input_features=input_features
        self._label_features=label_features
        self._input_feature_indices = {name:i for i, name in enumerate(input_features)}
        self._label_feature_indices = {name:i for i, name in enumerate(label_features)} if label_features else None
        self._seed = seed

        if verbose>0:
            print(f"=========================Creating Dataset=========================\nSeed: {self._seed}")
        
        # now, create the train and val split
        keys_list = list(split_df.keys())
        if shuffle_train_val:
            random.Random(self._seed).shuffle(keys_list) # shuffle, but with a seed for reproducability
        split_idx = int(len(keys_list) * train_val_split)
        self._train_keys = keys_list[:split_idx]
        self._val_keys = keys_list[split_idx:]
        if verbose>0:
            print(f"nTrain: {len(self._train_keys)} nVal: {len(self._val_keys)} ({len(self._train_keys)/len(keys_list):.2f})")
            print(f"Padding: {padding}")
            print(f"Scaling: {scale}")

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
        
        # Run sin over Mean and True Anomaly, to bring 0deg and 360deg next to each other (technically it would make sense to change the description, but oh my)
        for key in self._train_keys + self._val_keys:
            if ['Mean Anomaly (deg)'] in input_features:
                split_df[key]['Mean Anomaly (sin)'] = np.sin(np.deg2rad(split_df[key]['Mean Anomaly (deg)']))
            if ['True Anomaly (deg)'] in input_features:
                split_df[key]['True Anomaly (sin)'] = np.sin(np.deg2rad(split_df[key]['True Anomaly (deg)']))

        #perform scaling - fit the scaler on the train data, and then scale both datasets
        if scale:
            concatenated_train_df = pd.concat([split_df[k] for k in self._train_keys], ignore_index=True)
            scaler = StandardScaler().fit(concatenated_train_df[input_features].values)
            for key in self._train_keys + self._val_keys:
                split_df[key][input_features] = scaler.transform(split_df[key][input_features].values)

        # encode labels
        self._label_features_encoded = []
        if label_features:
            features_to_encode = ['EW_Node', 'NS_Node', 'EW_Type', 'NS_Type', 'EW', 'NS']
            possible_node_labels = ['SS', 'ES', 'ID', 'AD', 'IK']
            possible_type_labels = ['NK', 'CK', 'EK', 'HK']
            possible_combined_labels = [node_label + '-' + type_label for node_label in possible_node_labels for type_label in possible_type_labels]
            self._node_label_encoder = LabelEncoder().fit(possible_node_labels)
            self._type_label_encoder = LabelEncoder().fit(possible_type_labels)
            self._combined_label_encoder = LabelEncoder().fit(possible_combined_labels)
            for key, sub_df in split_df.items():
                sub_df['EW_Node_encoded'] = self._node_label_encoder.transform(sub_df['EW_Node'])
                sub_df['NS_Node_encoded'] = self._node_label_encoder.transform(sub_df['NS_Node'])
                sub_df['EW_Type_encoded'] = self._type_label_encoder.transform(sub_df['EW_Type'])
                sub_df['NS_Type_encoded'] = self._type_label_encoder.transform(sub_df['NS_Type'])
                sub_df['EW_encoded'] = self._combined_label_encoder.transform(sub_df['EW'])
                sub_df['NS_encoded'] = self._combined_label_encoder.transform(sub_df['NS'])
            self._label_features_encoded = [ft + '_encoded' if ft in features_to_encode else ft for ft in label_features]
            if verbose > 0:
                print(f"Labels: {self._label_features_encoded}")
        else:
            if verbose > 0:
                print("No Labels")

        # create datasets using timewindows
        self._train_ds = self.create_ds_from_dataframes(split_df, self._train_keys, self._input_features, self._label_features_encoded, input_history_steps, input_future_steps, stride, padding)
        self._val_ds = self.create_ds_from_dataframes(split_df, self.val_keys, self._input_features, self._label_features_encoded, input_history_steps, input_future_steps, stride, padding) if self.val_keys else None
                    
        if verbose > 0:
            print(f"=========================Finished Dataset=========================")

    def create_ds_from_dataframes(self, split_df, keys, input_features, label_features, input_history_steps, input_future_steps, stride, padding):
        n_rows = np.sum([len(split_df[key]) - (1 if padding else (input_history_steps+input_future_steps-1)) for key in keys]) # the last row (ES) is removed
        inputs = np.zeros(shape=(n_rows, input_history_steps+input_future_steps, len(input_features)))
        labels = np.zeros(shape=(n_rows, len(label_features) if label_features else 1), dtype=np.int32)
        element_identifiers = np.zeros(shape=(n_rows, 2))
        fast_compute = stride==1 and input_history_steps == input_future_steps
        if not fast_compute:
            print("Warning: asymmetrical window size leads to slow computation during dataset creation")
        current_row = 0
        for key in keys:
            extended_df = split_df[key].copy()
            if padding:
                # to make sure that we have as many inputs as we actually have labels, we need to add rows to the beginning of the df
                # this is to ensure that we will later be "able" to predict the first entry in the labels (otherwise, we would need to skip n input_history_steps)
                nan_df_history = pd.DataFrame(np.nan, index=pd.RangeIndex(input_history_steps-1), columns=split_df[key].columns)
                nan_df_future = pd.DataFrame(np.nan, index=pd.RangeIndex(input_future_steps-1), columns=split_df[key].columns)
                extended_df = pd.concat([nan_df_history, split_df[key], nan_df_future]).reset_index(drop=True)
                extended_df.bfill(inplace=True) # replace NaN values in the beginning with first actual value
                extended_df.ffill(inplace=True) # replace NaN values at the end with last actual value

            # for stride 1 and symmetrical horizons we can use the faster np implementation
            if fast_compute:
                window_size = input_history_steps + input_future_steps
                obj_len = len(split_df[key]) - (1 if padding else (window_size-1))
                inputs[current_row:current_row+obj_len] = np.lib.stride_tricks.sliding_window_view(extended_df[input_features].to_numpy(dtype=np.float32), window_size, axis=0).transpose(0,2,1)
                if label_features:
                    labels[current_row:current_row+obj_len] = extended_df[label_features][input_history_steps-1:-input_future_steps].to_numpy(dtype=np.int32)
                element_identifiers[current_row:current_row+obj_len] = extended_df[['ObjectID', 'TimeIndex']][input_history_steps-1:-input_future_steps].to_numpy(dtype=np.int32)
                current_row+=obj_len
            else:
                current_index = input_history_steps
                while(current_index+input_future_steps <= extended_df.shape[0]):
                    inputs[current_row] = extended_df[input_features][current_index-input_history_steps:current_index+input_future_steps].to_numpy(dtype=np.float32)
                    if label_features:
                        labels[current_row] = extended_df[label_features][current_index-1:current_index] # -1 as input slice indexing excludes last index, need to slice for 1 element to prevent keyerror
                    element_identifiers[current_row] = extended_df[['ObjectID', 'TimeIndex']][current_index-1:current_index].to_numpy(dtype=np.int32) # see above
                    current_index += stride
                    current_row+=1
        if label_features:
            ds = Dataset.zip((Dataset.from_tensor_slices((inputs)),
                            Dataset.from_tensor_slices(({feature:labels[:,ft_idx] for ft_idx, feature in enumerate(label_features)})),
                            Dataset.from_tensor_slices((element_identifiers))))
            return ds
        else:
            ds = Dataset.zip((Dataset.from_tensor_slices((inputs)),
                            Dataset.from_tensor_slices((element_identifiers))))
            return ds

    def get_datasets(self, batch_size=None, label_features=['EW', 'EW_Node', 'EW_Type', 'NS', 'NS_Node', 'NS_Type'], keep_identifier=False, shuffle=True):
        # returns copies of the datasets
        datasets = [self._train_ds, self._val_ds]
        if self._val_ds is None:
            datasets = [self._train_ds]
        datasets = [self.set_ds_outputs(ds, label_features, keep_identifier) for ds in datasets]
        if shuffle:
            datasets = [ds.shuffle(ds.cardinality(), seed=self._seed) for ds in datasets]
        if batch_size is not None:
            datasets = [ds.batch(batch_size) for ds in datasets]
        return datasets if len(datasets)>1 else datasets[0]
    
    def set_ds_outputs(self, ds, label_features, keep_identifier):
        if not set(label_features).issubset(self._label_features):
            print(f"Warning: labels {label_features} ar not contained in the labels of DatasetGenerator: {self._label_features}. Proceeding with all labels contained in DatasetGenerator")
            label_features = self._label_features
        # make sure we consider the fact that there may not be any labels
        if self._label_features:
            label_feature_indices = [self._label_feature_indices[feat] for feat in label_features]
            features_to_encode = ['EW_Node', 'NS_Node', 'EW_Type', 'NS_Type', 'EW', 'NS']
            def output_mapper(x,y,z):
                #outputs = [x] + [y[i] for i in label_feature_indices]
                outputs = [x] + [{ft: (y[ft + '_encoded'] if ft in features_to_encode else y[ft]) for ft in label_features}]
                #outputs = [x] + [tf.gather(y, label_feature_indices, axis=0)]
                if keep_identifier: outputs += [z]
                return tuple(outputs)
            return ds.map(output_mapper)
        else:
            def output_mapper(x,z):
                outputs = [x]
                if keep_identifier: outputs += [z]
                return tuple(outputs)
            return ds.map(output_mapper)
        
    def get_dataset_statistics(self, train=True, labels=False):
        input_features = np.array([(ft.numpy() if not labels else lb.numpy()) for ft, lb in (self.train_ds if train else self.val_ds)])
        ds_shape = input_features.shape
        mean = reduce_mean(input_features, axis=[0,1]).numpy()
        stddev = reduce_std(input_features, axis=[0,1]).numpy()
        print(f"Dataset Shape: {ds_shape}\nMean Values: {mean} => {np.mean(mean)}\nStddev Values: {stddev} => {np.mean(stddev)}")
    
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