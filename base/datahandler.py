import pandas as pd
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone
import copy
import gc


def load_and_prepare_dataframes(data_dir, labels_dir, dtype=np.float32):
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
        object_df = pd.read_csv(data_file,
                                dtype={
                                    'Eccentricity' : dtype,
                                    'Semimajor Axis (m)' : dtype,
                                    'Inclination (deg)' : dtype,
                                    'RAAN (deg)' : dtype,
                                    'Argument of Periapsis (deg)' : dtype,
                                    'True Anomaly (deg)' : dtype,
                                    'Latitude (deg)' : dtype,
                                    'Longitude (deg)' : dtype,
                                    'Altitude (m)' : dtype,
                                    'X (m)' : dtype,
                                    'Y (m)' : dtype,
                                    'Z (m)' : dtype,
                                    'Vx (m/s)' : dtype,
                                    'Vy (m/s)' : dtype,
                                    'Vz (m/s)' : dtype,
                                })
        object_df['ObjectID'] = int(data_file.stem)
        object_df['TimeIndex'] = range(len(object_df))

        # find all labels associated with the object
        if labels_dir is not None:
            object_labels = labels.loc[labels['ObjectID'] == object_id]

            # Separate the 'EW' and 'NS' types in the ground truth
            object_labels_EW = object_labels[object_labels['Direction'] == 'EW'].copy(deep=False)
            object_labels_NS = object_labels[object_labels['Direction'] == 'NS'].copy(deep=False)
            
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

    del data_files
    if labels_dir is not None:
        del object_labels_EW
        del object_labels_NS

    gc.collect()

    return object_dataframes

# now we need to create the datasets using a sliding window approach
# each window contains the input features over the last n feature steps, and tries to predict the current label (either EW or NS)
class DatasetGenerator():
    def __init__(self,
                 split_df,
                 exclude_objects=[], # option to exclude certain objects (after the train-val split has been performed) which may be erronous
                 input_history_steps=10, # how many history timesteps we get as input, including the current one
                 input_future_steps=0, # how many future timesteps we get as input
                 non_transform_features=["Eccentricity", "Semimajor Axis (m)", "Inclination (deg)", "RAAN (deg)", "Argument of Periapsis (deg)", "True Anomaly (deg)", "Latitude (deg)", "Longitude (deg)", "Altitude (m)", "X (m)", "Y (m)", "Z (m)", "Vx (m/s)", "Vy (m/s)", "Vz (m/s)"],
                 sin_transform_features=[],
                 sin_cos_transform_features=[],
                 diff_transform_features=[],
                 overview_features_mean=[],
                 overview_features_std=[],
                 add_daytime_feature=False,
                 add_yeartime_feature=False,
                 add_linear_timeindex=False,
                 with_labels=True,
                 pad_location_labels=0,
                 nonbinary_padding=[],
                 input_stride=1, # distance between input steps
                 padding='none', # wether to use none/zero/same padding at the beginning and end of each df
                 shuffle_train_val=True,
                 seed=42,
                 train_val_split=0.8,
                 unify_value_ranges=False,
                 scale=True,
                 per_object_scaling=False,
                 custom_scaler=None,
                 node_class_multipliers={},
                 input_dtype=np.float32,
                 sort_inputs=True,
                 verbose=1,
                 deepcopy=True):

        split_df = copy.deepcopy(split_df) if deepcopy else split_df

        self._input_features=non_transform_features
        self._with_labels=with_labels
        self._input_history_steps = input_history_steps
        self._input_future_steps = input_future_steps
        self._input_stride = input_stride
        self._input_dtype = input_dtype
        self._padding = padding
        self._seed = seed
        self._overview_features_mean=overview_features_mean
        self._overview_features_std=overview_features_std

        if verbose>0:
            print(f"=========================Creating Generator=======================\nSeed: {self._seed}")
        
        # now, create the train and val split
        keys_list = list(split_df.keys())
        if shuffle_train_val:
            random.Random(self._seed).shuffle(keys_list) # shuffle, but with a seed for reproducability
        split_idx = int(len(keys_list) * train_val_split)
        self._train_keys = keys_list[:split_idx]
        self._val_keys = keys_list[split_idx:]

        # exclude certain objects, e.g. if it is known that they are erronous
        if exclude_objects:
            self._train_keys = [key for key in self._train_keys if not (key in exclude_objects)]
            self._val_keys = [key for key in self._val_keys if not (key in exclude_objects)]
            for key in exclude_objects:
                split_df.pop(key, None)

        if verbose>0:
            print(f"nTrain: {len(self._train_keys)} nVal: {len(self._val_keys)} ({len(self._train_keys)/len(keys_list):.2f})")
            print(f"Padding: {self._padding}")
            print(f"Horizons: {self._input_history_steps}-{self._input_future_steps} @ stride {self._input_stride}")
            print(f"Scaling: {scale} {'(custom scaler)' if custom_scaler is not None else ''} {'(per-object)' if per_object_scaling is True else ''}")
            if node_class_multipliers:
                print(f"Node Class Multipliers: {node_class_multipliers}")

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

        # for some objects (221-250), Longitude and True Anomaly are shifted for some reason
        if unify_value_ranges:
            if verbose > 0:
                print(f"Limiting True Anomaly to [0.0, 360.0] and Longitude to [-180.0, 180.0]")
            for key in self._train_keys + self._val_keys:
                split_df[key].loc[split_df[key]['True Anomaly (deg)'] < 0.0, 'True Anomaly (deg)'] = split_df[key].loc[split_df[key]['True Anomaly (deg)'] < 0.0, 'True Anomaly (deg)'] + 360.0
                split_df[key].loc[split_df[key]['Longitude (deg)'] > 180.0, 'Longitude (deg)'] = split_df[key].loc[split_df[key]['Longitude (deg)'] > 180.0, 'Longitude (deg)'] - 360.0
        # Run sin+cos over deg fields, to bring 0deg and 360deg next to each other
        if sin_transform_features or sin_cos_transform_features:
            if verbose > 0:
                print(f"Sin-Transforming features: {sin_transform_features}")
                print(f"Sin-Cos-Transforming features: {sin_cos_transform_features}")
            for ft in sin_transform_features + sin_cos_transform_features:
                newft_sin = ft[:-5] + '(sin)'
                newft_cos = ft[:-5] + '(cos)'
                for key in self._train_keys + self._val_keys:
                    split_df[key][newft_sin] = np.sin(np.deg2rad(split_df[key][ft]))
                    if ft in sin_cos_transform_features:
                        split_df[key][newft_cos] = np.cos(np.deg2rad(split_df[key][ft]))
                self._input_features.append(newft_sin)
                if ft in sin_cos_transform_features:
                    self._input_features.append(newft_cos)

        if diff_transform_features:
            if verbose > 0:
                print(f"Diff Transforming features: {diff_transform_features}")
            for ft in diff_transform_features:
                newft = ft + ' (diff)'
                wraparound_offset = ([180, -180] if ft == 'Longitude (deg)' else
                                     [270, -90] if ft in ['True Anomaly (deg)'] else # Longitude should usually increase, but small decreases are possible
                                     [180, -180] if ft in ['Argument of Periapsis (deg)', 'RAAN (deg)'] else
                                     [])
                if verbose > 1:
                    print(f"Wraparound offset for ft {ft}: {wraparound_offset}")
                for key in self._train_keys + self._val_keys:
                    diff_vals = np.diff(split_df[key][ft], prepend=split_df[key][ft][0])
                    if wraparound_offset:
                        diff_vals[diff_vals > wraparound_offset[0]] -= 360
                        diff_vals[diff_vals < wraparound_offset[1]] += 360
                    split_df[key][newft] = np.diff(split_df[key][ft], prepend=split_df[key][ft][0])
                self._input_features.append(newft)

        if add_daytime_feature:
            if verbose>0:
                print("Adding daytime features.")
            for key in self._train_keys + self._val_keys:
                split_df[key]['Datetime'] = pd.to_datetime(split_df[key]['Timestamp'], format='%Y-%m-%d %H:%M:%S.%fZ', utc=True)
                split_df[key]['Epoch'] = (split_df[key]['Datetime'] - datetime(1970,1,1, tzinfo=timezone.utc)).dt.total_seconds()
                siderial_day = 86164 # seconds in a sidereal day
                split_df[key]['Epoch Day (sin)'] = np.sin((2*np.pi*split_df[key]['Epoch'])/siderial_day)
                split_df[key]['Epoch Day (cos)'] = np.cos((2*np.pi*split_df[key]['Epoch'])/siderial_day)
                split_df[key].drop(columns=['Datetime', 'Epoch'], inplace=True)
            self._input_features.append('Epoch Day (sin)')
            self._input_features.append('Epoch Day (cos)')

        if add_yeartime_feature:
            if verbose>0:
                print("Adding yeartime features.")
            for key in self._train_keys + self._val_keys:
                split_df[key]['Datetime'] = pd.to_datetime(split_df[key]['Timestamp'], format='%Y-%m-%d %H:%M:%S.%fZ', utc=True)
                split_df[key]['Epoch'] = (split_df[key]['Datetime'] - datetime(1970,1,1, tzinfo=timezone.utc)).dt.total_seconds()
                seconds_per_siderial_year = 86400 * 365.24 # seconds in a sidereal year
                split_df[key]['Epoch Year Fraction (sin)'] = np.sin((2*np.pi*split_df[key]['Epoch'])/seconds_per_siderial_year)
                split_df[key]['Epoch Year Fraction (cos)'] = np.cos((2*np.pi*split_df[key]['Epoch'])/seconds_per_siderial_year)
                split_df[key].drop(columns=['Datetime', 'Epoch'], inplace=True)
            self._input_features.append('Epoch Year Fraction (sin)')
            self._input_features.append('Epoch Year Fraction (cos)')

        if add_linear_timeindex:
            if verbose>0:
                print("Adding linear timeindex.")
            for key in self._train_keys + self._val_keys:
                split_df[key]['LinearTimeIndex'] = np.linspace(-1.0, 1.0, len(split_df[key]))
            self._input_features.append('LinearTimeIndex')

        #perform scaling - fit the scaler on the train data, and then scale both datasets
        if scale:
            if verbose>1:
                print("Scaling now.")
            if per_object_scaling:
                for key in self._train_keys + self._val_keys:
                    split_df[key][self._input_features] = StandardScaler().fit_transform(split_df[key][self._input_features].values)
            else:
                concatenated_train_df = pd.concat([split_df[k] for k in self._train_keys], ignore_index=True)
                scaler = StandardScaler().fit(concatenated_train_df[self._input_features].values) if custom_scaler is None else custom_scaler
                self._scaler = scaler
                for key in self._train_keys + self._val_keys:
                    split_df[key][self._input_features] = scaler.transform(split_df[key][self._input_features].values)

        # pad the location labels, making them "wider"
        if pad_location_labels>0 and with_labels:
            if verbose > 0:
                print(f"Padding node locations ({pad_location_labels})")
            for key, sub_df in split_df.items():
                timeindices = sub_df.loc[(sub_df['EW_Node_Location'] == 1), 'TimeIndex'].to_numpy()[1:] # only consider locations with timeindex > 1
                for timeindex in timeindices:
                    # TODO: Using timeindex as index? this correct???
                    sub_df.loc[timeindex-pad_location_labels:timeindex+pad_location_labels, 'EW_Node_Location'] = True
                timeindices = sub_df.loc[(sub_df['NS_Node_Location'] == 1), 'TimeIndex'].to_numpy()[1:] # only consider locations with timeindex > 1
                for timeindex in timeindices:
                    sub_df.loc[timeindex-pad_location_labels:timeindex+pad_location_labels, 'NS_Node_Location'] = True

        if nonbinary_padding:
            if verbose>1:
                print("Adding nb padding now.")
            pad_extended  = nonbinary_padding[::-1] + nonbinary_padding[1:]
            pad_len = len(nonbinary_padding)-1
            if verbose > 0:
                print(f"Padding node locations in non-binary fashion ({pad_extended})")
            for key, sub_df in split_df.items():
                for dir in ['EW', 'NS']:
                    sub_df[f'{dir}_Node_Location_nb'] = sub_df[f'{dir}_Node_Location'].astype(np.float32)
                    timeindices = sub_df.loc[(sub_df[f'{dir}_Node_Location_nb'] == 1), 'TimeIndex'].to_numpy()[1:] # only consider locations with timeindex > 1
                    for timeindex in timeindices:
                        node = sub_df.loc[timeindex, f'{dir}_Node']
                        scaling_factor = 1.0
                        if node_class_multipliers:
                            scaling_factor = node_class_multipliers[node]
                        #print(sub_df.loc[timeindex-pad_len:timeindex + pad_len, 'EW_Node_Location_nb'])
                        sub_df.loc[timeindex-pad_len:timeindex + pad_len, f'{dir}_Node_Location_nb'] = np.asarray(pad_extended, dtype=np.float32)*scaling_factor
                        #print(sub_df.loc[timeindex-pad_len:timeindex + pad_len, 'EW_Node_Location_nb'])

        # encode labels
        if verbose>1:
            print("Fitting Labelencoders now.")
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

        # Drop all of the columns we dont need to preserve memory
        self._input_features = list(dict.fromkeys(self._input_features)) # remove duplicates
        if sort_inputs:
            # sort to improve compatibility between models trained from different sources (namely sweeper and notebook)
            self._input_features.sort()
            self._overview_features_mean.sort()
            self._overview_features_std.sort()
        input_cols = list(dict.fromkeys(self._input_features + self._overview_features_mean + self._overview_features_std)) # remove duplicates
        columns_to_keep = ['ObjectID', 'TimeIndex'] + input_cols + (['EW_Node', 'EW_Type', 'EW', 'EW_Node_Location_nb', 'EW_Node_Location',
                                                   'NS_Node', 'NS_Type', 'NS', 'NS_Node_Location_nb', 'NS_Node_Location'] if with_labels else [])
        columns_to_keep = list(dict.fromkeys(columns_to_keep)) # remove duplicates
        columns_to_remove = [item for item in split_df[self._train_keys[0]].columns if item not in columns_to_keep]
        if verbose>1:
                print(f"Dropping {len(columns_to_remove)} unused columns now.")
        self._preprocessed_dataframes = {key : value.drop(columns=columns_to_remove, inplace=False)  for key, value in split_df.items()}
        
        # change dtypes in the dataframes
       
        for key, value in self._preprocessed_dataframes.items():
            value[input_cols] = value[input_cols].astype(input_dtype)

        split_df = None #remove reference to possibly save memory

        self._input_feature_indices = {name:i for i, name in enumerate(self._input_features)}
        if verbose > 0:
            print(f"Final {len(self._input_features) + len(self._overview_features_mean) + len(self._overview_features_std)} input features: {self._input_features} + overview of {self._overview_features_mean} (mean) and {self._overview_features_std} (std)")
  
        if verbose > 0:
            print(f"=========================Finished Generator=======================")

    def create_ds_from_dataframes(self,
                                  split_df,
                                  keys,
                                  input_features,
                                  overview_features_mean,
                                  overview_features_std,
                                  label_features,
                                  only_nodes,
                                  only_ew_sk,
                                  with_identifier,
                                  input_history_steps,
                                  input_future_steps,
                                  stride,
                                  keep_label_stride,
                                  input_stride,
                                  padding,
                                  verbose=1):
        
        if stride > 1 and keep_label_stride > 1:
            print("Warning: stride > 1 and keep_label_stride > 1 may lead to unexpected or erronous behavior!")

        window_size = input_history_steps + input_future_steps
        obj_lens = {key:len(split_df[key]) - (1 if padding != 'none' else (window_size-1)) for key in keys} # the last row (ES) is removed

        # "Reserve" np arrays (this is efficient but does not actually block the memory)
        n_rows = np.sum([ln for ln in obj_lens.values()])
        n_rows = n_rows//(stride+keep_label_stride-1)+30000 if not keep_label_stride else 15000 # consider stride with some buffer just to be sure
        inputs = np.zeros(shape=(n_rows, int(np.ceil(window_size/input_stride)), len(input_features) + len(overview_features_mean) + len(overview_features_std)), dtype=self._input_dtype) # dimensions are [index, time, features]
        labels = np.zeros(shape=(n_rows, len(label_features) if label_features else 1), dtype=np.int32 if not (('EW_Node_Location_nb' in label_features) or ('NS_Node_Location_nb' in label_features)) else np.float32)
        element_identifiers = np.zeros(shape=(n_rows, 2))

        if only_ew_sk:
            ew_sk_markers = np.zeros(shape=(n_rows), dtype=np.int32)
        
        current_row = 0
        keys.sort()
        for key in tqdm(keys, disable=verbose==0):
            extended_df = split_df[key]

            # First, add padding
            if padding != 'none':
                extended_df = split_df[key].copy()
                # to make sure that we have as many inputs as we actually have labels, we need to add rows to the beginning of the df
                # this is to ensure that we will later be "able" to predict the first entry in the labels (otherwise, we would need to skip n input_history_steps)
                nan_df_history = pd.DataFrame(np.nan, index=pd.RangeIndex(input_history_steps-1), columns=split_df[key].columns, dtype=np.float32)
                nan_df_future = pd.DataFrame(np.nan, index=pd.RangeIndex(input_future_steps-1), columns=split_df[key].columns, dtype=np.float32)
                extended_df = pd.concat([nan_df_history, split_df[key], nan_df_future]).reset_index(drop=True)
                if padding=='same':
                    extended_df.bfill(inplace=True) # replace NaN values in the beginning with first actual value
                    extended_df.ffill(inplace=True) # replace NaN values at the end with last actual value
                elif padding=='zero':
                    extended_df.fillna(0.0, inplace=True)
                else:
                    print(f"Warning: unknown padding method \"{padding}\"! Using zero-padding instead")
                    extended_df.fillna(0.0, inplace=True)

            # Get the labels. This is necessary here because they may be needed for the keep_label_stride
            obj_labels = None
            obj_locations = None
            if label_features:
                new_dt = np.int32
                if (('EW_Node_Location_nb' in label_features) or ('NS_Node_Location_nb' in label_features)):
                    new_dt = self._input_dtype
                elif (('EW_Node_Location' in label_features) or ('NS_Node_Location' in label_features)):
                    new_dt = np.bool_
                obj_labels = extended_df[label_features][input_history_steps-1:-input_future_steps].to_numpy(dtype=new_dt)
                dirs_locs = (['EW_Node_Location'] if any('EW' in ft for ft in label_features) else []) + (['NS_Node_Location'] if any('NS' in ft for ft in label_features) else [])
                obj_locations = extended_df[dirs_locs][input_history_steps-1:-input_future_steps].to_numpy(dtype=np.float32)
            
            # determine which indices to keep based on stride
            obj_indices_to_keep = np.zeros((obj_lens[key]), dtype=np.int32)
            if stride > 1:
                obj_indices_to_keep[::stride] = 1
            elif keep_label_stride > 1:
                obj_indices_to_keep[::keep_label_stride] = 1
                obj_indices_to_keep[np.argwhere(obj_locations[:,:] > 0.0)[:,0]] = 1
            elif only_nodes:
                obj_indices_to_keep[np.argwhere(obj_locations[:,:] > 0.0)[:,0]] = 1
            else: 
                obj_indices_to_keep[:] = 1
            obj_indices_to_keep = np.argwhere(obj_indices_to_keep==1)[:,0]

            # the final object length
            strided_obj_len = len(obj_indices_to_keep)

            # Add overview features based on the original, non-strided and non-padded df
            if overview_features_mean or overview_features_std:
                # Add overview features which are identical everywhere inside on object and capture a view of the entire feature development
                # here, intentionally do not use the extended df (we dont want padding in the overview)
                n_segments = int(np.ceil(window_size/input_stride))
                segment_length=len(split_df[key])//int(np.ceil(window_size/input_stride))
                object_border = (len(split_df[key])%n_segments)//2 # we have to cut off a bit on the left and right
                
                #mean
                segmented_features_mean=np.split(split_df[key][overview_features_mean].to_numpy()[object_border:object_border+n_segments*segment_length,:], n_segments)
                mean_vals = np.mean(segmented_features_mean, axis=1)
                inputs[current_row:current_row+strided_obj_len,:,len(input_features):len(input_features)+len(overview_features_mean)] = mean_vals
                #std
                segmented_features_std=np.split(split_df[key][overview_features_std].to_numpy()[object_border:object_border+n_segments*segment_length,:], n_segments)
                std_vals = np.std(segmented_features_std, axis=1)
                inputs[current_row:current_row+strided_obj_len,:,len(input_features)+len(overview_features_mean):] = std_vals
            
            # determine the inputs, labels and identifiers of the indices we want to keep
            inputs[current_row:current_row+strided_obj_len,:,:len(input_features)] = np.lib.stride_tricks.sliding_window_view(extended_df[input_features].to_numpy(dtype=self._input_dtype), window_size, axis=0).transpose(0,2,1)[obj_indices_to_keep,::input_stride,:,]
            labels[current_row:current_row+strided_obj_len] = obj_labels[obj_indices_to_keep] if label_features else 0.0
            element_identifiers[current_row:current_row+strided_obj_len] = extended_df[['ObjectID', 'TimeIndex']][input_history_steps-1:-input_future_steps].to_numpy(dtype=np.int32)[obj_indices_to_keep,:]

            if only_ew_sk:
                ew_sk_markers[current_row:current_row+strided_obj_len] = extended_df['EW_Type'][input_history_steps-1:-input_future_steps].to_numpy(dtype=np.int32)[obj_indices_to_keep]

            # advance to next object
            current_row+=strided_obj_len

            del extended_df # try to preserve some memory

        # limit the arrays to the actual "filled" portion
        inputs = inputs[:current_row]
        labels = labels[:current_row]
        element_identifiers = element_identifiers[:current_row]

        if only_ew_sk:
            # keep all fields where EW is stationkeeping
            ew_nk_label = self._type_label_encoder.transform(['NK'])[0]
            ew_sk_fields = np.argwhere(ew_sk_markers != ew_nk_label)[:,0]
            inputs = inputs[ew_sk_fields]
            labels = labels[ew_sk_fields]
            element_identifiers = element_identifiers[ew_sk_fields]
            
        # # if wanted, only preserve items with nodes
        # if only_nodes:
        #     # identify which direction we want
        #     ew_nodes = np.argwhere(node_location_markers[:,0] == 1)[:,0]
        #     ns_nodes = np.argwhere(node_location_markers[:,1] == 1)[:,0]
        #     with_ew = any('EW' in ft for ft in label_features)
        #     with_ns = any('NS' in ft for ft in label_features)
        #     nodes = [] + ([ew_nodes] if with_ew else []) + ([ns_nodes] if with_ns else [])
        #     nodes = np.sort(np.unique(np.concatenate(nodes)))
        #     inputs = inputs[nodes]
        #     labels = labels[nodes]
        #     element_identifiers = element_identifiers[nodes]
            
        # finally, create the dataset
        with tf.device("CPU"):
            dataset = tf.data.Dataset.from_tensor_slices((inputs, {feature:labels[:,ft_idx] for ft_idx, feature in enumerate(label_features)}, element_identifiers) if (label_features and with_identifier) else
                                                (inputs, element_identifiers) if ((not label_features) and with_identifier) else
                                                (inputs, {feature:labels[:,ft_idx] for ft_idx, feature in enumerate(label_features)}))

        # Do some garbage collection - StackOverflow is not sure if this will help or not
        del labels
        del inputs
        del element_identifiers
        gc.collect()

        return dataset
        
    def create_stateful_ds_from_dataframes(self, split_df, keys, input_features, label_features,
                                  with_identifier, input_history_steps, input_future_steps, stride,
                                  input_stride,
                                  padding,
                                  min_exp_length):
        
        # filter dfs by min_exp_length
        keys = [key for key in keys if len(split_df[key]) >= min_exp_length]

        window_size = input_history_steps + input_future_steps
        strided_obj_len = int(np.ceil(min_exp_length/stride)) - (1 if padding != 'none' else (window_size-1))
        n_rows = len(keys)*strided_obj_len # consider stride
        inputs = np.zeros(shape=(n_rows, int(np.ceil(window_size/input_stride)), len(input_features))) # dimensions are [index, time, features]
        labels = np.zeros(shape=(n_rows, len(label_features) if label_features else 1), dtype=np.int32 if not (('EW_Node_Location_nb' in label_features) or ('NS_Node_Location_nb' in label_features)) else np.float32)
        element_identifiers = np.zeros(shape=(n_rows, 2))
        keys.sort()
        for key_idx, key in enumerate(keys):
            extended_df = split_df[key].copy()
            extended_df = extended_df.iloc[:min_exp_length] # cut down to desired size

            # cut off the part that is too long
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
            
            inputs[key_idx::len(keys)] = np.lib.stride_tricks.sliding_window_view(extended_df[input_features].to_numpy(dtype=self._input_dtype), window_size, axis=0).transpose(0,2,1)[::stride,::input_stride,:,]
            if label_features:
                new_dt = np.int32
                if (('EW_Node_Location_nb' in label_features) or ('NS_Node_Location_nb' in label_features)):
                    new_dt = np.float32
                labels[key_idx::len(keys)] = extended_df[label_features][input_history_steps-1:-input_future_steps].to_numpy(dtype=new_dt)[::stride]
            element_identifiers[key_idx::len(keys)] = extended_df[['ObjectID', 'TimeIndex']][input_history_steps-1:-input_future_steps].to_numpy(dtype=np.int32)[::stride,:]

            del extended_df # try to preserve some memory
        gc.collect() # try to preserve some memory

        datasets = [tf.data.Dataset.from_tensor_slices((inputs))]
        if label_features:
            datasets += [tf.data.Dataset.from_tensor_slices(({feature:labels[:,ft_idx] for ft_idx, feature in enumerate(label_features)}))]
        if with_identifier:
            datasets += [tf.data.Dataset.from_tensor_slices((element_identifiers))]
        
        # Do some garbage collection - StackOverflow is not sure if this will help or not
        del labels
        del inputs
        del element_identifiers
        gc.collect()

        if len(datasets)>1:
            return tf.data.Dataset.zip(tuple(datasets)).batch(len(keys), drop_remainder=False)
        else:
            return datasets[0].batch(len(keys))
        
    def get_stateful_datasets(self, label_features=['EW', 'EW_Node', 'EW_Type', 'NS', 'NS_Node', 'NS_Type'], 
                     with_identifier=False,
                     train_keys=None, val_keys=None, stride=1, min_exp_length=2170):
        
        
        # create datasets
        train_keys = self._train_keys if train_keys is None else train_keys
        val_keys = self._val_keys if val_keys is None else val_keys
        train_ds = self.create_stateful_ds_from_dataframes(self._preprocessed_dataframes,
                                                keys=train_keys,
                                                input_features=self._input_features,
                                                label_features=label_features,
                                                with_identifier=with_identifier,
                                                input_history_steps=self._input_history_steps,
                                                input_future_steps=self._input_future_steps,
                                                stride=stride,
                                                input_stride=self._input_stride,
                                                padding=self._padding,
                                                min_exp_length=min_exp_length)
        datasets = [train_ds]
        if val_keys:
            val_ds = self.create_stateful_ds_from_dataframes(self._preprocessed_dataframes,
                                                    keys=val_keys,
                                                    input_features=self._input_features,
                                                    label_features=label_features,
                                                    with_identifier=with_identifier,
                                                    input_history_steps=self._input_history_steps,
                                                    input_future_steps=self._input_future_steps,
                                                    stride=stride,
                                                    input_stride=self._input_stride,
                                                    padding=self._padding,
                                                    min_exp_length=min_exp_length)
            datasets.append(val_ds)
            
        return datasets if len(datasets)>1 else datasets[0]

    def get_datasets(self, batch_size=None, label_features=['EW', 'EW_Node', 'EW_Type', 'NS', 'NS_Node', 'NS_Type'],
                     with_identifier=False, only_nodes=False, only_ew_sk=False, shuffle=True,
                     train_keys=None, val_keys=None, stride=1, keep_label_stride=1, verbose=0):
        
        # create datasets
        train_keys = self._train_keys if train_keys is None else train_keys
        val_keys = self._val_keys if val_keys is None else val_keys
        train_ds = self.create_ds_from_dataframes(self._preprocessed_dataframes,
                                                keys=train_keys,
                                                input_features=self._input_features,
                                                overview_features_mean=self._overview_features_mean,
                                                overview_features_std=self._overview_features_std,
                                                label_features=label_features,
                                                only_nodes=only_nodes,
                                                only_ew_sk=only_ew_sk,
                                                with_identifier=with_identifier,
                                                input_history_steps=self._input_history_steps,
                                                input_future_steps=self._input_future_steps,
                                                stride=stride,
                                                keep_label_stride=keep_label_stride,
                                                input_stride=self._input_stride,
                                                padding=self._padding,
                                                verbose=verbose)
        datasets = [train_ds]
        gc.collect()
        if val_keys:
            val_ds = self.create_ds_from_dataframes(self._preprocessed_dataframes,
                                                    keys=val_keys,
                                                    input_features=self._input_features,
                                                    overview_features_mean=self._overview_features_mean,
                                                    overview_features_std=self._overview_features_std,
                                                    label_features=label_features,
                                                    only_nodes=only_nodes,
                                                    only_ew_sk=only_ew_sk,
                                                    with_identifier=with_identifier,
                                                    input_history_steps=self._input_history_steps,
                                                    input_future_steps=self._input_future_steps,
                                                    stride=stride,
                                                    keep_label_stride=keep_label_stride,
                                                    input_stride=self._input_stride,
                                                    padding=self._padding,
                                                    verbose=verbose)
            datasets.append(val_ds)
            
        if shuffle:
            print("Train-DS Cardinality:", datasets[0].cardinality())
            print("Val-DS Cardinality:", datasets[1].cardinality())
            datasets = [ds.shuffle(ds.cardinality(), seed=self._seed) for ds in datasets]
        if batch_size is not None:
            datasets = [ds.batch(batch_size) for ds in datasets]
        return datasets if len(datasets)>1 else datasets[0]
    
    def plot_dataset_items(self, ds_with_identifier):
        # TODO: plot random dataset items for debugging
        print("TBD")
        inputs = np.concatenate([element for element in ds_with_identifier.map(lambda x,y,z:x).as_numpy_iterator()])
        identifiers = np.concatenate([element for element in ds_with_identifier.map(lambda x,y,z:z).as_numpy_iterator()])

        # Take some element
        print(inputs.shape, identifiers.shape)
        
        inputs=None
        identifiers=None
    
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