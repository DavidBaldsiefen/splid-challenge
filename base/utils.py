import pandas as pd
from tqdm import tqdm
from fastcore.basics import Path

# Function to prepare the data in a tabular format
def tabularize_data(data_dir, feature_cols, ground_truth=None, lag_steps=1, fill_na=True):
    merged_data = pd.DataFrame()
    test_data = Path(data_dir).glob('*.csv')
    # Check if test_data is empty
    if not test_data:
        raise ValueError(f'No csv files found in {data_dir}')
    for data_file in test_data:
        data_df = pd.read_csv(data_file)
        data_df['ObjectID'] = int(data_file.stem)
        data_df['TimeIndex'] = range(len(data_df))
    
        lagged_features = []
        new_feature_cols = list(feature_cols)  # Create a copy of feature_cols
        # Create lagged features for each column in feature_cols
        for col in feature_cols:
            for i in range(1, lag_steps+1):
                lag_col_name = f'{col}_lag_{i}'
                data_df[lag_col_name] = data_df.groupby('ObjectID')[col].shift(i)
                new_feature_cols.append(lag_col_name)  # Add the lagged feature to new_feature_cols
        
        # Add the lagged features to the DataFrame all at once
        data_df = pd.concat([data_df] + lagged_features, axis=1)

        if ground_truth is None:
            merged_df = data_df
        else:
            ground_truth_object = ground_truth[ground_truth['ObjectID'] == data_df['ObjectID'][0]].copy()
            # Separate the 'EW' and 'NS' types in the ground truth
            ground_truth_EW = ground_truth_object[ground_truth_object['Direction'] == 'EW'].copy()
            ground_truth_NS = ground_truth_object[ground_truth_object['Direction'] == 'NS'].copy()
            
            # Create 'EW' and 'NS' labels and fill 'unknown' values
            ground_truth_EW['EW'] = ground_truth_EW['Node'] + '-' + ground_truth_EW['Type']
            ground_truth_NS['NS'] = ground_truth_NS['Node'] + '-' + ground_truth_NS['Type']
            ground_truth_EW.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)
            ground_truth_NS.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)

            # Merge the input data with the ground truth
            merged_df = pd.merge(data_df, 
                                ground_truth_EW.sort_values('TimeIndex'), 
                                on=['TimeIndex', 'ObjectID'],
                                how='left')
            merged_df = pd.merge_ordered(merged_df, 
                                        ground_truth_NS.sort_values('TimeIndex'), 
                                        on=['TimeIndex', 'ObjectID'],
                                        how='left')

            # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
            merged_df['EW'].ffill(inplace=True)
            merged_df['NS'].ffill(inplace=True)
            
        merged_data = pd.concat([merged_data, merged_df])

    # Fill missing values (for the lagged features)
    print("ok")
    print(merged_data.head(5))
    if fill_na:
        merged_data.bfill(inplace=True)
    
    return merged_data, new_feature_cols

def convert_classifier_output(classifier_output):
    # Split the 'Predicted_EW' and 'Predicted_NS' columns into 
    # 'Node' and 'Type' columns
    ew_df = classifier_output[['TimeIndex', 'ObjectID', 'Predicted_EW']].copy()
    ew_df[['Node', 'Type']] = ew_df['Predicted_EW'].str.split('-', expand=True)
    ew_df['Direction'] = 'EW'
    ew_df.drop(columns=['Predicted_EW'], inplace=True)

    ns_df = classifier_output[['TimeIndex', 'ObjectID', 'Predicted_NS']].copy()
    ns_df[['Node', 'Type']] = ns_df['Predicted_NS'].str.split('-', expand=True)
    ns_df['Direction'] = 'NS'
    ns_df.drop(columns=['Predicted_NS'], inplace=True)

    # Concatenate the processed EW and NS dataframes
    final_df = pd.concat([ew_df, ns_df], ignore_index=True)

    # Sort dataframe based on 'ObjectID', 'Direction' and 'TimeIndex'
    final_df.sort_values(['ObjectID', 'Direction', 'TimeIndex'], inplace=True)

    # Apply the function to each group of rows with the same 'ObjectID' and 'Direction'
    groups = final_df.groupby(['ObjectID', 'Direction'])
    keep = groups[['Node', 'Type']].apply(lambda group: group.shift() != group).any(axis=1)

    # Filter the DataFrame to keep only the rows we're interested in
    keep.index = final_df.index
    final_df = final_df[keep]

    # Reset the index and reorder the columns
    final_df = final_df.reset_index(drop=True)
    final_df = final_df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']]
    final_df = final_df.sort_values(['ObjectID', 'TimeIndex', 'Direction'])

    return final_df


def smooth_predictions(pred_df, past_steps=5, fut_steps=5, verbose=1):
    '''Run a sliding window over the Predicted_EW/Predicted_NS columns in the dataframe and make them smoother'''
    # It's probably possible to do this with nps windowing function somehow..
    pred_df = pred_df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
    pred_df['Predicted_EW_smoothed'] = pred_df['Predicted_EW']
    pred_df['Predicted_EW_raw'] = pred_df['Predicted_EW']
    pred_df['Predicted_NS_smoothed'] = pred_df['Predicted_NS']
    pred_df['Predicted_NS_raw'] = pred_df['Predicted_NS']
    # move a window over data, set new value of cell to the most common label in the window
    # could use df.rolling...
    object_ids = pred_df['ObjectID'].unique()
    obj_dfs = []
    for obj_id in tqdm(object_ids, desc="Smoothing", disable=(verbose==0)):
        obj_data = pred_df[pred_df['ObjectID'].eq(obj_id)].reset_index(drop=True)
        cur_row = past_steps
        while cur_row < len(obj_data)-fut_steps-1:
            EW_preds = obj_data.loc[cur_row - past_steps:cur_row+fut_steps+1, 'Predicted_EW_raw'].to_list()
            NS_preds = obj_data.loc[cur_row - past_steps:cur_row+fut_steps+1, 'Predicted_NS_raw'].to_list()
            obj_data.loc[cur_row, 'Predicted_EW_smoothed'] = max(set(EW_preds), key=EW_preds.count)
            obj_data.loc[cur_row, 'Predicted_NS_smoothed'] = max(set(NS_preds), key=NS_preds.count)
            # if obj_id == 1 and cur_row > 1200 and cur_row < 1220:
            #     print(cur_row, NS_preds, max(set(NS_preds), key=NS_preds.count), obj_data.loc[cur_row, 'Predicted_NS_smoothed'])
            cur_row += 1
        obj_dfs.append(obj_data)
    obj_dfs = pd.concat(obj_dfs)
    obj_dfs['Predicted_EW'] = obj_dfs['Predicted_EW_smoothed']
    obj_dfs['Predicted_NS'] = obj_dfs['Predicted_NS_smoothed']
    return obj_dfs

def smooth_locations(pred_df, past_steps=5, fut_steps=5, verbose=1):
    '''Run a sliding window over the Predicted_EW/Predicted_NS columns in the dataframe and make them smoother'''
    # It's probably possible to do this with nps windowing function somehow..
    pred_df = pred_df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
    pred_df['Location_Pred_smoothed'] = pred_df['Location_Pred']
    pred_df['Location_Pred_raw'] = pred_df['Location_Pred']
    # move a window over data, set new value of cell to the most common label in the window
    # could use df.rolling...
    object_ids = pred_df['ObjectID'].unique()
    obj_dfs = []
    for obj_id in tqdm(object_ids, desc="Smoothing", disable=(verbose==0)):
        obj_data = pred_df[pred_df['ObjectID'].eq(obj_id)].reset_index(drop=True)
        cur_row = past_steps
        while cur_row < len(obj_data)-fut_steps-1:
            EW_preds = obj_data.loc[cur_row - past_steps:cur_row+fut_steps+1, 'Predicted_EW_raw'].to_list()
            NS_preds = obj_data.loc[cur_row - past_steps:cur_row+fut_steps+1, 'Predicted_NS_raw'].to_list()
            obj_data.loc[cur_row, 'Predicted_EW_smoothed'] = max(set(EW_preds), key=EW_preds.count)
            obj_data.loc[cur_row, 'Predicted_NS_smoothed'] = max(set(NS_preds), key=NS_preds.count)
            # if obj_id == 1 and cur_row > 1200 and cur_row < 1220:
            #     print(cur_row, NS_preds, max(set(NS_preds), key=NS_preds.count), obj_data.loc[cur_row, 'Predicted_NS_smoothed'])
            cur_row += 1
        obj_dfs.append(obj_data)
    obj_dfs = pd.concat(obj_dfs)
    obj_dfs['Predicted_EW'] = obj_dfs['Predicted_EW_smoothed']
    obj_dfs['Predicted_NS'] = obj_dfs['Predicted_NS_smoothed']
    return obj_dfs