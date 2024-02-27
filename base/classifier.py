import numpy as np
import pandas as pd
import gc

# Helper lambdas
def get_x_from_xy(x,y):
    return x
def get_y_from_xy(x,y):
    return y
def get_x_from_xyz(x,y,z):
    return x
def get_y_from_xyz(x,y,z):
    return y
def get_z_from_xyz(x,y,z):
    return z

def plot_confusion_matrix(ds_gen, ds_with_labels, model, output_names=['EW_Type', 'NS_Type']):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import seaborn as sns

    preds = model.predict(ds_with_labels, verbose=0)


    fig, axes = plt.subplots(nrows=1, ncols=len(output_names), figsize=(6*len(output_names),4))
    plt.tight_layout()
    for output_idx, output_name in enumerate(output_names):
        labels = np.concatenate([element for element in ds_with_labels.map(lambda x,y,z: y[output_name]).as_numpy_iterator()])

        ticklabels = [0,1,2,3]
        if output_name == 'EW_Type' or output_name == 'NS_Type':
            ticklabels = ds_gen.type_label_encoder.inverse_transform(ticklabels)
        
        confusion_mtx = tf.math.confusion_matrix(labels, np.argmax(preds[output_idx] if len(output_names)>1 else preds, axis=1))
        sns.heatmap(confusion_mtx,
                    xticklabels=ticklabels,
                    yticklabels=ticklabels,
                    annot=True, fmt='g', ax=axes[output_idx] if len(output_names)>1 else axes)
        
        if len(output_names)>1:
            axes[output_idx].set_title(output_name)
            axes[output_idx].set_xlabel('Prediction')
            axes[output_idx].set_ylabel('Label')
        else:
            axes.set_title(output_name)
            axes.set_xlabel('Prediction')
            axes.set_ylabel('Label')
    fig.show()

def create_prediction_df(ds_gen, model, train=False, test=False, model_outputs=['EW_Type', 'NS_Type'],
                         object_limit=None, only_nodes=False,
                         confusion_matrix=False,
                         prediction_batches=1,
                         verbose=1):
    
    all_identifiers = []
    all_predictions = []
    
    all_train_keys = ds_gen.train_keys[:(len(ds_gen.train_keys) if object_limit is None else object_limit)]
    train_batch_size = int(np.ceil(len(all_train_keys)/prediction_batches))
    all_val_keys = ds_gen.val_keys[:(len(ds_gen.val_keys) if object_limit is None else object_limit)]
    val_batch_size = int(np.ceil(len(all_val_keys)/prediction_batches))
    for batch_idx in range(prediction_batches):
        train_keys = all_train_keys[batch_idx*train_batch_size:batch_idx*train_batch_size+train_batch_size]
        val_keys = all_val_keys[batch_idx*val_batch_size:batch_idx*val_batch_size+val_batch_size]
        datasets = ds_gen.get_datasets(batch_size=512,
                                        label_features=[] if test else model_outputs,
                                        shuffle=False, # if we dont use the majority method, its enough to just evaluate on nodes
                                        with_identifier=True,
                                        train_keys=train_keys[:(len(train_keys) if (train or test) else 1)],
                                        val_keys=val_keys[:(len(val_keys) if not (train or test) else 1)],
                                        only_nodes=only_nodes,
                                        stride=1)
        ds = (datasets[0] if train else datasets[1]) if not test else datasets

        identifiers = np.concatenate([element for element in ds.map(get_y_from_xy).as_numpy_iterator()]) if test else np.concatenate([element for element in ds.map(get_z_from_xyz).as_numpy_iterator()])

        # get predictions
        preds = np.asarray(model.predict(ds, verbose=verbose)) # TODO: may fail if single-class pred

        all_identifiers.append(identifiers)
        all_predictions.append(preds)

        del ds
        del datasets
    gc.collect()

    all_identifiers = np.concatenate(all_identifiers)
    all_predictions = np.concatenate(all_predictions, axis=0 if len(model_outputs)==1 else 1)

    df = pd.DataFrame(np.concatenate([all_identifiers.reshape(-1,2)], axis=1), columns=['ObjectID', 'TimeIndex'], dtype=np.int32)

    # Ordering of model_outputs MUST MATCH with actual outputs!
    for output_idx, output_name in enumerate(model_outputs):
        preds_argmax = np.argmax(all_predictions[output_idx] if len(model_outputs) > 1 else all_predictions, axis=1)
        df[f'{output_name}_Pred'] = preds_argmax
        if output_name == 'EW_Node' or output_name == 'NS_Node':
            df[f'{output_name}'] = ds_gen.node_label_encoder.inverse_transform(df[f'{output_name}_Pred'])
        if output_name == 'EW_Type' or output_name == 'NS_Type':
            df[f'{output_name}'] = ds_gen.type_label_encoder.inverse_transform(df[f'{output_name}_Pred'])
        if output_name == 'EW' or output_name == 'NS':
            df[f'{output_name}'] = ds_gen.combined_label_encoder.inverse_transform(df[f'{output_name}_Pred'])
            df[[f'{output_name}_Node', f'{output_name}_Type']] = df[f'{output_name}'].str.split('-', expand=True)

    df = df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

    if confusion_matrix:
        # TODO: make this work for batched ds!
        plot_confusion_matrix(ds_gen, ds, model, output_names=model_outputs)

    return df

def fill_unknwon_nodes_based_on_type(df, dirs=['EW', 'NS']):
    """Take an almost finished submission df, and fill 'UNKNOWN' nodes according to the types
    Assumes df to have columns [OjectID, TimeIndex, Direction, Node, Type]
    REQUIRES df to include SS nodes
    """
    dfs = []
    for dir in dirs:
        # merge locs into preds
        sub_df = df.loc[df['Direction'] == dir].copy()
        sub_df = sub_df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True) # important

        # nodes always depend on the previous and current type
        object_ids = sub_df['ObjectID'].unique()
        for object_id in object_ids:
            obj_df=sub_df.loc[sub_df['ObjectID'] == object_id]
            for index, row in obj_df.iterrows():
                if sub_df.loc[index, 'Node'] == 'UNKNOWN':
                    if row['TimeIndex']==0:
                        sub_df.loc[index, 'Node'] = 'SS'
                    else:
                        prev_type = sub_df.loc[index-1, 'Type']
                        next_type = sub_df.loc[index, 'Type']
                        node = ''
                        if prev_type == 'NK' and next_type =='NK': node = 'AD'
                        elif prev_type in ['CK', 'EK', 'HK'] and next_type == 'NK': node = 'ID'
                        elif prev_type == 'NK' and next_type in ['CK', 'EK', 'HK']: node = 'IK'
                        else: node = 'IK' # something is wrong, so choose the most common node
                        sub_df.loc[index, 'Node'] = node

        dfs.append(sub_df)

    # concatenate both directions
    df = pd.concat(dfs)
    df = df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
    return df

def fill_unknown_types_based_on_preds(preds_df, location_df, dirs=['EW', 'NS']):
    """Take the locations from location_df and apply immediate prediction, but only if current type is 'UNKNOWN'
    Assumes preds_df to have columns [OjectID, TimeIndex, EW_Type, NS_Type]
    Assumes location_df to have columns [OjectID, TimeIndex, Direction, Node, Type]
    """

    dfs = []
    for dir in dirs:
        # merge locs into preds
        sub_df = location_df.copy()
        sub_df = sub_df.loc[sub_df['Direction'] == dir]
        sub_df = preds_df[['ObjectID', 'TimeIndex', f'{dir}_Type']].merge(sub_df, how='inner', on=['ObjectID', 'TimeIndex'])
        sub_df = sub_df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True) # important
        sub_df.loc[sub_df['Type'] == 'UNKNOWN', 'Type'] = sub_df.loc[sub_df['Type'] == 'UNKNOWN', f'{dir}_Type']

        dfs.append(sub_df)

    # concatenate both directions
    df = pd.concat(dfs)
    df = df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
    return df

def apply_one_shot_method(preds_df, location_df, dirs=['EW', 'NS']):
    """Take the locations from location_df and apply immediate prediction
    Assumes preds_df to have columns [OjectID, TimeIndex, EW_Type, NS_Type]
    Assumes location_df to have columns [OjectID, TimeIndex, Direction]
    """

    # make sure input locations are somewhat cleaned
    if 'Node' in location_df.columns:
        location_df = location_df.loc[location_df['Node'] != 'ES']
    location_df = location_df[['ObjectID', 'TimeIndex', 'Direction']]

    dfs = []
    for dir in dirs:
        # merge locs into preds
        df = location_df.copy()
        df = df.loc[df['Direction'] == dir]
        df = preds_df[['ObjectID', 'TimeIndex', f'{dir}_Type']].merge(df, how='inner', on=['ObjectID', 'TimeIndex'])
        df = df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True) # important

        df['Type'] = df[f'{dir}_Type']
        df['Node'] = 'UNKNOWN'

        # nodes always depend on the previous and currernt type
        object_ids = df['ObjectID'].unique()
        for object_id in object_ids:
            obj_df=df.loc[df['ObjectID'] == object_id]
            for index, row in obj_df.iterrows():
                if row['TimeIndex']==0:
                    df.loc[index, 'Node'] = 'SS'
                else:
                    prev_type = df.loc[index-1, 'Type']
                    next_type = df.loc[index, 'Type']
                    node = ''
                    if prev_type == 'NK' and next_type =='NK': node = 'AD'
                    elif prev_type in ['CK', 'EK', 'HK'] and next_type == 'NK': node = 'ID'
                    elif prev_type == 'NK' and next_type in ['CK', 'EK', 'HK']: node = 'IK'
                    else: node = 'IK' # something is wrong, so choose the most common node
                    df.loc[index, 'Node'] = node

        dfs.append(df)

    # concatenate both directions
    df = pd.concat(dfs)
    df = df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
    return df

def apply_majority_method(preds_df, location_df):
    """Take the locations from location_df, segment the preds_df, assign majority value for type and node
    Assumes preds_df to have columns [OjectID, TimeIndex, EW_Type, NS_Type]
    Assumes location_df to have columns [OjectID, TimeIndex, Direction]
    """

    # make sure input locations are somewhat cleaned
    if 'Node' in location_df.columns:
        location_df = location_df.loc[location_df['Node'] != 'ES']
    location_df = location_df[['ObjectID', 'TimeIndex', 'Direction']]

    dfs = []

    for dir in ['EW', 'NS']:

        # merge locs into preds
        df = location_df.copy()
        df[f'{dir}_Loc'] = False
        df.loc[df['Direction'] == dir, [f'{dir}_Loc']] = True
        df = df[['ObjectID', 'TimeIndex', f'{dir}_Loc']]
        df = preds_df.merge(df[['ObjectID', 'TimeIndex', f'{dir}_Loc']], how='left', on=['ObjectID', 'TimeIndex'])
        object_ids = df['ObjectID'].unique()
    
        # re-add initial rows in case they got removed
        if len(df.loc[df['TimeIndex'] == 0]) == 0:
            print("adding initial nodes manually")
            initial_node_df = pd.DataFrame(columns=df.columns)
            initial_node_df['ObjectID'] = object_ids
            initial_node_df['TimeIndex'] = 0
            initial_node_df[f'{dir}_Loc'] = True
            initial_node_df[f'{dir}_Type'] = 'NA'
            df = pd.concat([df, initial_node_df])

        # sort the dataframe - this is important for the next step
        df = df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

        for object_id in object_ids:
            vals = df[f'{dir}_Type'].to_numpy()

            obj_indices = df.index[(df['ObjectID'] == object_id)].to_list()
            loc_indices = df.index[(df['ObjectID'] == object_id) & (df[f'{dir}_Loc'] == True)].to_list() # df indices where loc=1
                
            # determine segments, determine majority value
            loc_indices = loc_indices + [np.max(obj_indices)]
            loc_indices = np.unique(loc_indices, axis=0) # it may happen that index 0 exists twice; in that case remove duplicates
            loc_indices.sort()
            locs = df.iloc[loc_indices]['TimeIndex'].to_list() # actual timeindex
            
            segments = np.split(vals, loc_indices)[1:-1] # remove the segments of length 1 at the start and end
            segment_types = []
            for segment in segments:
                types, type_idx = np.unique(segment, return_counts=True)
                segment_types.append(types[np.argmax(type_idx)])
                
            most_common_values = segment_types

            # assign majority type of segment _after_ a loc to the loc, and determine nodes accordingly
            for idx, major_val in enumerate(most_common_values):
                df.at[loc_indices[idx], f'{dir}_Type'] = major_val

                if idx==0:
                    df.at[loc_indices[idx], f'{dir}_Node'] = 'SS'
                else:
                    prev_type = most_common_values[idx-1]
                    next_type = most_common_values[idx]
                    node = ''
                    # TODO: the ordering here could have an effect on the number of TP/FP
                    if prev_type == 'NK' and next_type =='NK': node = 'AD'
                    elif prev_type == 'NK' and any(next_type == nd for nd in ['CK', 'EK', 'HK']): node = 'IK'
                    else: node = 'ID'
                    df.at[loc_indices[idx], f'{dir}_Node'] = node

        # create submission-style df
        df = df.loc[df[f'{dir}_Loc'] == True]
        df.loc[df[f'{dir}_Loc'] == True, 'Direction'] = dir
        df['Node'] = df[f'{dir}_Node']
        df['Type'] = df[f'{dir}_Type']

        dfs.append(df)

    # concatenate both directions
    df = pd.concat(dfs)
    df = df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
    return df


def apply_majority_method_legacy(preds_df, location_df):
    """Take the locations from location_df, segment the preds_df, assign majority value for type and node
    Assumes preds_df to have columns [OjectID, TimeIndex, EW_Type, NS_Type]
    """

    # merge locs into preds
    location_df = location_df.copy()
    location_df['Loc_EW'] = 0
    location_df['Loc_NS'] = 0
    location_df.loc[location_df['Direction'] == 'EW', ['Loc_EW']] = 1
    location_df.loc[location_df['Direction'] == 'NS', ['Loc_NS']] = 1
    location_df = location_df[['ObjectID', 'TimeIndex', 'Loc_EW', 'Loc_NS']]
    #location_df = location_df.groupby(['ObjectID', 'TimeIndex']).agg({'Loc_EW': 'max', 'Loc_NS': 'max'}).reset_index()

    # merge rows
    df = preds_df.merge(location_df[['ObjectID', 'TimeIndex', 'Loc_EW', 'Loc_NS']], how='left', on=['ObjectID', 'TimeIndex'])


    # sort the dataframe - this is important for the next step
    df = df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

    object_ids = df['ObjectID'].unique()

    for object_id in object_ids:

        for dir in ['EW', 'NS']:
            vals = df[f'{dir}_Type'].to_numpy() # TODO: this contains some data leakage, as points where the other direction has nodes will be counted double

            obj_indices = df.index[(df['ObjectID'] == object_id)].to_list()
            loc_indices = df.index[(df['ObjectID'] == object_id) & (df[f'Loc_{dir}'] == 1)].to_list() # df indices where loc=1

            # determine segments, determine majority value
            loc_indices = [np.min(obj_indices)] + loc_indices + [np.max(obj_indices)]
            loc_indices = np.unique(loc_indices, axis=0) # it may happen that index 0 exists twice; in that case remove duplicates
            loc_indices.sort()
            #locs = df.iloc[loc_indices]['TimeIndex'].to_list() # actual timeindex


            segments = np.split(vals, loc_indices)[1:-1] # remove the segments of length 1 at the start and end
            segment_types = []
            for segment in segments:
                types, type_idx = np.unique(segment, return_counts=True)
                segment_types.append(types[np.argmax(type_idx)])

            most_common_values = segment_types

            # assign majority type of segment _after_ a loc to the loc, and determine nodes accordingly
            for idx, major_val in enumerate(most_common_values):
                df.at[loc_indices[idx], f'{dir}_Type'] = major_val

                if idx==0:
                    df.at[loc_indices[idx], f'{dir}_Node'] = 'SS'
                else:
                    prev_type = most_common_values[idx-1]
                    next_type = most_common_values[idx]
                    node = ''
                    # TODO: the ordering here could have an effect on the number of TP/FP
                    if prev_type == 'NK' and next_type =='NK': node = 'AD'
                    elif prev_type == 'NK' and any(next_type == nd for nd in ['CK', 'EK', 'HK']): node = 'IK'
                    else: node = 'ID'
                    df.at[loc_indices[idx], f'{dir}_Node'] = node

    df = df.loc[(df['Loc_EW'] == 1) | (df['Loc_NS'] == 1)]
    print(df.head(5))

    # turn this into a submission-style df
    for dir in ['EW', 'NS']:
        df.loc[df[f'Loc_{dir}'] == 1, 'Direction'] = dir
        df.loc[(df[f'Loc_{dir}'] == 1) & (df['Direction'] == dir), 'Node'] = df.loc[(df[f'Loc_{dir}'] == 1) & (df['Direction'] == dir), f'{dir}_Node']
        df.loc[(df[f'Loc_{dir}'] == 1) & (df['Direction'] == dir), 'Type'] = df.loc[(df[f'Loc_{dir}'] == 1) & (df['Direction'] == dir), f'{dir}_Type']

    df = df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
    return df