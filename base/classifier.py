import numpy as np
import pandas as pd

# TODO: implement framework for non-majority method, i.e. normal predictor

def create_prediction_df(ds_gen, model, train=False, test=False, model_outputs=['EW_Type', 'NS_Type'], verbose=1):
    datasets = ds_gen.get_datasets(batch_size=512,
                                     label_features=model_outputs if not test else [],
                                     shuffle=False, # if we dont use the majority method, its enough to just evaluate on nodes
                                     with_identifier=True,
                                     stride=1)
    ds = (datasets[0] if train else datasets[1]) if not test else datasets

    inputs = np.concatenate([element for element in ds.map(lambda x,y,z: x).as_numpy_iterator()]) if not test else np.concatenate([element for element in ds.map(lambda x,y: x).as_numpy_iterator()])
    #labels = np.concatenate([element['EW_Node_Location'] for element in ds.map(lambda x,y,z: y).as_numpy_iterator()])
    identifiers = np.concatenate([element for element in ds.map(lambda x,y,z: z).as_numpy_iterator()])  if not test else np.concatenate([element for element in ds.map(lambda x,y: y).as_numpy_iterator()])

    df = pd.DataFrame(np.concatenate([identifiers.reshape(-1,2)], axis=1), columns=['ObjectID', 'TimeIndex'], dtype=np.int32)

    # get predictions
    preds = model.predict(inputs, verbose=verbose)

    # Ordering of model_outputs MUST MATCH with actual outputs!
    for output_idx, output_name in enumerate(model_outputs):
        preds_argmax = np.argmax(preds[output_idx] if len(model_outputs) > 1 else preds, axis=1)
        df[f'{output_name}_Pred'] = preds_argmax
        if output_name == 'EW_Node' or output_name == 'NS_Node':
            df[f'{output_name}'] = ds_gen.node_label_encoder.inverse_transform(df[f'{output_name}_Pred'])
        if output_name == 'EW_Type' or output_name == 'NS_Type':
            df[f'{output_name}'] = ds_gen.type_label_encoder.inverse_transform(df[f'{output_name}_Pred'])
        if output_name == 'EW' or output_name == 'NS':
            df[f'{output_name}'] = ds_gen.combined_label_encoder.inverse_transform(df[f'{output_name}_Pred'])
            df[[f'{output_name}_Node', f'{output_name}_Type']] = df[f'{output_name}'].str.split('-', expand=True)

    df = df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

    return df


def apply_majority_method(preds_df, location_df):
    """Take the locations from location_df, segment the preds_df, assign majority value for type and node
    Assumes preds_df to have columns [OjectID, TimeIndex, EW_Type, NS_Type]
    Assumes location_df to have columns [OjectID, TimeIndex, Direction]
    """

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
