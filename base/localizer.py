import numpy as np
import pandas as pd

def create_prediction_df(ds_gen, model, train=False, test=False, output_dirs=['EW', 'NS'], verbose=1):

    datasets = ds_gen.get_datasets(batch_size=512,
                                     label_features=[],
                                     shuffle=False, # if we dont use the majority method, its enough to just evaluate on nodes
                                     with_identifier=True,
                                     stride=1)
    ds = (datasets[0] if train else datasets[1]) if not test else datasets

    inputs = np.concatenate([element for element in ds.map(lambda x,y: x).as_numpy_iterator()])
    identifiers = np.concatenate([element for element in ds.map(lambda x,y: y).as_numpy_iterator()])

    df = pd.DataFrame(np.concatenate([identifiers.reshape(-1,2)], axis=1), columns=['ObjectID', 'TimeIndex'], dtype=np.int32)

    # get predictions
    preds = model.predict(inputs, verbose=verbose)

    # Ordering of model_outputs MUST MATCH with actual outputs!
    for dir_idx, dir in enumerate(output_dirs):
        df[f'{dir}_Loc'] = preds[dir_idx] if len(output_dirs)>1 else preds

    df = df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

    return df

def postprocess_predictions(preds_df, dirs=['EW', 'NS'], threshold=50.0, add_initial_node=False, clean_consecutives=True):
    """Expects input df with columns [ObjectID, TimeIndex, EW_Loc, NS_Loc]
    """

    df = preds_df.copy()
    object_ids = df['ObjectID'].unique()
    df['Any_Loc'] = False
    # apply threshold
    for dir in dirs:
        df[f'{dir}_Loc'] = df[f'{dir}_Loc'] >= threshold
        df['Any_Loc'] = df['Any_Loc'] | df[f'{dir}_Loc']

    # remove consecutive location predictions, and replace them only with their center
     # TODO: this fails when two consecutive objects have detections at exactly consecutive timeindices - a corner case I ignore for now ;)
    if clean_consecutives:
        df = df.loc[(df['Any_Loc'] == True)]
        df['consecutive'] = (df['TimeIndex'] - df['TimeIndex'].shift(1) != 1).cumsum()
        # Filter rows where any number of consecutive values follow each other
        df=df.groupby('consecutive').apply(lambda sub_df: sub_df.iloc[int(len(sub_df)/2), :]).reset_index(drop=True).drop(columns=['consecutive'])

    # add initial node
    if add_initial_node:
        initial_node_df = pd.DataFrame(columns=df.columns)
        initial_node_df['ObjectID'] = object_ids
        initial_node_df['TimeIndex'] = 0
        for dir in dirs:
            initial_node_df[f'{dir}_Loc'] = True
        df = pd.concat([df, initial_node_df])

    # bring it all into the submission format
    sub_dfs = []
    for dir in dirs:
        sub_df = df.loc[df[f'{dir}_Loc'] == True].copy()
        sub_df['Direction'] = dir
        sub_dfs.append(sub_df)
    df = pd.concat(sub_dfs)

    df = df[['ObjectID', 'TimeIndex', 'Direction']].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

    return df

def evaluate_localizer(subm_df, gt_path, object_ids, dirs=['EW', 'NS'], with_initial_node=False, return_scores=False, verbose=1):
    from base import evaluation
    # Load gt
    ground_truth_df = pd.read_csv(gt_path)

    # Filter objects
    print(225 in object_ids)
    ground_truth_df = ground_truth_df.loc[ground_truth_df['ObjectID'].isin(object_ids)]
    subm_df = subm_df.loc[subm_df['ObjectID'].isin(object_ids)]

    print(ground_truth_df.loc[ground_truth_df['ObjectID']==225])

    # Filter direction
    ground_truth_df = ground_truth_df.loc[(ground_truth_df['Direction'].isin(dirs + ['ES']))] # Keep es row, to maintain object for evaluation
    subm_df = subm_df.loc[(subm_df['Direction'].isin(dirs))]

    # Filter TimeIndices
    if not with_initial_node:
        ground_truth_df = ground_truth_df.loc[(ground_truth_df['TimeIndex'] != 0)]
        subm_df = subm_df.loc[(subm_df['TimeIndex'] != 0)]

    # Initiate evaluator
    evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=subm_df, ignore_classes=True)
    precision, recall, f2, rmse, total_tp, total_fp, total_fn = evaluator.score()

    if verbose>0:
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F2: {f2:.2f}')
        print(f'RMSE: {float(rmse):.4}')
        print(f'TP: {total_tp} FP: {total_fp} FN: {total_fn}')

    if return_scores:
        return {'Precision':precision, 'Recall':recall, 'F2':f2, 'RMSE':rmse, 'TP':total_tp, 'FP':total_fp, 'FN':total_fn}
    else:
        return evaluator, subm_df