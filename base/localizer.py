import numpy as np
import pandas as pd
from tensorflow.data import Dataset
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

def create_prediction_df(ds_gen,
                         model,
                         stateful=False,
                         train=False,
                         test=False,
                         output_dirs=['EW', 'NS'],
                         only_ew_sk=False, 
                         object_limit=None,
                         prediction_batches=1,
                         ds_batch_size=256,
                         verbose=1):
    if test and object_limit is not None:
        print("Warning: Object limit applied on test set - intentional?")
    if stateful and prediction_batches != 1:
        print("Warning: Prediction batches needs to be '1' for stateful LSTMs!")
    
    all_identifiers = []
    all_predictions = []

    all_train_keys = ds_gen.train_keys[:(len(ds_gen.train_keys) if object_limit is None else object_limit)]
    train_batch_size = int(np.ceil(len(all_train_keys)/prediction_batches))
    all_val_keys = ds_gen.val_keys[:(len(ds_gen.val_keys) if object_limit is None else object_limit)]
    val_batch_size = int(np.ceil(len(all_val_keys)/prediction_batches))
    for batch_idx in range(prediction_batches):

        train_keys = all_train_keys[batch_idx*train_batch_size:batch_idx*train_batch_size+train_batch_size]
        val_keys = all_val_keys[batch_idx*val_batch_size:batch_idx*val_batch_size+val_batch_size]

        datasets = None
        if not stateful:
            datasets = ds_gen.get_datasets(batch_size=ds_batch_size,
                                            label_features=[],
                                            shuffle=False, # if we dont use the majority method, its enough to just evaluate on nodes
                                            with_identifier=True,
                                            only_ew_sk=only_ew_sk,
                                            train_keys=train_keys[:(len(train_keys) if (train or test) else 1)],
                                            val_keys=val_keys[:(len(val_keys) if not (train or test) else 1)],
                                            stride=1)
            ds = (datasets[0] if train else datasets[1]) if not test else datasets

            identifiers = np.concatenate([element for element in ds.map(get_y_from_xy).as_numpy_iterator()])

            # get predictions
            preds = np.asarray(model.predict(ds, verbose=verbose))

            all_identifiers.append(identifiers)
            all_predictions.append(preds)
            #del inputs
        else:
            datasets = ds_gen.get_stateful_datasets(label_features=[],
                                        with_identifier=True,
                                        train_keys=train_keys[:(len(train_keys) if (train or test) else 1)],
                                        val_keys=val_keys[:(len(val_keys) if not (train or test) else 1)],
                                        stride=1)
            ds = (datasets[0] if train else datasets[1]) if not test else datasets

            inputs = np.stack([element for element in ds.map(get_x_from_xy).as_numpy_iterator()])
            identifiers = np.stack([element for element in ds.map(get_y_from_xy).as_numpy_iterator()])

            for obj in range(inputs.shape[1]):
                sub_inputs = inputs[:,obj,:,:]
                sub_ds = Dataset.from_tensor_slices((sub_inputs))
                sub_ds = sub_ds.batch(1)
                sub_identifiers = identifiers[:,obj,:]


                # get predictions
                preds = model.predict(sub_ds, verbose=verbose)

                all_identifiers.append(sub_identifiers)
                all_predictions.append(preds)

        del ds
        del datasets
        gc.collect()

    # now create df by concatenating all the individual lists
    # TODO: make compatible with one and two outputs
    all_identifiers = np.concatenate(all_identifiers)
    all_predictions = np.concatenate(all_predictions, axis=0 if len(output_dirs)==1 else 1)#, axis=1)

    df = pd.DataFrame(np.concatenate([all_identifiers.reshape(-1,2)], axis=1), columns=['ObjectID', 'TimeIndex'], dtype=np.int32)

    # Ordering of model_outputs MUST MATCH with actual outputs!
    for dir_idx, dir in enumerate(output_dirs):
        df[f'{dir}_Loc'] = all_predictions[dir_idx] if len(output_dirs)>1 else all_predictions

    df = df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

    return df

def plot_prediction_curve(ds_gen, model, label_features=['EW_Node_Location_nb'], object_ids=[1], threshold=50.0, zoom=False):
    import matplotlib.pyplot as plt

    ds, v_ds = ds_gen.get_datasets(batch_size=256,
                                label_features=label_features,
                                shuffle=False,
                                with_identifier=True,
                                train_keys=object_ids,
                                val_keys=[ds_gen.val_keys[0]],
                                stride=1)

    
    labels = np.concatenate([element for element in ds.map(lambda x,y,z: y[label_features[0]]).as_numpy_iterator()])
    identifiers = np.concatenate([element for element in ds.map(get_z_from_xyz).as_numpy_iterator()])
    preds = model.predict(ds, verbose=2).astype(np.int32) # we dont need float precision
    df_columns = np.hstack([identifiers, labels.reshape(-1,1), preds]).astype(np.int32)

    df = pd.DataFrame(df_columns, columns=['ObjectID', 'TimeIndex'] + label_features + ['Preds'], dtype=np.int32)

    # TODO: zoom on main parts, plot postprocessed preds
    # TODO: add nice breaks https://stackoverflow.com/questions/5656798/is-there-a-way-to-make-a-discontinuous-axis-in-matplotlib
    if zoom:
        max_val = np.max(df[label_features[0]])
        timeindices = df.index[(df[label_features[0]] == max_val) | (df['Preds'] >= threshold)].to_numpy() # only consider locations with timeindex > 1
        print(timeindices.shape)
        for timeindex in timeindices:
            # Using index as timeindex????
            df.loc[timeindex-20:timeindex+20, 'keep'] = True
        df = df.loc[df['keep'] == True]

    fig, axes = plt.subplots(nrows=len(object_ids), ncols=1, figsize=(16,4*len(object_ids)))
    plt.tight_layout()
    for idx, object_id in enumerate(object_ids):
        axes[idx].plot(df.loc[df['ObjectID'] == int(object_id), label_features].to_numpy())
        axes[idx].plot(df.loc[df['ObjectID'] == int(object_id), 'Preds'].to_numpy())
        axes[idx].axhline(y=threshold, color='g', linestyle='--')
        axes[idx].title.set_text(object_id)
    fig.show()

    

def postprocess_predictions(preds_df,
                            dirs=['EW', 'NS'],
                            threshold=50.0,
                            add_initial_node=False,
                            clean_consecutives=True,
                            deepcopy=True):
    """Expects input df with columns [ObjectID, TimeIndex, EW_Loc, NS_Loc]
    """

    df = preds_df.copy(deep=deepcopy)
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
        sub_df = df.loc[df[f'{dir}_Loc'] == True]
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
    ground_truth_df = ground_truth_df.loc[ground_truth_df['ObjectID'].isin(object_ids)]
    subm_df = subm_df.loc[subm_df['ObjectID'].isin(object_ids)]

    # Filter direction
    ground_truth_df = ground_truth_df.loc[(ground_truth_df['Direction'].isin(dirs + ['ES']))] # Keep es row, to maintain object for evaluation
    subm_df = subm_df.loc[(subm_df['Direction'].isin(dirs))]

    # Filter TimeIndices
    if not with_initial_node:
        ground_truth_df = ground_truth_df.loc[(ground_truth_df['TimeIndex'] != 0)]
        subm_df = subm_df.loc[(subm_df['TimeIndex'] != 0)]

    # Initiate evaluator
    evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=subm_df, ignore_classes=True)
    precision, recall, f2, rmse, total_tp, total_fp, total_fn, total_df = evaluator.score()

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