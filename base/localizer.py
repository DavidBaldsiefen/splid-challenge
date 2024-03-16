import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset
import gc
import pickle
from base import datahandler
#import datahandler

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
                         prediction_stride=1,
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
                                            overview_as_second_input=isinstance(model, list),
                                            convolve_input_stride=True,
                                            shuffle=False, # if we dont use the majority method, its enough to just evaluate on nodes
                                            with_identifier=True,
                                            only_ew_sk=only_ew_sk,
                                            train_keys=train_keys[:(len(train_keys) if (train or test) else 1)],
                                            val_keys=val_keys[:(len(val_keys) if not (train or test) else 1)],
                                            stride=prediction_stride)
            ds = (datasets[0] if train else datasets[1]) if not test else datasets

            identifiers = np.concatenate([element for element in ds.map(get_y_from_xy).as_numpy_iterator()])

            # get predictions
            if not isinstance(model, list) and not (model.layers[0]._name == list(ds.element_spec[0].keys())[0]):
                print(f"Renaming model input to \'{list(ds.element_spec[0].keys())[0]}\' to ensure ds compatibility.")
                model.layers[0]._name = list(ds.element_spec[0].keys())[0]
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

    identifiers = np.concatenate([element for element in ds.map(get_z_from_xyz).as_numpy_iterator()])
    labels = np.concatenate([[element[ft] for ft in label_features] for element in ds.map(get_y_from_xyz).as_numpy_iterator()], axis=1).transpose(1,0)
    
    preds = np.asarray(model.predict(ds, verbose=2))[:,:,0].astype(np.int32).transpose(1,0) # we dont need float precision
    df_columns = np.hstack([identifiers, labels, preds]).astype(np.int32)

    df = pd.DataFrame(df_columns, columns=['ObjectID', 'TimeIndex'] + label_features + [ft+'_pred' for ft in label_features], dtype=np.int32)

    # TODO: zoom on main parts, plot postprocessed preds
    # TODO: add nice breaks https://stackoverflow.com/questions/5656798/is-there-a-way-to-make-a-discontinuous-axis-in-matplotlib
    if zoom:
        if len(label_features) == 1:
            timeindices = df.index[(df[label_features[0]] >= threshold) | (df[f'{label_features[0]}_pred'] >= threshold)].to_numpy() # only consider locations with timeindex > 1
        else:
            timeindices = df.index[(df[label_features[0]] >= threshold) | (df[label_features[1]] >= threshold) |
                                   (df[f'{label_features[0]}_pred'] >= threshold) | (df[f'{label_features[1]}_pred'] >= threshold)].to_numpy() # only consider locations with timeindex > 1
        print(timeindices.shape)
        for timeindex in timeindices:
            # Using index as timeindex????
            df.loc[timeindex-20:timeindex+20, 'keep'] = True
        df = df.loc[df['keep'] == True]

    fig, axes = plt.subplots(nrows=len(object_ids), ncols=1, figsize=(16,4*len(object_ids)))
    plt.tight_layout()
    for idx, object_id in enumerate(object_ids):
        
        for ft in label_features:
            axes[idx].plot(df.loc[df['ObjectID'] == int(object_id), ft].to_numpy(), label=ft+'_gt')
            axes[idx].plot(df.loc[df['ObjectID'] == int(object_id), ft+'_pred'].to_numpy(), label=ft, linestyle='--')
        axes[idx].axhline(y=threshold, color='g', linestyle='--')
        axes[idx].title.set_text(object_id)
        axes[idx].legend()
    fig.show()

def postprocess_predictions(preds_df,
                            dirs=['EW', 'NS'],
                            thresholds=[50.0], # if len2, gets interpreted as per-direction threshold
                            add_initial_node=False,
                            clean_consecutives=True,
                            clean_neighbors_below_distance=-1,
                            legacy=False,
                            deepcopy=True):
    """Expects input df with columns [ObjectID, TimeIndex, EW_Loc, NS_Loc]
    """

    df = preds_df.copy(deep=deepcopy)
    object_ids = df['ObjectID'].unique()
    df['Any_Loc'] = False
    # apply threshold
    for dir_idx, dir in enumerate(dirs):
        dir_threshold = thresholds[0] if len(thresholds)==1 else thresholds[dir_idx]
        df[f'{dir}_Loc'] = df[f'{dir}_Loc'] >= dir_threshold
        df['Any_Loc'] = df['Any_Loc'] | df[f'{dir}_Loc']
    
    # remove consecutive location predictions, and replace them only with their center
    if clean_consecutives and not legacy:
        dir_dfs = []
        for dir in dirs:
            dir_df = df.loc[df[f'{dir}_Loc'] == True].copy().sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
            # the other direction needs to be set to false for this dir. this avoids duplicates later on
            other_dir = 'EW' if dir=='NS' else 'NS'
            dir_df[f"{other_dir}_Loc"] = False
            # find consecutives
            dir_df['consecutive'] = (dir_df['TimeIndex'] - dir_df['TimeIndex'].shift(1) != 1).cumsum()
            # Filter rows where any number of consecutive values follow each other
            dir_df=dir_df.groupby('consecutive').apply(lambda sub_df: sub_df.iloc[int(len(sub_df)/2), :]).reset_index(drop=True).drop(columns=['consecutive'])
            dir_dfs.append(dir_df)
        df = pd.concat(dir_dfs) if len(dir_dfs) > 1 else dir_dfs[0]
    # remove duplicates - if there are two detections at the same place (one for each dir), they will still be maintained
    df = df.loc[df.duplicated(keep='first')==False].reset_index(drop=True)

    if clean_consecutives and legacy:    # Legacy method
        df = df.loc[(df['Any_Loc'] == True)].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
        df['consecutive'] = (df['TimeIndex'] - df['TimeIndex'].shift(1) != 1).cumsum()
        # Filter rows where any number of consecutive values follow each other
        df=df.groupby('consecutive').apply(lambda sub_df: sub_df.iloc[int(len(sub_df)/2), :]).reset_index(drop=True).drop(columns=['consecutive'])

    # remove TPs that are just too close together, as its likely they are duplicate detections
    if clean_neighbors_below_distance>0:
        dir_dfs = []
        for dir in dirs:
            # compute diffs between current and previous timeindex
            sub_df = df.loc[df[f'{dir}_Loc'] == True].copy().sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
            sub_df['diff'] = 3000
            sub_df.loc[(df['TimeIndex'] > 0), 'diff'] = sub_df.loc[(df['TimeIndex'] > 0), 'TimeIndex'].diff()
            sub_df.loc[sub_df['diff']<0, 'diff'] = 3000
            # find the distances below the threshold
            short_distance_indices = sub_df.index[sub_df['diff']<clean_neighbors_below_distance]
            # for each diff loc where the previous entry is of the same object, replace both detections with one in the middle
            for index in short_distance_indices:
                if index == 0:
                    continue
                if sub_df.at[index-1, 'ObjectID'] == sub_df.at[index, 'ObjectID']:
                    sub_df.at[index, 'TimeIndex'] = int((sub_df.at[index-1, 'TimeIndex']+sub_df.at[index, 'TimeIndex'])/2.0)
                    sub_df.drop(index=(index-1), inplace=True)
            dir_dfs.append(sub_df)
        df = pd.concat(dir_dfs).reset_index(drop=True)
            


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
    df = pd.concat(sub_dfs) if len(sub_dfs)>1 else sub_dfs[0]

    df['Node'] = 'UNKNOWN'
    df['Type'] = 'UNKNOWN'

    df = df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

    return df

def perform_submission_pipeline(localizer_dir,
                                scaler_dir,
                                split_dataframes,
                                output_dirs,
                                thresholds,
                                legacy_clean_consecutives=False,
                                clean_neighbors_below_distance=-1,
                                non_transform_features=[],
                                diff_transform_features=[],
                                sin_transform_features=[],
                                sin_cos_transform_features=[],
                                overview_features_mean=[],
                                overview_features_std=[],
                                add_daytime_feature=False,
                                add_yeartime_feature=False,
                                add_linear_timeindex=False,
                                padding='zero',
                                input_history_steps=128,
                                input_future_steps=128,
                                input_stride=2,
                                per_object_scaling=False,
                                ):
    """Perform the entire submission pipeline, i.e. create ds_gen, run predictor, perform postprocessing"""

    print(f"Predicting locations using model \"{localizer_dir}\" and scaler \"{scaler_dir}\"")

    scaler = pickle.load(open(scaler_dir, 'rb')) if scaler_dir is not None else None
    ds_gen = datahandler.DatasetGenerator(split_df=split_dataframes,
                                            non_transform_features=non_transform_features,
                                            diff_transform_features=diff_transform_features,
                                            sin_transform_features=sin_transform_features,
                                            sin_cos_transform_features=sin_cos_transform_features,
                                            overview_features_mean=overview_features_mean,
                                            overview_features_std=overview_features_std,
                                            add_daytime_feature=add_daytime_feature,
                                            add_yeartime_feature=add_yeartime_feature,
                                            add_linear_timeindex=add_linear_timeindex,
                                            with_labels=False,
                                            train_val_split=1.0,
                                            input_stride=input_stride,
                                            padding=padding,
                                            input_history_steps=input_history_steps,
                                            input_future_steps=input_future_steps,
                                            per_object_scaling=per_object_scaling,
                                            custom_scaler=scaler,
                                            unify_value_ranges=True,
                                            input_dtype=np.float32,
                                            sort_inputs=True,
                                            seed=69)

    
    localizer = tf.keras.models.load_model(localizer_dir, compile=False)

    preds_df = create_prediction_df(ds_gen=ds_gen,
                                    model=localizer,
                                    train=False,
                                    test=True,
                                    output_dirs=output_dirs,
                                    prediction_batches=5,
                                    verbose=2)

    subm_df = postprocess_predictions(preds_df=preds_df,
                                                dirs=output_dirs,
                                                thresholds=thresholds,
                                                add_initial_node=False, # Do not add initial nodes just yet
                                                clean_consecutives=True,
                                                legacy=legacy_clean_consecutives,
                                                clean_neighbors_below_distance=clean_neighbors_below_distance)
    
    return subm_df

def evaluate_localizer(subm_df, gt_path, object_ids, dirs=['EW', 'NS'], with_initial_node=False, return_scores=False, nodes_to_consider=['ID', 'IK', 'AD'], verbose=1):
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

    # Initiate evaluator & get scoring df
    evaluator = evaluation.NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=subm_df, ignore_classes=True, verbose=verbose)
    precision, recall, f2, rmse, total_tp, total_fp, total_fn, total_df = evaluator.score()

    tp_ID = 0
    fn_ID = 0
    tp_IK = 0
    fn_IK = 0
    tp_AD = 0
    fn_AD = 0

    if total_df is not None:
        # FP cannot be accounted for independently, as we cannot know _what_ was falsely detected
        tp_ID = len(total_df.loc[(total_df['Node'] == 'ID') & (total_df['classification'] == 'TP')])
        fn_ID = len(total_df.loc[(total_df['Node'] == 'ID') & (total_df['classification'] == 'FN')])
        tp_IK = len(total_df.loc[(total_df['Node'] == 'IK') & (total_df['classification'] == 'TP')])
        fn_IK = len(total_df.loc[(total_df['Node'] == 'IK') & (total_df['classification'] == 'FN')])
        tp_AD = len(total_df.loc[(total_df['Node'] == 'AD') & (total_df['classification'] == 'TP')])
        fn_AD = len(total_df.loc[(total_df['Node'] == 'AD') & (total_df['classification'] == 'FN')])

    # Re-calculate metrics
    total_tp = 0
    total_tp += tp_ID if 'ID' in nodes_to_consider else 0
    total_tp += tp_IK if 'IK' in nodes_to_consider else 0
    total_tp += tp_AD if 'AD' in nodes_to_consider else 0
    total_fn = 0
    total_fn += fn_ID if 'ID' in nodes_to_consider else 0
    total_fn += fn_IK if 'IK' in nodes_to_consider else 0
    total_fn += fn_AD if 'AD' in nodes_to_consider else 0
    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp)
    else:
        precision = 0.0
        recall = 0.0
        f2 = 0.0
        rmse = 0.0

    if verbose>0:
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F2: {f2:.3f}')
        print(f'RMSE: {float(rmse):.4}')
        print(f'TP: {total_tp} FP: {total_fp} FN: {total_fn}')

        if total_df is not None:
            print(f"TP/FN based on Node:")
            print(f"ID: {tp_ID}|{fn_ID}")
            print(f"IK: {tp_IK}|{fn_IK}")
            print(f"AD: {tp_AD}|{fn_AD}")


    if return_scores:
        return {'Precision':precision, 'Recall':recall, 'F2':f2, 'RMSE':rmse, 'TP':total_tp, 'FP':total_fp, 'FN':total_fn,
                'TP_ID':tp_ID,'FN_ID':fn_ID,'TP_IK':tp_IK,'FN_IK':fn_IK,'TP_AD':tp_AD,'FN_AD':fn_AD}
    else:
        return evaluator, subm_df, total_df
    
def perform_evaluation_pipeline(ds_gen,
                                model,
                                ds_type,
                                gt_path,
                                output_dirs,
                                prediction_batches,
                                thresholds,
                                object_limit=None,
                                with_initial_node=False,
                                nodes_to_consider=['ID', 'IK', 'AD'],
                                prediction_stride=1,
                                clean_neighbors_below_distance=-1,
                                legacy_postprocessing=False,
                                verbose=2):
    
    preds_df = create_prediction_df(ds_gen=ds_gen,
                                model=model,
                                train=True if ds_type=='train' else False,
                                test=True if ds_type=='test' else False,
                                stateful=False,
                                output_dirs=output_dirs,
                                object_limit=object_limit,
                                only_ew_sk=False,
                                ds_batch_size=1024,
                                prediction_batches=prediction_batches,
                                prediction_stride=prediction_stride,
                                verbose=verbose)
    
    all_scores = []
    best_scores_per_dir = [-1.0, -1.0]
    best_thresholds_per_dir = [-1.0, -1.0]
    # perform threshold analysis
    for threshold in thresholds:
        subm_dfs = []
        for dir_idx, dir in enumerate(output_dirs):
            subm_df = postprocess_predictions(preds_df=preds_df,
                                            dirs=[dir],
                                            thresholds=[threshold],
                                            add_initial_node=with_initial_node,
                                            clean_consecutives=True,
                                            legacy=legacy_postprocessing,
                                            clean_neighbors_below_distance=clean_neighbors_below_distance,
                                            deepcopy=False)
            subm_dfs.append(subm_df)

            dir_score = evaluate_localizer(subm_df=subm_df,
                                    gt_path=gt_path,
                                    object_ids=list(map(int, ds_gen.val_keys if ds_type=='val' else ds_gen.train_keys))[:object_limit],
                                    dirs=[dir],
                                    with_initial_node=with_initial_node,
                                    nodes_to_consider=nodes_to_consider, 
                                    return_scores=True,
                                    verbose=0)
            if best_scores_per_dir[dir_idx] < dir_score['F2']:
                best_scores_per_dir[dir_idx] = dir_score['F2']
                best_thresholds_per_dir[dir_idx] = threshold
       
        subm_df = pd.concat(subm_dfs).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

        scores = evaluate_localizer(subm_df=subm_df,
                                    gt_path=gt_path,
                                    object_ids=list(map(int, ds_gen.val_keys if ds_type=='val' else ds_gen.train_keys))[:object_limit],
                                    dirs=output_dirs,
                                    with_initial_node=with_initial_node,
                                    nodes_to_consider=nodes_to_consider, 
                                    return_scores=True,
                                    verbose=0)
        scores['Threshold'] = threshold
        all_scores.append(scores)
        print(f"Threshold: {scores['Threshold']:.1f}\t Precision: {scores['Precision']:.2f} Recall: {scores['Recall']:.2f} F2: {scores['F2']:.3f} RMSE: {scores['RMSE']:.2f} | TP: {scores['TP']} FP: {scores['FP']} FN: {scores['FN']} (ID: {scores['TP_ID']}|{scores['FN_ID']} IK: {scores['TP_IK']}|{scores['FN_IK']} AD: {scores['TP_AD']}|{scores['FN_AD']})")
    
    if(len(output_dirs)>1):
        subm_df = postprocess_predictions(preds_df=preds_df,
                                            dirs=output_dirs,
                                            thresholds=best_thresholds_per_dir,
                                            add_initial_node=with_initial_node,
                                            clean_consecutives=True,
                                            legacy=legacy_postprocessing,
                                            clean_neighbors_below_distance=clean_neighbors_below_distance,
                                            deepcopy=False)
        subm_df = subm_df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
        scores = evaluate_localizer(subm_df=subm_df,
                                gt_path=gt_path,
                                object_ids=list(map(int, ds_gen.val_keys if ds_type=='val' else ds_gen.train_keys))[:object_limit],
                                dirs=output_dirs,
                                with_initial_node=with_initial_node,
                                nodes_to_consider=nodes_to_consider, 
                                return_scores=True,
                                verbose=0)
        scores['Threshold'] = best_thresholds_per_dir[0]+best_thresholds_per_dir[1]/100.0 # just to ease current workflow
        scores['Thresholds'] = best_thresholds_per_dir
        scores['Per-Dir-Thresholds'] = True
        all_scores.append(scores)
        print(f"Per-Dir-Thresholds: {scores['Thresholds'][0]:.1f}-{scores['Thresholds'][1]:.1f}\t Precision: {scores['Precision']:.2f} Recall: {scores['Recall']:.2f} F2: {scores['F2']:.3f} RMSE: {scores['RMSE']:.2f} | TP: {scores['TP']} FP: {scores['FP']} FN: {scores['FN']} (ID: {scores['TP_ID']}|{scores['FN_ID']} IK: {scores['TP_IK']}|{scores['FN_IK']} AD: {scores['TP_AD']}|{scores['FN_AD']})")

    return all_scores