#################################################################################################
####### Parts of this code have originated from https://github.com/ARCLab-MIT/splid-devkit ######
####### It has been modified under the license listed in the main directory                ######
#################################################################################################                                                                                               
## Original License:                                                                           ##                       
##                                                                                             ##
## MIT License                                                                                 ## 
##                                                                                             ##
## Copyright (c) 2023 Peng Mun Siew                                                            ## 
##                                                                                             ##
## Permission is hereby granted, free of charge, to any person obtaining a copy                ## 
## of this software and associated documentation files (the "Software"), to deal               ## 
## in the Software without restriction, including without limitation the rights                ## 
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell                   ## 
## copies of the Software, and to permit persons to whom the Software is                       ## 
## furnished to do so, subject to the following conditions:                                    ## 
##                                                                                             ##
## The above copyright notice and this permission notice shall be included in all              ## 
## copies or substantial portions of the Software.                                             ## 
#################################################################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error
import argparse
from fastcore.all import *

class NodeDetectionEvaluator:
    def __init__(self, ground_truth, participant, tolerance=6, ignore_classes=False, ignore_nodes=False, verbose=1):
        self.ground_truth = ground_truth.copy()
        self.participant = participant.copy()
        self.tolerance = tolerance
        self.ignore_classes=ignore_classes
        self.ignore_nodes=ignore_nodes
        if ignore_classes and verbose>0: print("Evaluator ignoring classifications")
        if ignore_nodes and verbose>0: print("Evaluator ignoring nodes (i.e. only evaluating type)")
        
    def evaluate(self, object_id):
        gt_object = self.ground_truth[(self.ground_truth['ObjectID'] == object_id) & \
                          (self.ground_truth['Direction'] != 'ES')].copy()
        p_object = self.participant[(self.participant['ObjectID'] == object_id) & \
                                    (self.participant['Direction'] != 'ES')].copy()
        p_object['matched'] = False
        p_object['classification'] = None
        p_object['distance'] = None
        gt_object['classification'] = None
        gt_object['distance'] = None
        tp = 0
        fp = 0
        fn = 0

        for gt_idx, gt_row in gt_object.iterrows():
            matching_participant_events = p_object[
                (p_object['TimeIndex'] >= gt_row['TimeIndex'] - self.tolerance) &
                (p_object['TimeIndex'] <= gt_row['TimeIndex'] + self.tolerance) &
                (p_object['Direction'] == gt_row['Direction']) &
                (p_object['matched'] == False)
            ]

            if len(matching_participant_events) > 0:
                p_idx = matching_participant_events.index[0]
                p_row = matching_participant_events.iloc[0]
                distance = p_row['TimeIndex'] - gt_row['TimeIndex']
                if (self.ignore_classes or 
                    (self.ignore_nodes and (p_row['Type'] == gt_row['Type']))
                    or (p_row['Node'] == gt_row['Node'] and p_row['Type'] == gt_row['Type'])):
                    tp += 1
                    gt_object.loc[gt_idx, 'classification'] = 'TP'
                    gt_object.loc[gt_idx, 'distance'] = distance
                    p_object.loc[p_idx, 'classification'] = 'TP'
                    p_object.loc[p_idx, 'distance'] = distance
                else:
                    fp += 1
                    gt_object.loc[gt_idx, 'classification'] = 'FP'
                    gt_object.loc[gt_idx, 'distance'] = distance
                    p_object.loc[p_idx, 'classification'] = 'FP'
                    p_object.loc[p_idx, 'distance'] = distance
                p_object.loc[matching_participant_events.index[0], 'matched'] = True
            else:
                fn += 1
                gt_object.loc[gt_idx, 'classification'] = 'FN'
                
        additional_fp = p_object[~p_object['matched']].copy()
        fp += len(additional_fp)
        p_object.loc[additional_fp.index, 'classification'] = 'FP'

        return tp, fp, fn, gt_object, p_object
    
    def score(self):
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_distances = []
        gt_objs = []
        p_objs = []
        for object_id in self.ground_truth['ObjectID'].unique():
            _, _, _, gt_object, p_object = self.evaluate(object_id)
            
            total_tp += len(p_object[p_object['classification'] == 'TP'])
            total_fp += len(p_object[p_object['classification'] == 'FP'])
            total_fn += len(gt_object[gt_object['classification'] == 'FN'])
            gt_objs.append(gt_object)
            p_objs.append(p_object.loc[p_object['classification'] == 'FP'])
            total_distances.extend(
                p_object[p_object['classification'] == 'TP']['distance'].tolist()
            )
        if ((total_tp + total_fp) < 1):
            print("Warning: No true AND false positives! Did you compare train results agaist val-ground_truth?")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None

        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp)
        rmse = np.sqrt((sum(d ** 2 for d in total_distances) / len(total_distances))) if total_distances else 0

        total_df = pd.concat(gt_objs+p_objs).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)

        return precision, recall, f2, rmse, total_tp, total_fp, total_fn, total_df


    def plot(self, object_id):
        tp, fp, fn, gt_object, p_object = self.evaluate(object_id)
        tp_distances = p_object[p_object['classification'] == 'TP']['distance']
        mse = np.sqrt(mean_squared_error([0] * len(tp_distances), tp_distances)) if tp_distances.size else 0
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ground_truth_EW = gt_object[gt_object['Direction'] == 'EW']
        ground_truth_NS = gt_object[gt_object['Direction'] == 'NS']
        participant_EW = p_object[p_object['Direction'] == 'EW']
        participant_NS = p_object[p_object['Direction'] == 'NS']
        self._plot_type_timeline(ground_truth_EW, participant_EW, ax1, 'EW')
        self._plot_type_timeline(ground_truth_NS, participant_NS, ax2, 'NS')
        plt.xlabel('TimeIndex')
        title_info = f"Object {object_id}: TPs={tp}, FPs={fp}, FNs={fn}"
        fig.suptitle(title_info, fontsize=10)
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        legend_elements = [
            plt.Line2D([0], [0], color='green', linestyle='dashed', label='True Positive (TP)'),
            plt.Line2D([0], [0], color='blue', linestyle='dashed', label='False Positive (FP)'),
            plt.Line2D([0], [0], color='red', linestyle='dashed', label='False Negative (FN)'),
            patches.Patch(color='grey', alpha=0.2, label='Tolerance Interval')
        ]
        plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4)
        plt.show()


    def _plot_type_timeline(self, ground_truth_type, participant_type, ax, 
                            type_label):
        for _, row in ground_truth_type.iterrows():
            label = row['Node'] + '-' + row['Type']
            ax.scatter(row['TimeIndex'], 2, color='black')
            ax.text(row['TimeIndex'] + 3, 2.05, label, rotation=45)
            ax.fill_betweenx([1, 2], row['TimeIndex'] - self.tolerance,
                             row['TimeIndex'] + self.tolerance, color='grey',
                             alpha=0.2)
            if row['classification'] == 'TP':
                ax.text(row['TimeIndex'] + self.tolerance + .5, 1.5, 
                        str(row['distance']), 
                        color='black')
                ax.plot([row['TimeIndex'], 
                         row['TimeIndex'] + row['distance']], [2, 1], 
                         color='green', linestyle='dashed')
            elif row['classification'] == 'FP':
                ax.plot([row['TimeIndex'], 
                         row['TimeIndex'] + row['distance']], [2, 1], 
                         color='blue', linestyle='dashed')
            elif row['classification'] == 'FN':
                ax.plot([row['TimeIndex'], 
                         row['TimeIndex']], [2, 2.2], color='red', 
                         linestyle='dashed')

        for _, row in participant_type.iterrows():
            label = row['Node'] + '-' + row['Type']
            ax.scatter(row['TimeIndex'], 1, color='black')
            ax.text(row['TimeIndex'] + 3, 1.05, label, rotation=45)
            if row['classification'] == 'FP' and row['matched'] == False:
                ax.plot([row['TimeIndex'], row['TimeIndex']], [1, 0.8], 
                        color='blue', linestyle='dashed')

        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.yaxis.grid(True)
        ax.set_yticks([1, 2])
        ax.set_yticklabels(['Participant', 'Ground truth'])
        ax.set_title(type_label)


def merge_label_files(label_folder):
    """
    Merges all label files in a given folder into a single pandas DataFrame. 
    The filenames must be in the format <ObjectID>.csv, and the object id will
    be extracted from the filename and added as a column to the DataFrame.

    Args:
        label_folder (str): The path to the folder containing the label files.

    Returns:
        pandas.DataFrame: A DataFrame containing the merged label data.
    """
    label_data = []
    label_folder = Path(label_folder).expanduser()
    for file_path in label_folder.ls():
        df = pd.read_csv(file_path)
        oid_s = os.path.basename(file_path).split('.')[0]  # Extract ObjectID from filename
        df['ObjectID'] = int(oid_s)
        label_data.append(df)

    label_data = pd.concat(label_data)
    label_data = label_data[['ObjectID'] + list(label_data.columns[:-1])]
    return label_data


def run_evaluator(ground_truth_path=None, participant_path=None, plot_object=None):

    print('participant_path:', participant_path)
    print('ground_truth_path:', ground_truth_path)

    if participant_path is None:
        print('Reading participant_toy.csv')
        participant_df = pd.read_csv('participant_toy.csv')
    else:
        participant_path = Path(participant_path).expanduser()
        
        if participant_path.is_dir():
            print("a")
            participant_df = merge_label_files(participant_path)  
        else:
            print("b", participant_path)
            participant_df = pd.read_csv(participant_path)
    
    if ground_truth_path is None:
        ground_truth_df = pd.read_csv('ground_truth_toy.csv')
    else:
        ground_truth_path = Path(ground_truth_path).expanduser()
        if ground_truth_path.is_dir():
            print('a')
            ground_truth_df = merge_label_files(ground_truth_path)
        else:
            print('b', ground_truth_path)
            ground_truth_df = pd.read_csv(ground_truth_path)
    
    print(participant_df.head(10))

    print(ground_truth_df.head(10))

    # Create a NodeDetectionEvaluator instance
    evaluator = NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=participant_df, tolerance=6)
    precision, recall, f2, rmse, total_tp, total_fp, total_fn, total_df = evaluator.score()
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F2: {f2:.3f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'TP: {total_tp} FP: {total_fp} FN: {total_fn}')

    # Plot the evaluation for the selected object (if any)
    if plot_object:
        evaluator.plot(object_id=plot_object)
    return precision, recall, f2, rmse

def evaluate_localizer(ds_gen, split_dataframes, gt_path, model, train=True, with_initial_node=False, remove_consecutives=True, direction='EW', prediction_threshold=0.5, return_scores=False, verbose=2):
    t_ds, v_ds = ds_gen.get_datasets(256, label_features=[f'{direction}_Node_Location'], shuffle=False, with_identifier=True, stride=1)
    ds = t_ds if train else v_ds

    def get_x(x,y,z): return x # with lambdas, there is an annoying warning otherwise
    def get_y(x,y,z): return y
    def get_z(x,y,z): return z
    inputs = np.concatenate([element for element in ds.map(get_x).as_numpy_iterator()])
    labels = np.concatenate([element[f'{direction}_Node_Location'] for element in ds.map(get_y).as_numpy_iterator()])
    identifiers = np.concatenate([element for element in ds.map(get_z).as_numpy_iterator()])

    # get predictions
    preds = model.predict(inputs, verbose=verbose)
    #preds_argmax = np.argmax(preds, axis=1) # for legacy (non-binary) classification
    preds_argmax = (preds>=prediction_threshold).astype(int) # for binary preds

    df = pd.DataFrame(np.concatenate([identifiers.reshape(-1,2)], axis=1), columns=['ObjectID', 'TimeIndex'], dtype=np.int32)
    df['Location'] = labels
    df[f'Location_Pred'] = preds_argmax
    df[f'Location_Pred_Raw'] = preds

    # add initial node prediction
    if with_initial_node:
        for obj in ds_gen.train_keys if train else ds_gen.val_keys:
            df = df.sort_index()
            df.loc[-1] = [int(obj), 0, 1, 1] # objid, timeindex, location, location_pred
            df.index = df.index + 1
            df = df.sort_index()

    # set direction as well as dummy values for node and type
    df['Direction'] = direction
    df['Node'] = 'None'
    df['Type'] = 'None'
    
    df = df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
    df_filtered = df.loc[(df['Location_Pred'] == 1)]

    # remove consecutives detections
    # TODO: this fails when two consecutive objects have detections at exactly consecutive timeindices - a corner case I ignore for now ;)
    if remove_consecutives:
        df_filtered['consecutive'] = (df_filtered['TimeIndex'] - df_filtered['TimeIndex'].shift(1) != 1).cumsum()
        # Filter rows where any number of consecutive values follow each other
        df_filtered=df_filtered.groupby('consecutive').apply(lambda df: df.iloc[int(len(df)/2), :]).reset_index(drop=True).drop(columns=['consecutive'])

    ground_truth_from_file = pd.read_csv(gt_path).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
    ground_truth_from_file = ground_truth_from_file[ground_truth_from_file['ObjectID'].isin(map(int, ds_gen.train_keys if train else ds_gen.val_keys))].copy()
    ground_truth_from_file = ground_truth_from_file[(ground_truth_from_file['Direction'] == direction)]

    # remove initial nodes, as they can always be localized anyway
    if not with_initial_node:
        df_filtered = df_filtered.loc[(df_filtered['TimeIndex'] != 0)]
        ground_truth_from_file = ground_truth_from_file.loc[(ground_truth_from_file['TimeIndex'] != 0)]
    evaluator = NodeDetectionEvaluator(ground_truth=ground_truth_from_file, participant=df_filtered, ignore_classes=True)
    precision, recall, f2, rmse, total_tp, total_fp, total_fn = evaluator.score()

    if verbose>0:
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F2: {f2:.3f}')
        print(f'RMSE: {float(rmse):.4}')
        print(f'TP: {total_tp} FP: {total_fp} FN: {total_fn}')

    if return_scores:
        return {'Precision':precision, 'Recall':recall, 'F2':f2, 'RMSE':rmse, 'TP':total_tp, 'FP':total_fp, 'FN':total_fn}
    else:
        return evaluator, df

def evaluate_classifier(ds_gen, gt_path, model, model_outputs=['EW', 'NS'], train=True, with_initial_node=True, only_initial_nodes=False, return_scores=False, majority_segment_labels=True, verbose=2):

    t_ds, v_ds = ds_gen.get_datasets(batch_size=512,
                                     label_features=model_outputs,
                                     shuffle=False,
                                     only_nodes=not majority_segment_labels, # if we dont use the majority method, its enough to just evaluate on nodes
                                     with_identifier=True,
                                     stride=1)
    ds = t_ds if train else v_ds

    ground_truth_df = pd.read_csv(gt_path).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
    ground_truth_df = ground_truth_df[ground_truth_df['ObjectID'].isin(map(int, ds_gen.train_keys if train else ds_gen.val_keys))].copy()

    inputs = np.concatenate([element for element in ds.map(lambda x,y,z: x).as_numpy_iterator()])
    #labels = np.concatenate([element['EW_Node_Location'] for element in ds.map(lambda x,y,z: y).as_numpy_iterator()])
    identifiers = np.concatenate([element for element in ds.map(lambda x,y,z: z).as_numpy_iterator()])

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

    objs = df['ObjectID'].unique()

    # merge locs into df
    ground_truth_df['Loc_EW'] = 0
    ground_truth_df['Loc_NS'] = 0
    ground_truth_df.loc[ground_truth_df['Direction'] == 'EW', ['Loc_EW']] = 1
    ground_truth_df.loc[ground_truth_df['Direction'] == 'NS', ['Loc_NS']] = 1
    df = df.merge(ground_truth_df[['ObjectID', 'TimeIndex', 'Loc_EW', 'Loc_NS']], how='left', on=['ObjectID', 'TimeIndex'])
    for obj in objs:
        node_indices = ground_truth_df.index[(ground_truth_df['ObjectID'] == obj)].to_list()

    # add initial nodes
    if with_initial_node and(np.min(identifiers[:,1] > 0)) and majority_segment_labels:
        # TODO: this may fuck shit up? not sure
        for obj in objs:
            new_index = df.index.max()+1
            df.loc[new_index] = df.loc[0].copy() # copy  a random row
            df.at[new_index, 'ObjectID'] = obj
            df.at[new_index, 'TimeIndex'] = 0
            df.at[new_index, 'Loc_EW'] = 0
            df.at[new_index, 'Loc_NS'] = 0
    df = df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True) # do not remove! index ordering is important

    # assign locations with the majority label in the following segment
    if ('NS_Type' in model_outputs or 'EW_Type' in model_outputs) and majority_segment_labels:
        for obj in objs:
            for dir in ['EW', 'NS']:
                vals = df[f'{dir}_Type_Pred'].to_numpy()

                obj_indices = df.index[(df['ObjectID'] == obj)].to_list()
                loc_indices = df.index[(df['ObjectID'] == obj) & (df[f'Loc_{dir}'] == 1)].to_list() # df indices where loc=1
                 
                # determine segments, determine majority value
                loc_indices = [np.min(obj_indices)] + loc_indices + [np.max(obj_indices)]
                loc_indices = np.unique(loc_indices, axis=0) # it may happen that index 0 exists twice; in that case remove duplicate
                loc_indices.sort()
                locs = df.iloc[loc_indices]['TimeIndex'].to_list() # actual timeindex
                
                segments = np.split(vals, loc_indices)
                most_common_values = [np.bincount(segment).argmax() for segment in segments[1:-1]]
                #print(locs, loc_indices, most_common_values)
                # assign majority value of segment _after_ a loc to the loc
                majority_vals_decoded = ds_gen.type_label_encoder.inverse_transform(most_common_values)
                for idx, major_val in enumerate(majority_vals_decoded):
                    df.at[loc_indices[idx], f'{dir}_Type'] = major_val
    
    # now, assign the real label to the locations
    if not ('EW_Node' in model_outputs or 'EW_Type' in model_outputs or 'EW' in model_outputs):
        if verbose > 0: print("Considering only NS direction")
        ground_truth_df = ground_truth_df[(ground_truth_df['Direction'] == 'NS')]
    if not ('NS_Node' in model_outputs or 'NS_Type' in model_outputs or 'NS' in model_outputs):
        if verbose > 0: print("Considering only EW direction")
        ground_truth_df = ground_truth_df[(ground_truth_df['Direction'] == 'EW')]
    gt_columns_to_keep = ['ObjectID', 'TimeIndex', 'Direction'] # available: direction, node, type
    if not ('NS_Node' in model_outputs or 'EW_Node' in model_outputs or 'EW' in model_outputs or 'NS' in model_outputs):
        if verbose > 0: print("Assuming perfect nodes")
        gt_columns_to_keep += ['Node']
    if not ('NS_Type' in model_outputs or 'EW_Type' in model_outputs or 'EW' in model_outputs or 'NS' in model_outputs):
        gt_columns_to_keep += ['Type']
        if verbose > 0: print("Assuming perfect types")
    df = df.merge(ground_truth_df[gt_columns_to_keep], how='right', on = ['ObjectID', 'TimeIndex'])

    if not 'Type' in df.columns.values.tolist():
        df['Type'] = pd.NA
    if not 'Node' in df.columns.values.tolist():
        df['Node'] = pd.NA
    if 'EW_Node' in df.columns.values.tolist():
        df.loc[df['Direction'] == 'EW', 'Node'] = df.loc[df['Direction'] == 'EW', 'EW_Node']
    if 'EW_Type' in df.columns.values.tolist():
        df.loc[df['Direction'] == 'EW', 'Type'] = df.loc[df['Direction'] == 'EW', 'EW_Type']
    if 'NS_Node' in df.columns.values.tolist():
        df.loc[df['Direction'] == 'NS', 'Node'] = df.loc[df['Direction'] == 'NS', 'NS_Node']
    if 'NS_Type' in df.columns.values.tolist():
        df.loc[df['Direction'] == 'NS', 'Type'] = df.loc[df['Direction'] == 'NS', 'NS_Type']

    # Lets add our background knowledge:
    # 1) For timeindex 0, the node is always SS
    df.loc[df['TimeIndex'] == 0, 'Node'] = 'SS'
    # 2) AD, ID is always combined with NK
    df.loc[(df['Node'] == 'AD') | (df['Node'] == 'ID'), 'Type'] = 'NK'
    # 3) IK is always combined with HK/CK/EK
    df.loc[(df['Node'] == 'IK') & (df['Type'] == 'NK'), 'Type'] = 'CK' # CK is most common

    # remove ES rows
    df = df.replace('na', pd.NA)
    df.dropna(inplace=True)

    # remove initial nodes
    if not with_initial_node:
        if verbose > 0: print("Ignoring initial nodes")
        df = df.loc[df['TimeIndex'] != 0]

    if only_initial_nodes:
        if verbose > 0: print("Considering ONLY initial nodes")
        df = df.loc[df['TimeIndex'] == 0]

    if only_initial_nodes and (not with_initial_node):
        print("Warning: No detections, as only_initial_node=True and with_ initial_node=False!")

    evaluator = NodeDetectionEvaluator(ground_truth=ground_truth_df, participant=df)
    precision, recall, f2, rmse, total_tp, total_fp, total_fn = evaluator.score()
    if verbose > 0:
        print(f'Precision: {precision:.2f}')
        print(f'TP: {total_tp} FP: {total_fp}')

    if return_scores:
        return {'Precision':precision, 'TP':total_tp, 'FP':total_fp}
    else:
        return df, ground_truth_df, evaluator

if __name__ == "__main__":
    if 'ipykernel' in sys.modules:
        run_evaluator(plot_object=True)
    else:
        # Parse the arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--participant', type=str, required=False, 
                            help='Path to the participant file or folder. \
                                If a folder is provided, all the CSV files in the \
                                    folder will be used. If none, the toy example \
                                        will be used.')
        parser.add_argument('--ground_truth', type=str, required=False,
                            help='Path to the ground truth file. If none, the toy \
                                example will be used.')
        parser.add_argument('--plot_object', type=int, required=False,
                    help='Object ID to plot.')
        args = parser.parse_args()
        run_evaluator(args.ground_truth, args.participant, args.plot_object)