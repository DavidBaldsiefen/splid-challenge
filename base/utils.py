import pandas as pd
from tqdm import tqdm
from fastcore.basics import Path
import random
import numpy as np
import tensorflow as tf

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

def plot_object(ObjectID, dataset_path, features=[], label_df=None, pred_df=None, file_prefix=''):
    import matplotlib.pyplot as plt

    object_df = pd.read_csv(Path(dataset_path) / Path(file_prefix + str(ObjectID) + '.csv'))
    object_df = object_df.iloc[0:200]
    object_df['TimeIndex'] = range(len(object_df))
    labels = None if label_df is None else label_df.loc[label_df['ObjectID'] == int(ObjectID)]

    if not features:
        features = ['Eccentricity', 'Semimajor Axis (m)', 'Inclination (deg)', 'RAAN (deg)',
                    'Argument of Periapsis (deg)', 'True Anomaly (deg)', 'Latitude (deg)',
                    'Longitude (deg)']
        
    fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(16,3*len(features)))
    for ft_idx, input_ft in enumerate(features):

        axes[ft_idx].plot(object_df['TimeIndex'][::1], object_df[input_ft][::1], label=input_ft)
        axes[ft_idx].plot(object_df['TimeIndex'][::5], object_df[input_ft][::5], label=input_ft)

        wraparound_offset = ([180, -180] if input_ft == 'Longitude (deg)' else
                                     [270, -90] if input_ft in ['True Anomaly (deg)'] else # True Anomaly should usually increase, but small decreases are possible
                                     [180, -180] if input_ft in ['Argument of Periapsis (deg)', 'RAAN (deg)'] else
                                     [])
        diff_vals = np.diff(object_df[input_ft], prepend=object_df[input_ft][0])
        #axes[ft_idx].plot(object_df['TimeIndex'], diff_vals, label=input_ft)
        if wraparound_offset:
            diff_vals[diff_vals > wraparound_offset[0]] -= 360
            diff_vals[diff_vals < wraparound_offset[1]] += 360
        #axes[ft_idx].plot(object_df['TimeIndex'], diff_vals, label=input_ft)
        #axes[ft_idx].plot(object_df['TimeIndex'], np.sin(np.deg2rad(object_df[input_ft]))*100.0, label=input_ft)

        axes[ft_idx].text(-50.0,np.min(object_df[input_ft]),'NS',rotation=0.0)
        axes[ft_idx].text(-50.0,np.max(object_df[input_ft]),'EW',rotation=0.0)
        if labels is not None:
            for index, row in labels.iterrows():
                if row['Direction'] == 'EW':
                    axes[ft_idx].axvline(row['TimeIndex'],color='g', linestyle="--")
                    axes[ft_idx].axvspan(row['TimeIndex']-6, row['TimeIndex']+6, alpha=0.2, color='g')
                    label=row['Node'] + '-' + row['Type']
                    #axes[ft_idx].text(row['TimeIndex'] + 1.0,np.max(object_df[input_ft]),label,rotation=0.0)
                elif row['Direction'] == 'NS':
                    axes[ft_idx].axvline(row['TimeIndex'],color='g', linestyle="-.")
                    axes[ft_idx].axvspan(row['TimeIndex']-6, row['TimeIndex']+6, alpha=0.2, color='g')
                    label=row['Node'] + '-' + row['Type']
                    #axes[ft_idx].text(row['TimeIndex'] + 1.0,np.min(object_df[input_ft]),label,rotation=0.0)
        axes[ft_idx].axvline(0,color='black', linestyle="--")
        axes[ft_idx].axvline(np.max(object_df['TimeIndex']),color='black', linestyle="--")
        axes[ft_idx].title.set_text(input_ft)

    plt.show()