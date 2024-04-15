import pandas as pd
import numpy as np
import pickle
import gc
from pathlib import Path
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from base import datahandler, prediction_models, evaluation, utils, localizer

# Useful methods to perform SHAP feature importance analysis. https://github.com/shap/shap

def get_shap_values(model, X_t, X_v):
    #tf.compat.v1.disable_eager_execution() # sometimes needed for compateblity
    shap_values_dict = {}
    for sub_model_output_idx, sub_model_output in enumerate(model.output):
        sub_model = tf.keras.Model(model.input, model.layers[-(1+sub_model_output_idx)].output)
        explainer = shap.GradientExplainer(sub_model, X_t)
        shap_values = explainer.shap_values(X_v)
        shap_values = np.abs(shap_values) # we are interested in absolute contributions
        shap_values = np.mean(shap_values, axis=1) # mean over the number of X_v samples
        shap_values_dict[sub_model_output.name] = shap_values
    return shap_values_dict

def get_shap_values_from_ds(model, train_ds, val_ds, n_t, n_v):
    def get_x_from_xy(x,y):
        return x['local_in'] # x is a dict
    X_t = np.concatenate([element for element in train_ds.map(get_x_from_xy).as_numpy_iterator()], axis=0)
    X_v = np.concatenate([element for element in val_ds.map(get_x_from_xy).as_numpy_iterator()])
    return get_shap_values(model, X_t[:n_t], X_v[n_t:n_t+n_v]) # take later elements for X_v, in case that train and val ds are identical

def plot_ft_importance_bars(shap_values, feature_names, title_prefix=''):
    for shap_values_name, shap_values in shap_values.items():
        n_classes = shap_values.shape[0]
        per_class_series = []
        for class_id in range(n_classes):
            df = pd.DataFrame(shap_values[class_id,:,:], columns=feature_names)
            df['TIME'] = range(-shap_values.shape[1], 0)

            scaler=MinMaxScaler((0.1,1.0))
            
            sum_of_columns = df[feature_names].sum()
            scaled_data = scaler.fit_transform(sum_of_columns.values.reshape(-1,1))
            sum_of_columns.update(pd.Series(scaled_data[:,0], index=sum_of_columns.index))
            sum_of_columns.sort_values(inplace=True)
            per_class_series.append(sum_of_columns)
        
        combined_df = pd.concat(per_class_series, axis=1)
        fig, ax  = plt.subplots(figsize=(11,6))
        combined_df.plot(kind='barh',
                         ax=ax,
                         position=0.8,
                         label='Feature importance sum')

        ax.set_title(title_prefix + f' Summed feature importance (output {shap_values_name[:2]})')
        ax.legend()
        plt.show()

def plot_ft_importance_over_time(shap_values, feature_names, title_prefix='', savefile=None, time_hist=None, time_fut=None, time_stride=None, sum_over_classes=False):
    for shap_values_name, shap_values in shap_values.items():
        n_classes = shap_values.shape[0]
        if sum_over_classes:
            shap_values[0,:,:] = np.sum(shap_values, axis=0)
            n_classes=1
        for class_id in range(n_classes):
            df = pd.DataFrame(shap_values[class_id,:,:], columns=feature_names)
            df['TIME'] = range(-shap_values.shape[1], 0)
            if time_hist is not None and time_fut is not None and time_stride is not None:
                df['TIME'] = range(-time_hist, time_fut, time_stride)

            # Order features according to summed importance

            df.plot.area(x='TIME',
                        y = df[feature_names].sum().sort_values(inplace=False).index.to_list()[::-1],
                        figsize=(11, 6),
                        cmap='plasma',
                        xlabel='Relative Timestep',
                        ylabel='Importance')
            plt.title(title_prefix + f' Feature Importance over time ({shap_values_name[:2]}' + (f' class {class_id})' if n_classes > 1 else (' - sum over all classes)' if sum_over_classes else ')')))
            if savefile is not None:
                plt.savefig(savefile + '_' + shap_values_name[:2] + '.png', dpi=1200, bbox_inches="tight")
            plt.show()
