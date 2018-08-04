# coding: utf-8

# ### Module imports
import matplotlib as mpl
mpl.use('TkAgg')
import os
import sys
import datetime as dt
import time as t
import numpy as np
import pandas as pd
from pandas import datetime
from Code.ForecastingModel import models_calibration
from Code.ForecastingModel import data_loading_updating
from Code.ForecastingModel.LSTM_Params import *

from tabulate import tabulate
from keras import backend as K

mpl.rcParams['figure.figsize'] = (9, 5)

# Import custom module functions
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(module_path)

# Model category name used throughout the subsequent analysis

# ## Overall configuration
# These parameters are later used, but shouldn't have to change between different model categories (model 1-5)


# Directory with dataset
path = os.path.join(os.path.abspath(''), 'data/last_anonym_2017_vartime.csv')
# path = os.path.join(os.path.abspath(''), 'Source/lstm_load_forecasting/data/fulldataset.csv')


# Dataframe containing the relevant data from training of all models
results = pd.DataFrame(columns=['model_name', 'config', 'dropout',
                                'train_loss', 'train_rmse', 'train_mae', 'train_mape',
                                'valid_loss', 'valid_rmse', 'valid_mae', 'valid_mape',
                                'test_rmse', 'test_mae', 'test_mape',
                                'epochs', 'batch_train', 'input_shape',
                                'total_time', 'time_step', 'splits'
                                ])

# ## Preparation and model generation
# Necessary preliminary steps and then the generation of all possible models based on the settings at the top of this notebook.


# Generate output folders and files
code_path = os.getcwd()
res_dir = code_path + '/Output/LSTM/' + model_cat_id + forecast_scheme + '/Model_Output/results/'
plot_dir = code_path + '/Output/LSTM/' + model_cat_id + forecast_scheme + '/Model_Output/plots/'
model_dir = code_path + '/Output/LSTM/' + model_cat_id + forecast_scheme + '/Model_Output/models/'
os.makedirs(res_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
output_table = res_dir + 'results.csv'
test_output_table = res_dir + 'test_results.csv'

# Generate model combinations
# models = []
models = models_calibration.generate_combinations(
    model_name=model_cat_id + '_', layer_conf=layer_conf, cells=cells, dropout=dropout,
    batch_size=batch_size, timesteps=timesteps)

# ## Loading the data:

# Load data and prepare for standardization
X_train, y_train, X_test, y_test, y_scaler = data_loading_updating.load_dataset(path=path,
                                                                                modules=features,
                                                                                forecasting_interval=int(
                                                                                    timesteps[0] / 96),
                                                                                split_date=split_date,
                                                                                forecast_scheme=forecast_scheme,
                                                                                grouped_up=group_up,
                                                                                transform='0-1',
                                                                                new_exo=new_exo,
                                                                                exo_lag=exo_lag)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# ## Running through all generated models
# Note: Depending on the above settings, this can take very long!


start_time = t.time()
for idx, m in enumerate(models):
    print(idx, m)
    # if idx == 100:
    #     break

    stopper = t.time()
    print('========================= Model {}/{} ========================='.format(idx + 1, len(models)))
    print(tabulate([['Starting with model', m['name']], ['Starting time', datetime.fromtimestamp(stopper)]],
                   tablefmt="jira", numalign="right", floatfmt=".3f"))
    try:
        # Creating the Keras Model
        # In the training dataset, we have 240 samples, each sample with 7 sub-samples (last 7 days samples), and each
        # sample has 72 features
        model = models_calibration.create_models(layers=m['layers'], sample_size=X_train.shape[0],
                                                 batch_size=m['batch_size'],
                                                 timesteps=m['timesteps'], features=X_train.shape[2],
                                                 forecast_length=forecast_length)
        # Training...
        history = models_calibration.train_models(model=model, mode='fit', y=y_train, X=X_train,
                                                  batch_size=m['batch_size'], timesteps=m['timesteps'], epochs=epochs,
                                                  rearrange=False, validation_split=validation_split, verbose=verbose,
                                                  early_stopping=early_stopping, min_delta=min_delta, patience=patience)

        # Write results
        min_loss = np.min(history.history['val_loss'])
        min_idx = np.argmin(history.history['val_loss'])
        min_epoch = min_idx + 1

        if verbose > 0:
            print('______________________________________________________________________')
            print(tabulate([['Minimum validation loss at epoch', min_epoch, 'Time: {}'.format(t.time() - stopper)],
                            ['Training loss & MAE', history.history['loss'][min_idx],
                             history.history['mean_absolute_error'][min_idx]],
                            ['Validation loss & mae', history.history['val_loss'][min_idx],
                             history.history['val_mean_absolute_error'][min_idx]],
                            ], tablefmt="jira", numalign="right", floatfmt=".3f"))
            print('______________________________________________________________________')

        result = [
            {'model_name': m['name'], 'config': m, 'train_loss': history.history['loss'][min_idx], 'train_rmse': 0,
             'train_mae': history.history['mean_absolute_error'][min_idx], 'train_mape': 0,
             'valid_loss': history.history['val_loss'][min_idx], 'valid_rmse': 0,
             'valid_mae': history.history['val_mean_absolute_error'][min_idx], 'valid_mape': 0,
             'test_rmse': 0, 'test_mae': 0, 'test_mape': 0, 'epochs': '{}/{}'.format(min_epoch, epochs),
             'batch_train': m['batch_size'],
             'input_shape': (X_train.shape[0], timesteps, X_train.shape[1]), 'total_time': t.time() - stopper,
             'time_step': 0, 'splits': str(split_date), 'dropout': m['layers'][0]['dropout']
             }]
        results = results.append(result, ignore_index=True)

        # Saving the model and weights
        model.save(model_dir + m['name'] + '.h5')

        # Write results to csv
        results.to_csv(output_table, sep=';')

        K.clear_session()
        import tensorflow as tf

        tf.reset_default_graph()

    # Shouldn't catch all errors, but for now...
    except BaseException as e:
        print('=============== ERROR {}/{} ============='.format(idx + 1, len(models)))
        print(tabulate([['Model:', m['name']], ['Config:', m]], tablefmt="jira", numalign="right", floatfmt=".3f"))
        print('Error: {}'.format(e))
        result = [{'model_name': m['name'], 'config': m, 'train_loss': str(e)}]
        results = results.append(result, ignore_index=True)
        results.to_csv(output_table, sep=';')
        continue
