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
from Code.ForecastingModel import CNF_ConfigurationModel
from Code.ForecastingModel import OPR_PreprocessAndLoad
from Code.ForecastingModel.LSTM_Params import *

from tabulate import tabulate
from keras import backend as K


# ## Overall configuration
# These parameters are later used, but shouldn't have to change between different model categories (model 1-5)


class LSTM:

    """
    Calibrate LSTM model from given hyperparamteres
    """

    def __init__(self, model_name, hyperparams):

        self.model_name = model_name

        # Hyperparameters setting
        self.hyperparams = hyperparams

    def data_preprocessing(self):

        pass

# Directory with dataset
file_path = os.path.join(os.path.abspath(''), 'data/last_anonym_2017_vartime.csv')
# path = os.path.join(os.path.abspath(''), 'Source/lstm_load_forecasting/data/fulldataset.csv')


# Dataframe containing the relevant data from training of all models
results = pd.DataFrame(columns=['model_name', 'config', 'dropout',
                                'train_loss', 'train_rmse', 'train_mae', 'train_mape',
                                'valid_loss', 'valid_rmse', 'valid_mae', 'valid_mape',
                                'test_rmse', 'test_mae', 'test_mape',
                                'epochs', 'batch_train', 'input_shape',
                                'total_time', 'time_step', 'splits'
                                ])

# Preparation and model generation
# Generate model combinations
models_hyperparams = CNF_ConfigurationModel.hyperparameters_configuration(model_type=model_type,
                                                                          stateful=stateful,
                                                                          neurons=neurons,
                                                                          dropout=dropout,
                                                                          batch_sizes=batch_sizes,
                                                                          lagged_days=lagged_days,
                                                                          cell_type=cell_type)

# Loading the data:
X_train, y_train, time_frame_train, X_test, y_test, time_frame_test, y_scaler = OPR_PreprocessAndLoad.PreprocessingData(
    file_path=file_path,
    lagged_days=lagged_days,
    sample_perday=sample_perday,
    horizon=horizon,
    split_date=split_date,
    forecast_scheme=forecast_scheme,
    group_up=group_up,
    transform=transform,
    new_exo=new_exo,
    exo_lag=exo_lag)

# ## Running through all generated models
# Note: Depending on the above settings, this can take very long!


# test for creating model
# import keras.models as kmodel
# import keras.layers as klayer
# import keras.callbacks as kc
#
#
# batch_size = [20]
# N_batch = int(X_train.shape[0] / batch_size[0])
#
# X_train = X_train[0:N_batch * batch_size[0]]
# y_train = y_train[0:N_batch * batch_size[0]]
#
# model = kmodel.Sequential()
# model.add(klayer.LSTM(units=50,
#                       input_shape=(lagged_days*sample_perday, 4),
#                       batch_input_shape=(batch_size[0], lagged_days * sample_perday, 4),
#                       return_sequences=True,
#                       stateful=True
#                       ))
# model.add(klayer.LSTM(50, return_sequences=True, stateful=True))
# model.add(klayer.LSTM(50, return_sequences=False, stateful=True))
# model.add(klayer.Dropout(0.3))
# model.add(klayer.Dense(horizon * sample_perday))
#
# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#
# early_stop = kc.EarlyStopping(monitor='loss',
#                               min_delta=min_delta,
#                               patience=4,
#                               verbose=2,
#                               mode='auto')
#
# history = model.fit(X_train, y_train,
#                     epochs=50,
#                     batch_size=batch_size[0],
#                     validation_split=0.3,
#                     verbose=2,
#                     callbacks=[early_stop],
#                     )
#


def model_training(model, X_train, y_train, ):
    pass




start_time = t.time()
for model_id, model_hyperparams in enumerate(models_hyperparams):
    print(model_id, model_hyperparams)
    # if model_id == 100:
    #     break

    stopper = t.time()
    print(f'========================= Model {model_id + 1}/{len(models_hyperparams)} =========================')
    print(tabulate([['Starting with model', model_hyperparams['name']], ['Starting time', datetime.fromtimestamp(stopper)]],
                   tablefmt="jira", numalign="right", floatfmt=".3f"))

    # Model training params
    features = X_train.shape[2]
    loss = 'mse'
    optimizer = 'adam'
    metrics = ['mae']


    try:
        # Creating the Keras Model
        # In the training dataset, we have 240 samples, each sample with 7 sub-samples (last 7 days samples), and each
        # sample has 72 features

        model = CNF_ConfigurationModel.model_setting(model_hyperparams=model_hyperparams,
                                                     features=features,
                                                     loss=loss,
                                                     optimizer=optimizer,
                                                     horizon=horizon,
                                                     sample_perday=sample_perday,
                                                     cell_type=cell_type,
                                                     metrics=metrics)

        model.summary()

        # Training...
        history = CNF_ConfigurationModel.model_fitting(model=model,
                                                       model_hyperparams =model_hyperparams,
                                                       y_train=y_train,
                                                       X_train=X_train,
                                                       batch_size=model_hyperparams['batch_size'],
                                                       epochs=epochs,
                                                       validation_split=validation_split,
                                                       verbose=verbose,
                                                       early_stopping=early_stopping,
                                                       min_delta=min_delta,
                                                       patience=patience,
                                                       tensor_board=tensor_board)

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
            {'model_name': model_hyperparams['name'], 'config': model_hyperparams, 'train_loss': history.history['loss'][min_idx], 'train_rmse': 0,
             'train_mae': history.history['mean_absolute_error'][min_idx], 'train_mape': 0,
             'valid_loss': history.history['val_loss'][min_idx], 'valid_rmse': 0,
             'valid_mae': history.history['val_mean_absolute_error'][min_idx], 'valid_mape': 0,
             'test_rmse': 0, 'test_mae': 0, 'test_mape': 0, 'epochs': '{}/{}'.format(min_epoch, epochs),
             'batch_train': model_hyperparams['batch_size'],
             'input_shape': (X_train.shape[0], lagged_days, X_train.shape[1]), 'total_time': t.time() - stopper,
             'time_step': 0, 'splits': str(split_date), 'dropout': model_hyperparams['layers'][0]['dropout']
             }]
        results = results.append(result, ignore_index=True)

        # Saving the model and weights
        model.save(model_dir + model_hyperparams['name'] + '.h5')

        # Write results to csv
        results.to_csv(results_file, sep=';')

        K.clear_session()
        import tensorflow as tf

        tf.reset_default_graph()

    # Shouldn't catch all errors, but for now...
    except BaseException as e:
        print('=============== ERROR {}/{} ============='.format(model_id + 1, len(models_hyperparams)))
        print(tabulate([['Model:', model_hyperparams['name']], ['Config:', model_hyperparams]], tablefmt="jira", numalign="right", floatfmt=".3f"))
        print('Error: {}'.format(e))
        result = [{'model_name': model_hyperparams['name'], 'config': model_hyperparams, 'train_loss': str(e)}]
        results = results.append(result, ignore_index=True)
        results.to_csv(results_file, sep=';')
        continue
