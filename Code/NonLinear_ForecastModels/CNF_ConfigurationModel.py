"""
Configurate hyperparameters for models

"""

# module imports
import os
import sys
import math
from decimal import *
import itertools
import time as t
import numpy as np
import pandas as pd
from numpy import newaxis
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa import stattools

import math
# import keras as keras
import keras.layers as kl
import keras.models as km
import keras.callbacks as kc

from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from Code.NonLinear_ForecastModels.LSTM_Params import log_dir
from tabulate import tabulate


def hyperparameters_configuration(model_type=None,
                                  stateful=None,
                                  neurons=None,
                                  dropout=None,
                                  batch_sizes=None,
                                  lagged_days=None,
                                  cell_type=None):
    """
    Given different hyperparameters, configurate all combination
    :param model_name:
    :param stateful:
    :param neurons:
    :param dropout:
    :param batch_sizes:
    :param lagged_days:
    :return:
    """

    models_hyperparams = []

    layer_comb = list(itertools.product(*neurons))
    configs = list(itertools.product(*[layer_comb, dropout, batch_sizes, [lagged_days]]))

    # Format: (
    # (num_nuerons_layer1, num_nuerons_layer1,..),
    # (dropout_rate),
    # (batchsize for fitting),
    # (lagged days))

    for struc_id, struc in enumerate(configs):
        # if struc_id == 0:
        #     break
        model_name = model_type + '_' + str(struc_id + 1)
        # Now the list of layers needs to be generated
        layers = []
        for layer_id, nuro_num in enumerate(struc[0]):
            # if layer_id == 100:
            #     break
            # else:
            #     pass
            #     print(layer_id, nuro_num)

            return_sequence = True

            if (layer_id + 1) == len(struc[0]):  # the last layer doesn't return sequence
                return_sequence = False

            if nuro_num > 0:
                layers.append({'cell_type': cell_type,
                               'neurons': nuro_num,
                               'dropout': struc[1],
                               'stateful': stateful[layer_id],
                               'ret_seq': return_sequence})
                model_name += '_l-' + str(nuro_num)

        # Add dropout identifier to name
        if struc[1] > 0:
            model_name += '_d' + str(struc[1])
        # Add model config
        model_config = {
            'name': model_name,
            'layers': layers,
            'batch_size': struc[2],
            'lagged_days': struc[3]
        }
        models_hyperparams.append(model_config)

    print('==================================')
    print(tabulate(
        [['Number of model configs generated', len(configs)]], tablefmt="jira", numalign="right", floatfmt=".3f"))
    print('==================================')
    return models_hyperparams


def model_setting(model_hyperparams, features, loss, optimizer, horizon, sample_perday, cell_type, metrics):
    """

    :param layers:
    :param batch_size:
    :param lagged_days:
    :param features:
    :param loss:
    :param optimizer:
    :param horizon:
    :param sample_perday:
    :param cell_type:
    :param metrics:
    :return:
    """

    layers = model_hyperparams['layers']
    batch_size = model_hyperparams['batch_size']
    lagged_days = model_hyperparams['lagged_days']

    model = km.Sequential()
    # For all configured layers
    # model.add(kl.Dense(units=horizon * sample_perday))

    for layer_id, l in enumerate(layers):

        # if layer_id == 0:
        #     break

        if cell_type == 'lstm':
            if layer_id == 0:

                # First LSTM layer indicated with input size and batch size

                model.add(kl.LSTM(
                    l["neurons"],
                    input_shape=(lagged_days * sample_perday, features),
                    batch_input_shape=(batch_size, lagged_days * sample_perday, features),
                    return_sequences=l["ret_seq"],
                    stateful=l['stateful']
                ))

            # Only add additional layers, if cell is > 0

            elif layer_id > 0 and l["neurons"] > 0:
                model.add(kl.LSTM(
                    l["neurons"],
                    batch_input_shape=(batch_size, lagged_days * sample_perday, features),
                    return_sequences=l["ret_seq"],
                    stateful=l['stateful']))

        # TODO: Finish GRU model later
        # if cell_type == 'gru':
        #     if layer_id == 0:
        #         model.add(kl.GRU(l['neurons'],
        #                       input_shape=(lagged_days * sample_perday, features),
        #                       batch_input_shape=(batch_size, lagged_days * sample_perday, features),
        #                       return_sequences=l['ret_seq'],
        #                       stateful=l['stateful']
        #                       ))

        # Add dropout.
        if l['dropout'] > 0:
            model.add(kl.Dropout(l['dropout']))
    model.add(kl.Dense(horizon * sample_perday))
    # model.add(Activation('relu'))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def model_fitting(model, model_hyperparams, y_train, X_train, batch_size, epochs, verbose, validation_split,
                  early_stopping, min_delta, patience, tensor_board):
    """

    :param model:
    :param y_train:
    :param X_train:
    :param mode:
    :param batch_size:
    :param epochs:
    :param verbose:
    :param validation_split:
    :param early_stopping:
    :param min_delta:
    :param patience:
    :return:
    """
    # Set clock
    print(f'Computing Model{model_hyperparams["name"]}...')
    start_time = t.time()

    # Set callbacks
    # TODO: Add Keras callback such that log files will be written for TensorBoard integration
    # tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1,
    # write_graph=True,   write_images=True)
    # Set early stopping (check if model has converged) callbacks
    if early_stopping:
        early_stop = kc.EarlyStopping(monitor='mae',
                                      min_delta=min_delta,
                                      patience=patience,
                                      verbose=verbose,
                                      mode='auto')
    else:
        early_stop = kc.Callback()

    if tensor_board:
        """TensorBoard basic visualizations.

        [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
        is a visualization tool provided with TensorFlow.

        This callback writes a log for TensorBoard, which allows
        you to visualize dynamic graphs of your training and test
        metrics, as well as activation histograms for the different
        layers in your model.

        If you have installed TensorFlow with pip, you should be able
        to launch TensorBoard from the command line:
        ```sh
        tensorboard --logdir=/full_path_to_your_logs
        ```

        When using a backend other than TensorFlow, TensorBoard will still work
        (if you have TensorFlow installed), but the only feature available will
        be the display of the losses and metrics plots.

        # Arguments
            log_dir: the path of the directory where to save the log
                files to be parsed by TensorBoard.
            histogram_freq: frequency (in epochs) at which to compute activation
                and weight histograms for the layers of the model. If set to 0,
                histograms won't be computed. Validation data (or split) must be
                specified for histogram visualizations.
            write_graph: whether to visualize the graph in TensorBoard.
                The log file can become quite large when
                write_graph is set to True.
            write_grads: whether to visualize gradient histograms in TensorBoard.
                `histogram_freq` must be greater than 0.
            batch_size: size of batch of inputs to feed to the network
                for histograms computation.
            write_images: whether to write model weights to visualize as
                image in TensorBoard.
            embeddings_freq: frequency (in epochs) at which selected embedding
                layers will be saved. If set to 0, embeddings won't be computed.
                Data to be visualized in TensorBoard's Embedding tab must be passed
                as `embeddings_data`.
            embeddings_layer_names: a list of names of layers to keep eye on. If
                None or empty list all the embedding layer will be watched.
            embeddings_metadata: a dictionary which maps layer name to a file name
                in which metadata for this embedding layer is saved. See the
                [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
                about metadata files format. In case if the same metadata file is
                used for all embedding layers, string can be passed.
            embeddings_data: data to be embedded at layers specified in
                `embeddings_layer_names`. Numpy array (if the model has a single
                input) or list of Numpy arrays (if the model has multiple inputs).
                Learn [more about embeddings]
                (https://www.tensorflow.org/programmers_guide/embedding).
            update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
                the losses and metrics to TensorBoard after each batch. The same
                applies for `'epoch'`. If using an integer, let's say `10000`,
                the callback will write the metrics and losses to TensorBoard every
                10000 samples. Note that writing too frequently to TensorBoard
                can slow down your training.
        """

        tensor_board = kc.TensorBoard(log_dir=log_dir + f'{model_hyperparams["name"]}logs',
                                      write_graph=True)

    # Divide observations into N_batch, each batch has batch_size observations

    N_batch = int(X_train.shape[0] / batch_size)
    train_batches = int((1 - validation_split) * N_batch)
    valid_batches = N_batch - train_batches
    effective_split = valid_batches / N_batch  # effective_split is proportion of validation batches to all batches

    X_train = X_train[0:N_batch * batch_size]
    y_train = y_train[0:N_batch * batch_size]

    # Fitting the model. Keras function
    history = model.fit(
        shuffle=True,
        x=X_train,
        y=y_train,
        validation_split=effective_split,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=[early_stop]
    )

    stopper = t.time()
    run_time = round(stopper - start_time, 1)
    print(f'{run_time} seconds')

    return history


def plot_history(model_config=None, history=None, path=None, metrics='mean_absolute_error', interactive=False,
                 display=False):
    # Turn interactive plotting off
    if not interactive:
        plt.ioff()

    # summarize history for accuracy
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Training history: Mean absolute error ')
    plt.ylabel('Mean absolute error')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    filename = path + model_config['name'] + '_mae'
    plt.savefig(filename + '.pgf')
    plt.savefig(filename + '.pdf')

    if display:
        plt.show()
    elif not display:
        # Reset
        plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training history: Loss ')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')

    filename = path + model_config['name'] + '_loss'
    plt.savefig(filename + '.pgf')
    plt.savefig(filename + '.pdf')

    if display:
        plt.show()
    elif not display:
        # Reset
        plt.clf()


def evaluate_model(model=None, X=None, y=None, batch_size=1, lagged_days=None, verbose=0):
    max_batch_count = int(X.shape[0] / batch_size)

    if (max_batch_count * batch_size) < (X.shape[0]) and verbose > 0:
        print('Warnining: Division "sample_size/batch_size" not a natural number.')
        print('Dropped the last {} of {} number of obs.'.format(X.shape[0] - max_batch_count * batch_size, X.shape[0]))

    X = X[0:max_batch_count * batch_size]
    y = y[0:max_batch_count * batch_size]
    X = np.reshape(np.array(X), (X.shape[0], lagged_days, X.shape[2]))
    test_loss, test_mae = model.evaluate(X, y, batch_size=batch_size, verbose=verbose, sample_weight=None)

    return test_loss, test_mae


def get_predictions(model=None, X=None, batch_size=1, lagged_days=1, verbose=0):
    max_batch_count = int(X.shape[0] / batch_size)

    if (max_batch_count * batch_size) < (X.shape[0]) and verbose > 0:
        print('Warnining: Division "sample_size/batch_size" not a natural number.')
        print('Dropped the last {} of {} number of obs.'.format(X.shape[0] - max_batch_count * batch_size, X.shape[0]))

    X = X[0:max_batch_count * batch_size]
    X = np.reshape(np.array(X), (X.shape[0], lagged_days, X.shape[2]))
    predictions = model.predict(x=X, batch_size=batch_size, verbose=verbose)

    return predictions

# def plot_predictions(predictions=None):
#         # Turn interactive plotting off
#     if not interactive:
#         plt.ioff()
#
#     # summarize history for accuracy
#     plt.plot(history.history['mean_absolute_error'])
#     plt.plot(history.history['val_mean_absolute_error'])
#     plt.title('Training history: Mean absolute error ')
#     plt.ylabel('Mean absolute error')
#     plt.xlabel('Epoch')
#     plt.legend(['Training', 'Validation'], loc='upper right')
#
#     filename = path + model_config['name'] + '_mae'
#     plt.savefig(filename + '.pgf')
#     plt.savefig(filename + '.pdf')
#
#     if display:
#         plt.show()
#     elif not display:
#         # Reset
#         plt.clf()
#
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Training history: Loss ')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Test'], loc='upper right')
#
#     filename = path + model_config['name'] + '_loss'
#     plt.savefig(filename + '.pgf')
#     plt.savefig(filename + '.pdf')
#
#     if display:
#         plt.show()
#     elif not display:
#         # Reset
#         plt.clf()
