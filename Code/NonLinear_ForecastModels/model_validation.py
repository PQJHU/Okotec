import matplotlib as mpl

mpl.use('TkAgg')
import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import custom module functions
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(module_path)
from Code.NonLinear_ForecastModels import CNF_ConfigurationModel, OPR_PreprocessAndLoad
from Code.NonLinear_ForecastModels.LSTM_Params import *

# Load data and prepare for standardization
# Split date for train and test data
split_date = dt.datetime(2017, 1, 2, 0, 0, 0, 0)
path = os.path.join(os.path.abspath(''), 'data/last_anonym_2017_vartime.csv')


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



def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def nr_root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / ((np.max(y_true) - np.min(y_true)))


def nm_root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2)) / (np.mean(y_true))


def niqr_root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2)) / ((np.percentile(y_true, 75) - np.percentile(y_true, 25)))


def mean_absolute_scale_error(y_true, y_pred, period=96):
    y_true_t = np.reshape(y_true, [-1, period])
    diff = y_true_t[1:] - y_true_t[:-1]
    qt = np.abs(y_true - y_pred) / np.mean(np.abs(diff))
    return np.mean(qt)


## Model selection based on the validation MAE

# Select the top 5 models based on the Mean Absolute Error in the validation data:
# http://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error
#
# Number of the selected top models
selection = 5
# If run in the same instance not necessary. If run on the same day, then just use output_table
# reset redir
results_fn = results_file

results_csv = pd.read_csv(results_fn, delimiter=';')
top_models = results_csv.nsmallest(selection, 'valid_mae')

## Evaluate top 5 models
# Init test results table
test_results = pd.DataFrame(columns=['Model name', 'MSE', 'MAE', 'MAPE', 'MASE', 'nrRMSE', 'nmRMSE', 'niqrRMSE'])

# Init empty predictions
predictions = pd.DataFrame()

# Loop through models
i = 0
for index, config in top_models.iterrows():
    print(index)
    print(config)
    if i == 0:
        break
    else:
        i += 1

    filename = model_dir + config['model_name'] + '.h5'
    model_name = config['model_name']
    model = load_model(filename)
    batch_size = int(config['batch_train'])
    length = int(y_test.shape[0] / batch_size) * batch_size

    # Load model and generate predictions
    model.reset_states()
    forecast_model_result = CNF_ConfigurationModel.get_predictions(model=model, X=X_test, batch_size=batch_size,
                                                                   lagged_days=lagged_days[0], verbose=0)
    forecast_result_df = pd.DataFrame(data=forecast_model_result.reshape(-1,1), index=time_frame_test[0:length*96], columns=[model_name])
    predictions = pd.concat([predictions, forecast_result_df], axis=1)
    # Otherwise, we get a memory leak!
    K.clear_session()

    tf.reset_default_graph()

    # predictions_raw = np.reshape(np.round(predictions * (max-min) + min), (-1,1))
    predictions_raw = np.reshape(forecast_model_result, (-1, 1))
    # actual_raw = np.reshape(np.round(y_test * (max-min) + min)[0:size * batch_size], (-1,1))
    actual_raw = np.reshape(y_test[0:length], (-1, 1))

    # Calculate benchmark and model MAE and MSE
    mse = mean_squared_error(actual_raw, predictions_raw)
    mae = mean_absolute_error(actual_raw, predictions_raw)
    mape = mean_absolute_percentage_error(y_true=actual_raw, y_pred=predictions_raw)
    mase = mean_absolute_scale_error(y_true=actual_raw, y_pred=predictions_raw, period=horizon)
    nrRMSE_result = nr_root_mean_square_error(y_true=actual_raw, y_pred=predictions_raw)
    nmRMSE_result = nm_root_mean_square_error(y_true=actual_raw, y_pred=predictions_raw)
    niqrRMSE_result = niqr_root_mean_square_error(y_true=actual_raw, y_pred=predictions_raw)

    print('===============')
    print('mse', mse)
    print('mae', mae)
    print('mape', mape)
    print('mase', mase)
    print('nrRMSE', nrRMSE_result)
    print('nmRMSE', nmRMSE_result)
    print('niqrRMSE', niqrRMSE_result)
    print('===============')

    # plt.plot(actual_raw)
    # plt.plot(predictions_raw)

    # Store results
    result = [{'Model name': model_name,
               'MSE': mse,
               'MAE': mae,
               'MAPE': mape,
               'MASE': mase,
               'nrRMSE': nrRMSE_result,
               'nmRMSE': nmRMSE_result,
               'niqrRMSE': niqrRMSE_result
               }]
    test_results = test_results.append(result, ignore_index=True)

    graph = True
    if graph:
        plt.figure(figsize=(12, 5))
        plt.plot(actual_raw, color='black', linewidth=0.5)
        plt.plot(predictions_raw, color='blue', linewidth=0.5)
        # plt.title('LSTM Model: ${}$'.format(mod_name))
        plt.ylabel('Electricity load')
        plt.show()
        filename = plot_dir + model_name + 'top_model_predictions'
        # plt.tight_layout()
        plt.savefig(filename + '.png', dpi=300)
        plt.close()

test_results = test_results.sort_values('MAE', ascending=True)

best_prediction = predictions[test_results.head(1)['Model name'].values[0]].to_frame()


# Transform back to real consumption

predictions_output = pd.DataFrame(index=best_prediction.index)
predictions_output['true_load'] = y_test[0:length].reshape(-1,1)
predictions_output['best_predict'] = best_prediction.values
predictions_output['errors'] = predictions_output['true_load'] - predictions_output['best_predict']
raw_true_load = y_scaler.inverse_transform(y_test[0:length].reshape(-1, 1))
raw_best_pred = y_scaler.inverse_transform(best_prediction.values.reshape(-1,1))
predictions_output['raw_true_load'] = raw_true_load
predictions_output['raw_best_pred'] = raw_best_pred

# =============== Test for accuracy of inverse tranform:

# src = pd.read_csv('data/last_anonym_2017_vartime.csv', sep=';', index_col=0, parse_dates=True)
#
# predictions_output['raw_load_prev_trans'] = src['l']
#
# plt.plot(predictions_output['raw_true_load'])
# plt.plot(predictions_output['raw_load_prev_trans'])

# =============== Finish test

# ============ Forecast errors plot
# raw_load_before_transform = y_test
# plt.scatter(x=predictions_output['true_load'],y=predictions_output['best_predict'])
# error = predictions_output.values.reshape(1,-1)[0]
# plt.figure(figsize=(15,5))
# plt.hist(error, bins=500)
# plt.plot(predictions_output['errors'])
# predictions_output = pd.concat([predictions_output, fastec_prediction['forecast_fastec']], axis=1)
# ===========Finish


predictions_output.to_csv(res_dir + 'Best_model_forecaste_result.csv')

# fastec_prediction = pd.read_csv(fastec_dir + 'actual_forecast.csv', index_col=0, parse_dates=True)
# fastec_prediction['forecast_fastec'] = y_scaler.inverse_transform(fastec_prediction['Yhat'].values.reshape(-1,1))
# fastec_prediction['errors'] = fastec_prediction['Y'] - fastec_prediction['Yhat']
#
# plt.figure(figsize=(15,5))
# plt.hist(fastec_prediction['errors'], bins=500)
# plt.plot(fastec_prediction['errors'])



# if not os.path.isfile(test_output_table):
#     test_results.to_csv(test_output_table, sep=';')
# else:  # else it exists so append without writing the header
#     test_results.to_csv(test_output_table, mode='a', header=False, sep=';')
#
# print(tabulate(test_results, headers='keys', tablefmt="grid", numalign="right", floatfmt=".3f"))
