import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import custom module functions
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
print(module_path)
from Code.ForecastingModel import models_calibration, data_loading_updating

# ========================
# Configure the parameters
# ========================
model_cat_id = "schedule_pred_ungrouped_newexo_lag"
features = ['all']  # Which features from the dataset should be loaded:
group_up = False  # If group up the 2 product schedules
# LSTM Layer configuration
# Size of how many samples are used for one forward/backward pass
batch_size = [5, 10, 15, 20]
# timesteps indicates how many subsamples you want to input in one training sample
timesteps = [96 * 1]  # 7 days sample to train
# forecasting_length indicates how many steps you want to forecast in the future
forecast_length = 96
# forecasting scheme
forecast_scheme = 'quarterly2quarterly'
# If adding new exogenous variable
new_exo = True
mpl.rcParams['figure.figsize'] = (9, 5)
# If shift exogenous variables back 2 hours, because those schedule are lagged
exo_lag = True


# ========================
# Configure paths/folders and load the data
# ========================

# Generate output folders and files
code_path = os.getcwd()
res_dir = code_path + '/' + model_cat_id + forecast_scheme + '/Model_Output/results/'
plot_dir = code_path + '/' + model_cat_id + forecast_scheme + '/Model_Output/plots/'
model_dir = code_path + '/' + model_cat_id + forecast_scheme + '/Model_Output/models/'
os.makedirs(res_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
output_table = res_dir + 'results.csv'
test_output_table = res_dir + 'test_results.csv'

# Load data and prepare for standardization
# Split date for train and test data.
split_date = dt.datetime(2017, 1, 2, 0, 0, 0, 0)
path = os.path.join(os.path.abspath(''), 'data/last_anonym_2017_vartime.csv')
X_train, y_train, X_test, y_test, y_scaler = data_loading_updating.load_dataset(
    path=path,
    modules=features,
    forecasting_interval=int(timesteps[0] / 96),
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
results_fn = output_table

results_csv = pd.read_csv(results_fn, delimiter=';')
top_models = results_csv.nsmallest(selection, 'valid_mae')

## Evaluate top 5 models
# Init test results table
test_results = pd.DataFrame(columns=['Model name', 'MSE', 'MAE', 'MAPE', 'MASE', 'nrRMSE', 'nmRMSE', 'niqrRMSE'])

# Init empty predictions
predictions = {}

# Loop through models
i = 0
for index, config in top_models.iterrows():
    print(index)
    print(config)
    # if i == 0:
    #     break
    # else:
    #     i += 1


    filename = model_dir + config['model_name'] + '.h5'
    model = load_model(filename)
    batch_size = int(config['batch_train'])

    # Load model and generate predictions
    model.reset_states()
    predictions = models_calibration.get_predictions(model=model, X=X_test, batch_size=batch_size,
                                                     timesteps=timesteps[0], verbose=0)

    # Otherwise, we get a memory leak!
    K.clear_session()
    import tensorflow as tf

    tf.reset_default_graph()

    # Get the scaling params from the standarization and rescale
    min = y_scaler.data_min_[0]
    max = y_scaler.data_max_[0]

    # predictions_raw = np.reshape(np.round(predictions * (max-min) + min), (-1,1))
    predictions_raw = np.reshape(predictions, (-1, 1))
    size = int(y_test.shape[0] / batch_size)
    # actual_raw = np.reshape(np.round(y_test * (max-min) + min)[0:size * batch_size], (-1,1))
    actual_raw = np.reshape(y_test[0:size * batch_size], (-1, 1))

    # Calculate benchmark and model MAE and MSE
    mse = mean_squared_error(actual_raw, predictions_raw)
    mae = mean_absolute_error(actual_raw, predictions_raw)
    mape = mean_absolute_percentage_error(y_true=actual_raw, y_pred=predictions_raw)
    mase = mean_absolute_scale_error(y_true=actual_raw, y_pred=predictions_raw, period=forecast_length)
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

    """
schedule_pred_59_l-20_l-20_l-20_d-0.2
grouped / daily forecasting

===============
mse 0.038502115933534746
mae 0.13717127402713447
mape 0.9021977851001907
mase 1.2692555783888013
nrRMSE 0.20751954805286701
nmRMSE 0.27324226333091456
niqrRMSE 1.3773689261377218
===============


schedule_pred_29_l-20_l-10_l-50_d-0.1
un_grouped / quarterly hour forcasting

===============
mse 0.04299788930442022
mae 0.12266572966509628
mape inf
mase 14.672076730733995
nrRMSE 0.20735932413185626
nmRMSE 0.2887547546039288
niqrRMSE 1.7026764788790814
===============

schedule_pred_21_l-20_l-10_l-20_d-0.2
un_grouped / daily forecasting
===============
mse 0.02356964135159845
mae 0.09604060132482677
mape 0.40182463159320264
mase 0.8886705314061434
nrRMSE 0.16236529325934768
nmRMSE 0.21378737874501125
niqrRMSE 1.0776667148566215
===============


ungrouped/ quarterly hour forecasting
use T+1 exogenous variables as input
===============
mse 0.020068406313788156
mae 0.08338969285256921
mape inf
mase 9.974260744428395
nrRMSE 0.1416630026287321
nmRMSE 0.19727044217458925
niqrRMSE 1.163228436980961
===============

upgrouped / daily forecasting
===============
mse 0.0062374469112718035
mae 0.05520214493259663
mape 0.21054694497243973
mase 0.5107893827745988
nrRMSE 0.08352570410514959
nmRMSE 0.10997880753954362
niqrRMSE 0.5543849263728076
===============

grouped / quarterly forecasting

===============
mse 0.021264922402955574
mae 0.08833286109774215
mape inf
mase 10.565514259033645
nrRMSE 0.14582497180851972
nmRMSE 0.2030661226640496
niqrRMSE 1.1974033507829458
===============

ungrouped/ daily forecasting/ shift exogenous variables 2 hours 

schedule_pred_ungrouped_newexo_lag_6_l-20_l-10_l-10_d-0.1

===============
mse 0.0062899010357888585
mae 0.051375738573383345
mape 0.12137987096628083
mase 0.47538337192386404
nrRMSE 0.08387617563229145
nmRMSE 0.11044027555163127
niqrRMSE 0.5567111100769966
===============

ungrouped/ quarterlyhour forecasting/ shift exogenous variables 2 hours

schedule_pred_ungrouped_newexo_lag_85_l-20_l-50_l-20

===============
mse 0.020250457431730068
mae 0.07963004269791378
mape inf
mase 0.5989887039666191
nrRMSE 0.14230410194976836
nmRMSE 0.1981631942989396
niqrRMSE 1.1684926552124
===============

"""

    # plt.plot(actual_raw)
    # plt.plot(predictions_raw)

    # Store results
    mod_name = config['model_name']
    result = [{'Model name': mod_name,
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
        filename = plot_dir + mod_name + 'top_model_predictions'
        # plt.tight_layout()
        plt.savefig(filename + '.png', dpi=300)
        plt.close()

test_results = test_results.sort_values('MAE', ascending=True)

# if not os.path.isfile(test_output_table):
#     test_results.to_csv(test_output_table, sep=';')
# else:  # else it exists so append without writing the header
#     test_results.to_csv(test_output_table, mode='a', header=False, sep=';')
#
# print(tabulate(test_results, headers='keys', tablefmt="grid", numalign="right", floatfmt=".3f"))
