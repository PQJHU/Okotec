import datetime as dt
import os
import random

random.seed(1124)

# Data Preparation
# ================

# If group up the 3 product schedules
group_up = True
# If adding new exogenous variable
new_exo = False
# If shift exogenous variables back 2 hours, because those schedule are lagged
exo_lag = False
# Method to transform data
transform = '0-1'

# LSTM Layer configuration
# ========================

# Sample Number per day
sample_perday = 96

# Type of cell
cell_type = 'lstm'

# Stateful True or false
stateful = [True, True, True, True]

# Number of LSTM layers and Number of neurons per layer
# [neurons in layer1, neurons in layer2, neurons in layer3,...]
# neurons = [[20, 50, 100, 150, 200], [20, 50, 100, 150, 200], [20, 50, 100, 150, 200], [20, 50, 100, 150, 200]]
# neurons = [[150, 200], [150, 200], [150, 200],[150, 200]]
neurons = [[1, 3, 5], [1, 3, 5, 10]]

# ！number of elements of neurons has to be the same as number of stateful

# Regularization, a way to conqour overfitting
# dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
dropout = [0.1, 0.2, 0.3]

# Define number of samples to fit model every time
# batch_size = [20, 50, 80, 100, 120]
batch_sizes = [20, 50]

# timesteps indicates how many subsamples you want to input in one training sample
# [number of sample per day * K], K days lagged samples as independent variables to forecast
# This will significantly affect consuming time to fit models
lagged_days = 1

# forecasting_length indicates how many steps you want to forecast in the future
horizon = 1

# forecasting scheme
forecast_scheme = 'quarterly2quarterly'

# Early stopping parameters
early_stopping = True
min_delta = 0.006
patience = 2

# Splitdate for train and test data.
split_date = dt.datetime(2017, 8, 1, 0, 0, 0, 0)

# Split percentage when doing cross-validation
validation_split = 0.2

# How many epochs in total. (Repeating times of fitting same batch of data)
epochs = 150

# If report model fitting progress
verbose = 1

# If log model fitting to tensorboard
tensor_board = True

# Forecast results

# Paths
# Directory with dataset
file_path = os.path.join(os.path.abspath(''), 'data/last_anonym_2017_vartime.csv')
# path = os.path.join(os.path.abspath(''), 'Source/lstm_load_forecasting/data/fulldataset.csv')
model_type = "Groupmodel"
code_path = os.getcwd()
fastec_dir = code_path + '/Output/FASTEC/'
output_dir = code_path + '/Output/LSTM/' + forecast_scheme + '_' + model_type + f'_lag-{lagged_days}_h-{horizon}_tf-{transform}'
res_dir = output_dir + '/results/'
plot_dir = output_dir + '/plots/'
model_dir = output_dir + '/models/'
log_dir = output_dir + '/logs/'
os.makedirs(res_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

results_file = res_dir + 'results.csv'
