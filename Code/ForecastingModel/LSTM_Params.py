import datetime as dt
import os

model_cat_id = "forecasting_model_grouped_"

# Which features from the dataset should be loaded:
features = ['all']
# If group up the 2 product schedules
group_up = True
# If adding new exogenous variable
new_exo = False
# If shift exogenous variables back 2 hours, because those schedule are lagged
exo_lag = False

# LSTM Layer configuration
# ========================
# Stateful True or false
layer_conf = [True, True, True]
# Number of neurons per layer
cells = [[20, 50, 100, 150, 200], [10, 20, 50], [10, 20, 50]]
# Regularization per layer
dropout = [0, 0.1, 0.2]
# Size of how many samples are used for one forward/backward pass
batch_size = [5, 10, 15, 20]
# timesteps indicates how many subsamples you want to input in one training sample
timesteps = [96 * 1]  # 1 days sample to train
# forecasting_length indicates how many steps you want to forecast in the future
forecast_length = 96
# forecasting scheme
forecast_scheme = 'quarterly2quarterly'

# Early stopping parameters
early_stopping = True
min_delta = 0.006
patience = 2

# Splitdate for train and test data.
split_date = dt.datetime(2017, 8, 1, 0, 0, 0, 0)

# Validation split percentage
validation_split = 0.2
# How many epochs in total
epochs = 30
# Set verbosity level. 0 for only per model, 1 for progress bar...
verbose = 1

code_path = os.getcwd()
fastec_dir = code_path + '/Output/FASTEC/'
res_dir = code_path + '/Output/LSTM/' + model_cat_id + forecast_scheme + '/Model_Output/results/'
plot_dir = code_path + '/Output/LSTM/' + model_cat_id + forecast_scheme + '/Model_Output/plots/'
model_dir = code_path + '/Output/LSTM/' + model_cat_id + forecast_scheme + '/Model_Output/models/'
os.makedirs(res_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
output_table = res_dir + 'results.csv'
test_output_table = res_dir + 'test_results.csv'

