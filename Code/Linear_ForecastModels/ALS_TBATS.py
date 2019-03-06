import matplotlib as mlp
mlp.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tbats import TBATS
import os
from Code.PreAnalyzing.ReadData import read_data_leipzig
import datetime as dt
from Code.Linear_ForecastModels.CAL_TBATS import get_hyperparams, train_test_sample_split
import pickle
from Code.Linear_ForecastModels.LocalParams import *

ts_model_dir = 'Output/TS/models/'
# os.makedirs(ts_model_dir, exist_ok=True)

# Read data, okotec 2ed dataset
leipzig_cooling_air = read_data_leipzig('cold_air')
cooling = leipzig_cooling_air['Cold (kW)']
air = leipzig_cooling_air['Air (kW)']
split_date = dt.date(2017, 5, 1)  # first 4 months


hyperparams, model_specs = get_hyperparams()

# with open(ts_model_dir + f'bats_cooling_{model_specs}.pkl', 'rb') as model_file:
#     bats_cooling = pickle.load(model_file)
#
# with open(ts_model_dir + f'bats_air_{model_specs}.pkl', 'rb') as model_file:
#     bats_air = pickle.load(model_file)

with open(ts_model_dir + f'tbats_cooling_{model_specs}.pkl', 'rb') as model_file:
    tbats_cooling = pickle.load(model_file)

with open(ts_model_dir + f'tbats_air_{model_specs}.pkl', 'rb') as model_file:
    tbats_air = pickle.load(model_file)


cooling_train_sample, cooling_test_sample = train_test_sample_split(cooling)
air_train_sample, air_test_sample = train_test_sample_split(air)


#
#
# summary = bats_fitted_cooling.summary()
# print(summary)
# y_hat = bats_fitted_cooling.y_hat
# residual =bats_fitted_cooling.resid
# alpha = bats_fitted_cooling.params.alpha
# beta = bats_fitted_cooling.params.x0
#
# import matplotlib as mlp
# mlp.use('TKAgg')
# import matplotlib.pyplot as plt
#
# plt.plot(residual)

# ====analysis
# print(bats_cooling.summary())
# print(bats_air.summary())
print(tbats_cooling.summary())
print(tbats_air.summary())

# ========IN SAMPLE
# cooling y_hat
cooling_train_sample['y_hat'] = tbats_cooling.y_hat
air_train_sample['y_hat'] = tbats_air.y_hat


# ======= OUT SAMPLE FORECASTING
tbats_cooling_pred = tbats_cooling.forecast(steps=len(cooling_test_sample))
cooling_test_sample['y_hat'] = tbats_cooling_pred



plt.plot(cooling_test_sample['Cold (kW)'])


fig = plt.figure(figsize=(15,7))
ax = plt.subplot(2,1,1)
plt.plot(cooling_train_sample['Cold (kW)'], color='k')
plt.xlim([cooling_train_sample.index[0].date(), cooling_train_sample.index[-1].date()])
x_tick = ax.get_xticks()

x_tick = np.delete(x_tick, [1,3,5,7,9,11])
ax.set_xaxis(x_tick)
plt.plot(cooling_train_sample['y_hat'], color='b')
plt.tight_layout()
plt.savefig(plot_dir_ts + 'test.png', dpi=300)
