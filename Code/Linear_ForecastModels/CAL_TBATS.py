from Code.PreAnalyzing.ReadData import read_data_leipzig
from tbats import TBATS, BATS
import pandas as pd
import numpy as np
import pickle
import os
import datetime as dt

ts_model_dir = 'Output/TS/models/'
os.makedirs(ts_model_dir, exist_ok=True)

# Read data, okotec 2ed dataset
leipzig_cooling_air = read_data_leipzig('cold_air')
cooling = leipzig_cooling_air['Cold (kW)']
air = leipzig_cooling_air['Air (kW)']
split_date = dt.date(2017, 5, 1)  # first 4 months


# use_box_cox = [True, None]
# seasonal_periods = [96, 96*7, 96*7*30, 96*7*30*3 ]  # daily, weekly, monthly, quarter monthly
# seasonal_periods = (96, 96*5, 96*7, 96*5*30, 96*7*30, 96*5*30*3, 96*7*30*3 )  # daily, weekly, monthly, quarter monthly


def train_test_sample_split(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    train_sample = df[df.index.date < split_date]
    test_sample = df[df.index.date >= split_date]
    return train_sample, test_sample


# Hyperparameters

def get_hyperparams():
    hyper_params = {'use_box_cox': None,
                    # 'box_cox_bounds': (0, 1),
                    'use_trend': None,
                    'use_damped_trend': True,
                    'seasonal_periods': (96, 672),
                    'use_arma_errors': True
                    }
    model_specs = '_'.join(str(value[1]) for value in list(hyper_params.items()))
    return hyper_params, model_specs


def model_calibrate(model_name, **kwargs):
    if model_name == 'bats':
        return BATS(**kwargs, show_warnings=True, n_jobs=None, context=None)

    if model_name == 'tbats':
        return TBATS(**kwargs, show_warnings=True, n_jobs=None, context=None)
    else:
        raise KeyError("Wrong keyword Input")


def model_estimation(estimator, y, model_name):
    fitted_estimator = estimator.fit(y)
    with open(ts_model_dir + f'{model_name}.pkl', 'wb') as model_file:
        pickle.dump(fitted_estimator, model_file)
    return fitted_estimator


def models_estimation():
    cooling_train_sample, cooling_test_sample = train_test_sample_split(cooling)
    air_train_sample, air_test_sample = train_test_sample_split(air)
    hyper_params, model_specs = get_hyperparams()

    bats_estimator = model_calibrate(model_name='bats', **hyper_params)

    bats_cooling = model_estimation(estimator=bats_estimator, y=cooling_train_sample,
                                    model_name=f'bats_cooling_{model_specs}')
    bats_air = model_estimation(estimator=bats_estimator, y=air_train_sample, model_name=f'bats_air_{model_specs}')

    tbats_estimator = model_calibrate(model_name='tbats', **hyper_params)
    tbats_cooling = model_estimation(estimator=tbats_estimator, y=cooling_train_sample,
                                     model_name=f'tbats_cooling_{model_specs}')
    tbats_air = model_estimation(estimator=tbats_estimator, y=air_train_sample, model_name=f'tbats_air_{model_specs}')
