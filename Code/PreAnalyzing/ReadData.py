# import matplotlib as mpl
# mpl.use('TKAgg')
# import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.special import logit
import numpy as np
import datetime as dt

"""
Read Example:

load_mm = read_transform_grouped(query_vars='load', transform='min_max')
load_st = read_transform_grouped(query_vars='load', transform='standard')
sch = read_transform_grouped(query_vars='schedules', grouped=False)
sch_g = read_transform_grouped(query_vars='schedules', grouped=True)
"""



def read_transform_grouped_1st(query_vars=None, transform=None, grouped=True):
    print(os.getcwd())
    ts = pd.read_csv('data/last_anonym_2017_vartime.csv', sep=';', index_col=0, parse_dates=True)
    print(ts)

    # Drop out the dates that has consisten value, 70865.614814...
    ts = ts[~ ((ts['l'].values < 70865.615) & (ts['l'].values > 70865.614))]

    # schedules

    # Separate the load curve into weekdays and weekends
    # weekdays_load_data = load_data[load_data.index.weekday.isin([0, 1, 2, 3, 4])]
    # weekdays_load_data_day_group = weekdays_load_data.groupby(weekdays_load_data.index.date, axis=0)
    # weekdays_load_data_month_group = weekdays_load_data.groupby(weekdays_load_data.index.month, axis=0)

    # The exogenous variables can be separated into 3 groups
    group_1 = ts.columns[1:19]
    group_2 = ts.columns[19:44]
    group_3 = ts.columns[44:73]

    src_group_1_sum = ts.loc[:, group_1].sum(axis=1)
    src_group_2_sum = ts.loc[:, group_2].sum(axis=1)
    src_group_3_sum = ts.loc[:, group_3].sum(axis=1)

    if query_vars == 'load':

        # Transform load
        load_data = ts.loc[:, 'l']
        if transform == 'min_max':
            load_data = min_max_scaler(load_data)
        elif transform == 'standard':
            load_data = standard_scaler(load_data)
        else:
            pass
        return load_data

    elif query_vars == 'schedules':

        # Group schedules
        if grouped is True:
            exo_vars = pd.concat([src_group_1_sum, src_group_2_sum, src_group_3_sum], axis=1)
        else:
            exo_vars = ts.drop(labels='l', axis=1)

        return exo_vars

    else:
        raise KeyError('Querying with Wrong Keyword')

def read_data_leipzig(query_var = None):
    os.getcwd()
    if query_var == 'cold_air':
        cold_air_ts = pd.read_csv('data/leipzig_2017.csv', index_col=0, parse_dates=True)
        cold_air_ts = cold_air_ts[~ (cold_air_ts.index.date == dt.date(2018, 1, 1))]

        return cold_air_ts
    elif query_var == 'production':
        production = pd.read_csv('data/Production_2017.csv', index_col=0, parse_dates=True, decimal=',')
        production = production[~ (production.index.date == dt.date(2018,1,1))]

        return production
    else:
        raise KeyError('Invalid keyword')



def standard_scaler(df):
    # Get all float type columns and standardize them
    df = pd.DataFrame(df)
    floats = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['float64']]
    scaler = StandardScaler()
    scaled_columns = scaler.fit_transform(df[floats])
    df[floats] = scaled_columns
    return df, scaler


def min_max_scaler(df):
    # Get all float type columns and standardize them
    df = pd.DataFrame(df)
    floats = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['float64']]
    scaler = MinMaxScaler()
    scaled_columns = scaler.fit_transform(df[floats])
    df[floats] = scaled_columns
    return df, scaler


def numeric_bound(x):
    for row in range(len(x)):
        for col in range(len(x[row])):
            val = x[row][col]
            if val == 0:
                # print(val)
                x[row][col] = val + 0.01
                # print(x[row][col])
            elif val == 1:
                # print(val)
                x[row][col] = val - 0.01
                # print(x[row][col])
    return x


def features_log_transform(df):
    df = pd.DataFrame(df)
    floats = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['float64']]
    scaler = MinMaxScaler()
    scaled_columns = scaler.fit_transform(df[floats])
    scaled_columns_numeric_bound = numeric_bound(scaled_columns)
    log_transorm_scaled = logit(scaled_columns_numeric_bound)
    df[floats] = log_transorm_scaled
    return df, scaler
