import matplotlib as mpl

mpl.use('TKAgg')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.special import logit
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt


def plot_acf_pacf(signal, lags=None, name=None):
    plt.figure(figsize=(20, 5))
    acf = plt.subplot(1, 2, 1)
    smt.graphics.plot_acf(signal, lags=lags, ax=acf)
    plt.xlabel('Time Lag', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    pacf = plt.subplot(1, 2, 2)
    smt.graphics.plot_pacf(signal, lags=lags, ax=pacf)
    plt.xlabel('Time Lag', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(name + '.pdf', dpi=300)
    plt.show()


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


def load_dataset(path=None, modules=None, forecasting_interval=7, split_date=None, forecast_scheme=None,
                 grouped_up=False, transform='standard', new_exo=False, exo_lag=True):
    # Take 7-day subsample forecasting as an example:
    # Use the previous 7 days X data to forecasting 1 day's load
    # X_batch:
    # [X_i ~ X_i+7] --> y_i+8
    # 357 days * 7 days * 96 15-min intervals * 72 features
    # y_batch: 257 days * 96 15-min intervals
    df = pd.read_csv(path, delimiter=';', parse_dates=True, index_col=0)

    # shift exogenous variables back 2 hours
    if exo_lag:
        df.loc[:, df.columns[1:]] = df.loc[:, df.columns[1:]].shift(-8)
        df = df[~(df.index.date == max(df.index.date))]
    else:
        pass

    # add new exogenous variable, n_exo_1= 1 if l < 50,000, else: 0
    if new_exo:
        l_copy = df['l'].copy()
        l_copy.loc[l_copy.values <= 50000] = 1
        l_copy.loc[l_copy.values > 50000] = 0
        df['n_exo_1'] = l_copy
    else:
        pass

    df_scaled = df.copy()
    df_scaled = df_scaled.dropna()

    # scale the data using different method
    if transform == 'standard':
        df_pred, y_scaler = standard_scaler(df=df_scaled['l'])
        # df_exogenous, X_scaler = standard_scaler(df=df_exogenous)
    elif transform == 'log':
        df_pred, y_scaler = features_log_transform(df=df_scaled['l'])
        # df_exogenous, X_scaler = features_log_transform(df=df_exogenous)
    elif transform == '0-1':
        df_pred, y_scaler = min_max_scaler(df=df_scaled['l'])
        # df_exogenous, X_scaler = min_max_scaler(df=df_exogenous)
    else:
        raise ValueError("transform method is not specified")

    df_scaled['l'] = df_pred.values

    df_train = df_scaled.loc[(df_scaled.index < split_date)].copy()
    df_test = df_scaled.loc[df_scaled.index >= split_date].copy()

    # Split in train and test dataset
    X_train, y_train, time_frame_train = contruct_forecast_sample(df_splitted_sample=df_train,
                                                forecasting_interval=forecasting_interval,
                                                forecast_scheme=forecast_scheme,
                                                grouped_up=grouped_up,
                                                new_exo=new_exo)
    X_test, y_test, time_frame_test = contruct_forecast_sample(df_splitted_sample=df_test,
                                              forecasting_interval=forecasting_interval,
                                              forecast_scheme=forecast_scheme,
                                              grouped_up=grouped_up,
                                              new_exo=new_exo)

    return X_train, y_train, time_frame_train, X_test, y_test, time_frame_test, y_scaler


def contruct_forecast_sample(df_splitted_sample, forecasting_interval=7, forecast_scheme=None, grouped_up=False, new_exo=False):
    indicator_vars = {'all': df_splitted_sample.columns[0:],
                      'group_1': df_splitted_sample.columns[1:19].append(df_splitted_sample.columns[43:73]),
                      'group_2': df_splitted_sample.columns[19:44],
                      'group_3': df_splitted_sample.columns[44:73],
                      'group_new_exo': df_splitted_sample.columns[73:]
                      }

    df_splitted_sample = df_splitted_sample.sort_index()
    # df = df[indicator_vars[modules[0]]]

    if grouped_up is True:
        df_pred = df_splitted_sample['l'].copy()
        df_X = df_splitted_sample['l'].copy()

        group_1 = df_splitted_sample[indicator_vars['group_1']].sum(axis=1)
        group_1.name = 'group_1'

        group_2 = df_splitted_sample[indicator_vars['group_2']].sum(axis=1)
        group_2.name = 'group_2'

        group_3 = df_splitted_sample[indicator_vars['group_3']].sum(axis=1)
        group_3.name = 'group_3'
        # df_X = pd.concat([df_X, group_2], axis=1)

        if new_exo:
            group_new_exo = df_splitted_sample[indicator_vars['group_new_exo']].sum(axis=1)
            group_new_exo.name = 'new_exo'
            df_X = pd.concat([df_X, group_1, group_2, group_3, group_new_exo], axis=1)
        else:
            df_X = pd.concat([df_X, group_1, group_2, group_3], axis=1)

    else:
        df_pred = df_splitted_sample['l'].copy()
        df_X = df_splitted_sample.copy()

    # ************
    # shift the y variable in df_X 1 day (96 15-min intervals) forward to avoid using future information
    shifted_y = df_X['l'].shift(96)
    df_X['l'] = shifted_y
    # ************

    # drop the 1 day observations in df_pred and df_X, since one cannot forecast the 1st day
    first_day_of_sample = df_X.head(1).index.date[0]
    df_pred = df_pred[~(df_pred.index.date == first_day_of_sample)]
    df_X = df_X[~(df_X.index.date == first_day_of_sample)]

    feature_num = len(df_X.columns)

    if forecast_scheme == 'quarterly2quarterly':
        # df_X = df.drop(labels='l', axis=1).copy()
        # split data into daily array, every day with 96 samples and 72 features every sample
        y_split_daily = np.reshape(np.array(df_pred), (-1, 96))
        X_split_daily = np.reshape(np.array(df_X), (-1, 96, feature_num))

    elif forecast_scheme == 'quarterly2daily':
        y_split_daily = np.reshape(np.array(df_pred), (-1, 96)).mean(axis=1)
        X_split_daily = np.reshape(np.array(df_X), (-1, 96, feature_num))

    elif forecast_scheme == 'daily2daily':
        y_split_daily = np.reshape(np.array(df_pred), (-1, 96)).mean(axis=1)
        X_split_daily = np.reshape(np.array(df_X), (-1, 96, feature_num)).mean(axis=1)

    else:
        raise ValueError("scheme argu must input properly")

    X_batch = []
    y_batch = []

    # Split in features and label data for supervise learning
    # sequence_length = forecasting_interval + 1  # don't need to use 1 step forward for y because y variable is already shifted
    # forecasting_interval = 1
    for index in range(len(y_split_daily) - forecasting_interval + 1):
        print(index)
        print(index + forecasting_interval)
        # Forecasting 1 day forward using the previous
        exogenous_predictor_sample = X_split_daily[index: index + forecasting_interval]
        X_batch.append(exogenous_predictor_sample.reshape(-1, feature_num))
        # X_batch.append(exogenous_predictor_sample)
        y_batch.append(y_split_daily[index + forecasting_interval - 1])
        print(len(X_split_daily[index: index + forecasting_interval]))
        print('====')
    time_frame = df_pred.index
    return X_batch, y_batch, time_frame


def shift_load_data(df, shift_interval=None):
    """
    Depreciated

    """
    df_load = df['l'].copy()
    df_shifted_load = df.drop('l', axis=1).copy()

    df_shifted_load['load_shifted'] = np.nan
    for dt_idx in df_shifted_load.index:
        # define the shifting time
        if shift_interval == 'hour':
            dt_idx_shift = dt_idx + pd.DateOffset(hours=1)
        elif shift_interval == 'day':
            dt_idx_shift = dt_idx + pd.DateOffset(days=1)
        elif shift_interval == 'month':
            dt_idx_shift = dt_idx + pd.DateOffset(months=1)
        elif shift_interval == '15min':
            dt_idx_shift = dt_idx + pd.DateOffset(minutes=15)
        else:
            raise ValueError("parameter equals 'hour, 'day', 'month','15min'")
        print('===original time===', dt_idx)
        print('===shifted to time===', dt_idx_shift)
        try:
            df_shifted_load.loc[dt_idx, 'load_shifted'] = df_load[df_load.index == dt_idx_shift].values[0]
        except Exception as e:
            print(e)
    df_shifted_load.dropna(inplace=True, axis=0)
    return df_shifted_load
