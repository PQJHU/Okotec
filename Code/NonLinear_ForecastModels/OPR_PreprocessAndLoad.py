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


def PreprocessingData(file_path, sample_perday, lagged_days, horizon, split_date, forecast_scheme,
                      group_up, transform, new_exo, exo_lag):
    """
    Take 7-day subsample forecasting as an example:
    Use the previous l days X data to forecasting h day's load
    Y_t+1 ~ Y_t+h = f([X_t-l ~ X_t]), t = l,l+1,...,T-h
    Where h is forecast horizon, l is lagged periods, sample start from 0 to T

    :param file_path:
    :param sample_perday:
    :param lagged_days:
    :param horizon:
    :param split_date:
    :param forecast_scheme:
    :param group_up:
    :param transform:
    :param new_exo:
    :param exo_lag:
    :return:
    """

    df = pd.read_csv(file_path, delimiter=';', parse_dates=True, index_col=0)

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
        df_scaled['l'], y_scaler = standard_scaler(df=df_scaled['l'])
        # df_exogenous, X_scaler = standard_scaler(df=df_exogenous)
    elif transform == 'log':
        df_scaled['l'], y_scaler = features_log_transform(df=df_scaled['l'])
        # df_exogenous, X_scaler = features_log_transform(df=df_exogenous)
    elif transform == '0-1':
        df_scaled['l'], y_scaler = min_max_scaler(df=df_scaled['l'])
        # df_exogenous, X_scaler = min_max_scaler(df=df_exogenous)
    else:
        raise ValueError("transform method is not specified")

    # df_scaled['l'] = df_scaled['l'].values

    df_train = df_scaled.loc[(df_scaled.index < split_date)].copy()
    df_test = df_scaled.loc[df_scaled.index >= split_date].copy()

    # Split in train and test dataset
    X_train, y_train, time_frame_train = contruct_forecast_sample(df_splitted_sample=df_train,
                                                                  sample_perday=sample_perday,
                                                                  lagged_days=lagged_days,
                                                                  horizon=horizon,
                                                                  forecast_scheme=forecast_scheme,
                                                                  group_up=group_up,
                                                                  new_exo=new_exo)

    X_test, y_test, time_frame_test = contruct_forecast_sample(df_splitted_sample=df_test,
                                                               sample_perday=sample_perday,
                                                               lagged_days=lagged_days,
                                                               horizon=horizon,
                                                               forecast_scheme=forecast_scheme,
                                                               group_up=group_up,
                                                               new_exo=new_exo)

    return X_train, y_train, time_frame_train, X_test, y_test, time_frame_test, y_scaler


def contruct_forecast_sample(df_splitted_sample, sample_perday, lagged_days, horizon, forecast_scheme=None,
                             group_up=False, new_exo=False):
    indicator_vars = {'all': df_splitted_sample.columns[0:],
                      'group_1': df_splitted_sample.columns[1:19].append(df_splitted_sample.columns[43:73]),
                      'group_2': df_splitted_sample.columns[19:44],
                      'group_3': df_splitted_sample.columns[44:73],
                      'group_new_exo': df_splitted_sample.columns[73:]
                      }

    df_splitted_sample = df_splitted_sample.sort_index()
    # df = df[indicator_vars[modules[0]]]

    if group_up is True:
        Y = df_splitted_sample['l'].copy()
        X = df_splitted_sample['l'].copy()

        group_1 = df_splitted_sample[indicator_vars['group_1']].sum(axis=1)
        group_1.name = 'group_1'

        group_2 = df_splitted_sample[indicator_vars['group_2']].sum(axis=1)
        group_2.name = 'group_2'

        group_3 = df_splitted_sample[indicator_vars['group_3']].sum(axis=1)
        group_3.name = 'group_3'
        # X = pd.concat([X, group_2], axis=1)

        if new_exo:
            group_new_exo = df_splitted_sample[indicator_vars['group_new_exo']].sum(axis=1)
            group_new_exo.name = 'new_exo'
            X = pd.concat([X, group_1, group_2, group_3, group_new_exo], axis=1)
        else:
            pass
        X = pd.concat([X, group_1, group_2, group_3], axis=1)

    else:
        Y = df_splitted_sample['l'].copy()
        X = df_splitted_sample.copy()

    # ************
    # shift the Y variable in X h-day (96 15-min intervals) forward to avoid using future information
    # X['l'] = X['l'].shift(int(sample_perday*(horizon/sample_perday)))
    # ************

    # drop the 1 day observations in Y and X, since one cannot forecast the 1st day
    # first_day_of_sample = X.head(1).index.date[0]
    # Y = Y[~(Y.index.date == first_day_of_sample)]
    # X = X[~(X.index.date == first_day_of_sample)]

    feature_num = len(X.columns)
    days = sorted(list(set(Y.index.date)))

    if forecast_scheme == 'quarterly2quarterly':
        # X = df.drop(labels='l', axis=1).copy()
        # split data into daily array, every day with N samples and K features every sample
        Y_split_daily = np.reshape(np.array(Y), (-1, sample_perday))
        X_split_daily = np.reshape(np.array(X), (-1, sample_perday, feature_num))
        t_split_daily = np.reshape(np.array(Y.index), (-1, sample_perday))

    elif forecast_scheme == 'quarterly2daily':
        Y_split_daily = np.reshape(np.array(Y), (-1, sample_perday)).mean(axis=1)
        X_split_daily = np.reshape(np.array(X), (-1, sample_perday, feature_num))
        t_split_daily = np.reshape(np.array(days), (-1, 1))

    elif forecast_scheme == 'daily2daily':
        Y_split_daily = np.reshape(np.array(Y), (-1, sample_perday)).mean(axis=1)
        X_split_daily = np.reshape(np.array(X), (-1, sample_perday, feature_num)).mean(axis=1)
        t_split_daily = np.reshape(np.array(days), (-1, 1))

    else:
        raise ValueError("scheme argu must input properly")

    # Split in features and label data for supervise learning
    # sequence_length = forecasting_interval + 1  # don't need to use 1 step forward for Y because Y variable is already shifted
    # forecasting_interval = 1

    # for index in np.arange(start=int(lagged_days / sample_perday), stop=len(Y_split_daily)):
    #     print(index)

    X_batch = []
    Y_batch = []
    t_batch = []

    # lagged_days =2
    # horizon =2

    for index in range(len(Y_split_daily) - horizon - lagged_days+1):
        # print(index)
        X_sample = X_split_daily[index: index + lagged_days]
        X_batch.append(X_sample.reshape(-1, feature_num))
        # print(X_sample.shape)

        # Put the T-point at the (index + lagged_days - 1) day
        t_sample = t_split_daily[index+lagged_days -1]
        t_batch.append(t_sample.reshape(-1, 1))
        # print(t_sample)

        # X_batch.extend(X_sample[0][:,0])
        Y_sample = Y_split_daily[index + lagged_days: index + horizon + lagged_days]
        Y_batch.append(Y_sample.reshape(horizon*sample_perday, -1))
        # Y_batch.extend(Y_sample[0])
        # print(Y_sample.shape)
        # print('====')

    X_batch = np.reshape(np.array(X_batch), (len(X_batch), lagged_days*sample_perday, feature_num))
    Y_batch = np.reshape(np.array(Y_batch), (len(Y_batch),-1))
    t_batch = np.reshape(np.array(t_batch), (len(t_batch), -1))
    print(f'X shape: {X_batch.shape}')
    print(f'Y shape: {Y_batch.shape}')
    print(f'TimeFrame shape: {t_batch.shape}')
    # for index in range(len(Y_split_daily) - lagged_days + 1):
    #     print(index)
    #     print(index + lagged_days)
    #     # Forecasting 1 day forward using the previous
    #     exogenous_predictor_sample = X_split_daily[index: index + lagged_days]
    #     X_batch.append(exogenous_predictor_sample.reshape(-1, feature_num))
    #     # X_batch.append(exogenous_predictor_sample)
    #     Y_batch.append(Y_split_daily[index + lagged_days - 1])
    #     print(len(X_split_daily[index: index + lagged_days]))
    #     print('====')
    # time_frame = Y.index

    # ======= Use plot to test
    # print(np.shape(X_batch))
    # test_df_x = pd.DataFrame(X_batch, columns=['l_x'])
    # test_df_y = pd.DataFrame(Y_batch, columns=['l_y'])
    #
    # t1 = X_batch[:,:,0]
    #
    # plt.plot(test_df_x['l_x'])
    # plt.plot(test_df_y['l_y'])
    # ======= Finish test, separation algorithm works well

    return X_batch, Y_batch, t_batch
