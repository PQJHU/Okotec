import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
from pylab import xlim
import statsmodels.tsa.api as smt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.special import logit
import numpy as np

print(os.getcwd())
src = pd.read_csv('data/last_anonym_2017_vartime.csv', sep=';', index_col=0, parse_dates=True)
print(src)
output_path = ('Output/plot/')

# Drop out the dates that has consisten value, 70865.614814...
src = src[~ ((src['l'].values < 70865.615) & (src['l'].values > 70865.614))]

# Separate the load curve into weekdays and weekends
load_data = src.loc[:, 'l']
weekdays_load_data = load_data[load_data.index.weekday.isin([0, 1, 2, 3, 4])]
weekdays_load_data_day_group = weekdays_load_data.groupby(weekdays_load_data.index.date, axis=0)
weekdays_load_data_month_group = weekdays_load_data.groupby(weekdays_load_data.index.month, axis=0)

# The exogenous variables can be separated into 3 groups
group_1 = src.columns[1:19]
group_2 = src.columns[19:44]
group_3 = src.columns[44:73]

src_group_1_sum = src.loc[:, group_1].sum(axis=1)
src_group_2_sum = src.loc[:, group_2].sum(axis=1)
src_group_3_sum = src.loc[:, group_3].sum(axis=1)


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


def double_axs(time_index, data1, data2):
    fig, ax1 = plt.subplots(figsize=(40, 10), dpi=80)
    t = pd.to_datetime(time_index)
    a, = ax1.plot(t, data1.values, 'b-', linewidth=2, label=data1.name)
    ax1.set_xlabel('Date', fontsize=16)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(data1.name, fontsize=16)
    ax1.tick_params('y')

    ax2 = ax1.twinx()
    b, = ax2.plot(t, data2.values, 'r-', linewidth=2, label=data2.name)
    ax2.set_ylabel(data2.name, fontsize=16)
    ax2.tick_params('y')
    """
    if data1.max() > data2.max():
        high_mark = data1.max()
    else:
        high_mark = data2.max()
    if data1.min() < data2.min():
        low_mark = data1.min()
    else:
        low_mark = data2.min()
    ylim(low_mark * 1.1, high_mark * 1.1)
    """
    xlim(t.min() - datetime.timedelta(30), t.max() + datetime.timedelta(30))
    fig.tight_layout()

    p = [a, b]
    ax1.legend(p, [p_.get_label() for p_ in p],
               loc='upper right', fontsize=13)
    # plt.show()
    fig.savefig(os.path.join(output_path, data2.name + '_load_daxis_.png'), format='png', dpi=200)


for col_name in src.columns:
    uni_col_data = src.loc[:, col_name]
    print(uni_col_data.describe())
    plt.plot(uni_col_data)
    plt.title(col_name)
    plt.savefig(output_path + col_name + '.png', dpi=200)
    plt.close()

plt.figure(figsize=(20, 4))
for col_name in group_2:
    uni_col_data = src.loc[:, col_name]
    print(uni_col_data)
    plt.plot(uni_col_data)
plt.title('group_2_stack')
plt.savefig(output_path + 'group_2_stack.png', dpi=200)
plt.close()

plt.plot(src_group_1_sum)
plt.plot(src.loc[:, 'l'])

for col_name in group_1:
    load_data = src.loc[:, 'l']
    uni_col_data = src.loc[:, col_name]
    print(uni_col_data)
    double_axs(time_index=load_data.index, data1=load_data, data2=uni_col_data)

for num, data in enumerate(weekdays_load_data_month_group):
    print(num)
    print('===')
    print(data)
    fig, ax1 = plt.subplots(figsize=(40, 10), dpi=80)
    plt.plot(data[1])
    # plt.savefig(output_path + 'weekdays/monthly_load/' + data[0].strftime('%Y%m') + '.png')
    plt.savefig(output_path + 'weekdays/monthly_load/' + str(data[0]) + '.png')
    plt.close()

weekends_load_data = load_data[load_data.index.weekday.isin([5, 6])]
days = list(set(weekends_load_data.index.date))
days.sort()
# t = days[-1]
# t.weekday()


# plot standardized time series
load_data_sded = min_max_scaler(load_data)[0]
plt.figure(figsize=(15, 5))
plt.plot(load_data_sded, color='b', alpha=0.7)
plt.xlabel('Time', fontsize=18)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.savefig(output_path + 'load_sd_ts.pdf', dpi=300)

# plot exogenous variables in 3 groups
plt.figure(figsize=(15, 5))
plt.subplot(3, 1, 1)
plt.plot(src.loc[:, group_1].values)
plt.xticks([], [])
plt.ylim([0, 30])
plt.yticks(fontsize=17)
plt.subplot(3, 1, 2)
plt.plot(src.loc[:, group_2].values)
plt.xticks([], [])
plt.ylim([0, 30])
plt.yticks(fontsize=17)
plt.subplot(3, 1, 3)
plt.plot(src.loc[:, group_3])
plt.ylim([0, 30])
plt.yticks(fontsize=17)
plt.xticks(fontsize=17)
plt.xlabel('Time', fontsize=18)
plt.tight_layout()
plt.savefig(output_path + 'exogenous_var_ts_plot.pdf', dpi=300)

# plot load curve and exogenous variable aggregated into weekly and monthly
src_4_var = pd.concat([load_data, src_group_1_sum, src_group_2_sum, src_group_3_sum], axis=1)
src_4_var = min_max_scaler(src_4_var)[0]

# src_4_var['dayofweek'] = src_4_var.index.dayofweek
# src_4_var_weekly_agg = src_4_var.groupby(by='dayofweek', axis=0).mean()

src_4_var['weekofyear'] = src_4_var.index.weekofyear
src_4_var_weekofyeaer_agg = src_4_var.groupby(by='weekofyear', axis=0).mean()

plt.figure(figsize=(15, 5))
plt.plot(src_4_var_weekofyeaer_agg['l'], color='b', alpha=1)
plt.plot(src_4_var_weekofyeaer_agg[0], color='red', alpha=1)
plt.plot(src_4_var_weekofyeaer_agg[1], color='green', alpha=1)
plt.plot(src_4_var_weekofyeaer_agg[2], color='orange', alpha=1)
# plt.plot(src_4_var_weekofyeaer_agg[[0,1,2]],color=['green', 'red', 'yellow'], alpha=0.6)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlabel('Week of the year', fontsize=18)
plt.tight_layout()
plt.savefig(output_path + 'weeklyagg_plot.pdf', dpi=300)
