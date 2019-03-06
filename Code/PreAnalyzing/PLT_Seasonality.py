import matplotlib as mlp

mlp.use('TKAgg')
from Code.PreAnalyzing.ReadData import read_data_leipzig
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os

plot_dir = 'Output/plot/Leipzig/BasicTS/'
os.makedirs(plot_dir, exist_ok=True)

load = read_data_leipzig(query_var='cold_air')
cold = load['Cold (kW)']
cold_smooth = cold.rolling(window=40).mean()
air = load['Air (kW)']

production = read_data_leipzig(query_var='production')

# ======load time series plot
# full period
plt.figure(figsize=(15, 7))
plt.subplot(2, 1, 1)
plt.plot(cold)
plt.ylabel('Cooling Power (kW)', fontsize=15)

plt.subplot(2, 1, 2)
plt.plot(air)
plt.ylabel('Air Power (kW)', fontsize=15)
plt.xlabel('Time', fontsize=16)
plt.tight_layout()

plt.savefig(plot_dir + 'cold_air_fullperiod.png', dpi=300)

# Daily seasonality
load['hour'] = [day.hour for day in load.index]

hourly_mean = load.groupby(by=load.index.hour, axis=0, sort=True).mean()
hourly_95quantile = load.groupby(by=load.index.hour, axis=0, sort=True).quantile(0.95)
hourly_5quantile = load.groupby(by=load.index.hour, axis=0, sort=True).quantile(0.05)

plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.scatter(x=load['hour'], y=load['Cold (kW)'], color='b')
plt.plot(hourly_mean['Cold (kW)'], linestyle='--', color='k')
plt.plot(hourly_95quantile['Cold (kW)'], linestyle='--', color='g')
plt.plot(hourly_5quantile['Cold (kW)'], linestyle='--', color='r')
# plt.xlabel('Weekday', fontsize=15)
plt.ylabel('Cooling Power (kW)', fontsize=15)
# plt.tight_layout()
# plt.savefig(plot_dir + 'cold_weekly_seasonality.png', dpi=300)

# air power seasonality
# plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 2)
plt.scatter(x=load['hour'], y=load['Air (kW)'], color='b')
plt.plot(hourly_mean['Air (kW)'], linestyle='--', color='k')
plt.plot(hourly_95quantile['Air (kW)'], linestyle='--', color='g')
plt.plot(hourly_5quantile['Air (kW)'], linestyle='--', color='r')
plt.xlabel('Hour', fontsize=15)
plt.ylabel('Air Power (kW)', fontsize=15)
plt.tight_layout()
plt.savefig(plot_dir + 'cold_air_daily_seasonality.png', dpi=300)

daily_samples = load.groupby(by=load.index.date, axis=0, sort=True)

for num, daily_sample in enumerate(daily_samples):
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.title(daily_sample[0])
    plt.plot(daily_sample[1]['Cold (kW)'])
    plt.ylabel('Cooling Power (kW)', fontsize=15)

    plt.subplot(2, 1, 2)
    plt.plot(daily_sample[1]['Air (kW)'])
    plt.ylabel('Air Power (kW)', fontsize=15)

    plt.xlabel('Time', fontsize=15)
    plt.tight_layout()
    plt.savefig(plot_dir + f'/daily_load/{num}_cold_air_daily.png', dpi=300)
    plt.close()

# ======Weekly seasonality

load['week'] = [day.dayofweek for day in load.index]
weekday_mean = load.groupby(by='week', axis=0, sort=True).mean()
weekday_95quantile = load.groupby(by='week', axis=0, sort=True).quantile(0.95)
weekday_5quantile = load.groupby(by='week', axis=0, sort=True).quantile(0.05)

# cold power seasonality
cold_weekdays = load[['Cold (kW)', 'week']]
air_weekdays = load[['Air (kW)', 'week']]

plt.figure(figsize=(15, 6))

plt.subplot(2, 1, 1)
plt.scatter(x=cold_weekdays['week'], y=cold_weekdays['Cold (kW)'], color='b')
plt.plot(weekday_mean['Cold (kW)'], linestyle='--', color='k')
plt.plot(weekday_95quantile['Cold (kW)'], linestyle='--', color='g')
plt.plot(weekday_5quantile['Cold (kW)'], linestyle='--', color='r')
# plt.xlabel('Weekday', fontsize=15)
plt.ylabel('Cooling Power (kW)', fontsize=15)
# plt.tight_layout()
# plt.savefig(plot_dir + 'cold_weekly_seasonality.png', dpi=300)

# air power seasonality
# plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 2)
plt.scatter(x=air_weekdays['week'], y=air_weekdays['Air (kW)'], color='b')
plt.plot(weekday_mean['Air (kW)'], linestyle='--', color='k')
plt.plot(weekday_95quantile['Air (kW)'], linestyle='--', color='g')
plt.plot(weekday_5quantile['Air (kW)'], linestyle='--', color='r')
plt.xlabel('Weekday', fontsize=15)
plt.ylabel('Air Power (kW)', fontsize=15)
plt.tight_layout()
plt.savefig(plot_dir + 'cold_air_weekly_seasonality.png', dpi=300)

# plt.savefig(plot_dir + 'air_weekly_seasonality.png', dpi=300)

# Weekly sample
week_sample = load[load['week'] == 10]
plt.subplot(2, 1, 1)
plt.plot(week_sample['Cold (kW)'])
plt.ylabel('Cooling Power (kW)', fontsize=15)

plt.subplot(2, 1, 2)
plt.plot(week_sample['Air (kW)'])
plt.ylabel('Air Power (kW)', fontsize=15)

plt.xlabel('Time', fontsize=15)
plt.savefig(plot_dir + 'cold_air_weekly.png', dpi=300)

# =======Monthly seasonality
load['date'] = [day.date().day for day in load.index]
monthly_mean = load.groupby(by=load['date'], axis=0, sort=True).mean()
monthly_95quantile = load.groupby(by=load['date'], axis=0, sort=True).quantile(0.95)
monthly_5quantile = load.groupby(by=load['date'], axis=0, sort=True).quantile(0.05)

# cold power monthly seasonality
cold_monthlyday = load.loc[:, ['Cold (kW)', 'date']]
plt.figure(figsize=(15, 6))

plt.subplot(2, 1, 1)
plt.scatter(x=cold_monthlyday['date'], y=cold_monthlyday['Cold (kW)'], color='b')
plt.plot(monthly_mean['Cold (kW)'], linestyle='--', color='k')
plt.plot(monthly_95quantile['Cold (kW)'], linestyle='--', color='g')
plt.plot(monthly_5quantile['Cold (kW)'], linestyle='--', color='r')
# plt.xlabel('Montly Day', fontsize=15)
plt.ylabel('Cooling Power (kW)', fontsize=15)
# plt.tight_layout()
# plt.savefig(plot_dir + 'cold_monthly_seasonality.png', dpi=300)

plt.subplot(2, 1, 2)
air_monthlyday = load.loc[:, ['Air (kW)', 'date']]
# plt.figure(figsize=(15,6))
plt.scatter(x=air_monthlyday['date'], y=air_monthlyday['Air (kW)'], color='b')
plt.plot(monthly_mean['Air (kW)'], linestyle='--', color='k')
plt.plot(monthly_95quantile['Air (kW)'], linestyle='--', color='g')
plt.plot(monthly_5quantile['Air (kW)'], linestyle='--', color='r')
plt.xlabel('Monthly day', fontsize=15)
plt.ylabel('Air Power (kW)', fontsize=15)
plt.tight_layout()
plt.savefig(plot_dir + 'cold_air_monthly_seasonality.png', dpi=300)

# One month
load['month'] = [day.month for day in load.index]
month_sample = load[load['month'] == 1]

plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(month_sample['Cold (kW)'])
plt.ylabel('Cooling Power (kW)', fontsize=15)

plt.subplot(2, 1, 2)
plt.plot(month_sample['Air (kW)'])
plt.ylabel('Air Power (kW)', fontsize=15)

plt.xlabel('Time', fontsize=15)
plt.savefig(plot_dir + 'cold_air_month_7.png', dpi=300)

# =======Load with production

plt.figure(figsize=(15, 7))
plt.subplot(2, 1, 1)
plt.plot(cold)
plt.ylabel('Cooling Power (kW)', fontsize=15)

for col in production.columns:
    plt.subplot(2, 1, 2)
    plt.plot(production[col])
plt.ylabel('Production', fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.savefig(plot_dir + 'cold_production_full.png', dpi=300)

plt.figure(figsize=(15, 7))
plt.subplot(2, 1, 1)
plt.plot(air)
plt.ylabel('Air Power (kW)', fontsize=15)

for col in production.columns:
    plt.subplot(2, 1, 2)
    plt.plot(production[col])
plt.ylabel('Production', fontsize=15)
plt.xlabel('Time', fontsize=15)
plt.savefig(plot_dir + 'air_production_full.png', dpi=300)

# ======impulse Response analysis
