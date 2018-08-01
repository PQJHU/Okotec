import pandas as pd
import os
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import datetime
from pylab import xlim
import statsmodels.tsa.api as smt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.special import logit
import numpy as np
from sklearn.neighbors import KernelDensity
import datetime as dt

# ===========Read Data===============
src = pd.read_csv('data/last_anonym_2017_vartime.csv', sep=';', index_col=0, parse_dates=True)
# Drop out the dates that has consisten value, 70865.614814...
src = src[~ ((src['l'].values < 70865.615) & (src['l'].values > 70865.614))]
outplot_path = ('Output/plot/')
load_curve = src['l']
schedules = src.copy(deep=True)
schedules.drop(labels='l', inplace=True, axis=1)

schedules_accu = list()

for row in schedules.index:
    print(row)
    row_value = schedules.loc[row, :]
    nonzero = row_value[row_value.values != 0].values
    schedules_accu.extend(nonzero)

# ===============Transform==================
# load_curve = np.asarray(load_curve[load_curve.index.date < dt.date(2017, 7, 1)])
load_curve = np.asarray(load_curve)
schedules_accu = np.asarray(schedules_accu)

# Transform data to 0-1
scaler = MinMaxScaler()

load_curve_scaled = scaler.fit_transform(X=load_curve.reshape(-1, 1))
schedules_accu = scaler.fit_transform(X=schedules_accu.reshape(-1, 1))


# kernels = ['gaussian', 'tophat', 'epanechnikov']

class PlotKernelDensityEstimator:
    """
    A object for plotting a series of KDE with given bandwidths and kernel functions
    """

    def __init__(self, data_points, mark_name):
        if isinstance(data_points, list):
            data_points = np.asarray(data_points)
        self.data_points = data_points
        # Default parameters
        self.kernel = 'epanechnikov'
        self.log_densities = dict()
        self.band_widths = np.arange(0.001, 0.05, 0.005)
        self.x_plot = np.linspace(0, 1, 100000)
        self.file_name = mark_name

    def log_density_estimate(self):
        for bandwidth in self.band_widths:
            kde = KernelDensity(bandwidth=bandwidth, kernel=self.kernel).fit(X=self.data_points.reshape(-1, 1))
            log_dens = kde.score_samples(self.x_plot.reshape(-1, 1))
            self.log_densities[f'{bandwidth}'] = log_dens
        return self.log_densities

    def plot_curve_hist_kde(self, log_dens, bin_num, hist_density=True):
        self.fig = plt.figure(figsize=(15, 5))
        plt.hist(self.data_points, bins=bin_num, density=hist_density)
        plt.plot(self.x_plot, np.exp(log_dens), '-')
        plt.title(f'{self.kernel} Kernel Estimation')
        return self.fig

    def save_plot(self):
        for bandwidth in self.band_widths:
            self.plot_curve_hist_kde(log_dens=self.log_densities[str(bandwidth)], bin_num=1000, hist_density=True)
            self.fig.savefig(f'{self.file_name}_{self.kernel}_{bandwidth}.png', dpi=300)


# ===========================Kernel density estimation for load curve points==============================

kde = PlotKernelDensityEstimator(data_points=load_curve_scaled, mark_name='LoadCurve')
kde.log_density_estimate()
kde.plot_curve_hist_kde(log_dens=kde.log_densities['0.006'], bin_num=1000)
kde.save_plot()

# ========================Plot exogenous variables (schedules)=======================


plt.hist(schedules_accu, bins=500)

kde_schedules_agg = PlotKernelDensityEstimator(data_points=schedules_accu, mark_name='schedules_agg')
kde_schedules_agg.log_density_estimate()
kde_schedules_agg.plot_curve_hist_kde(log_dens=kde_schedules_agg.log_densities['0.001'], bin_num=1000)
kde_schedules_agg.save_plot()
