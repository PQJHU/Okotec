import matplotlib as mlp
mlp.use('TkAgg')
from Code.PreAnalyzing.ReadData import read_transform_grouped, min_max_scaler
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import os
from statsmodels.tsa.stattools import acf, pacf

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


def plot_acf_full_sub(y, lags=None, name=None):
    plt.figure(figsize=(20, 5))
    acf_all = plt.subplot(2, 1, 1)


    # ======test
    corr= acf(x=y, unbiased=False,nlags=1000, alpha=0.05)
    plt.plot(corr[0])
    plt.plot(corr[1])

    lags=1000
    smt.graphics.plot_acf(y, lags=lags, ax=acf_all, unbiased=True)


    smt.graphics.plot_acf(y, lags=None, ax=acf_all, unbiased=True)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title('')
    acf_lags = plt.subplot(2, 1, 2)
    smt.graphics.plot_acf(y, lags=None, ax=acf_lags, unbiased=True)
    plt.xlabel('Time Lag', fontsize=18)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.title('')
    plt.tight_layout()
    plt.savefig(name + '.pdf', dpi=300)
    plt.show()



load_data, scaler = read_transform_grouped(query_vars='load', transform='min_max')
plot_dir = ('Output/plot/BasicTS/')
os.makedirs(plot_dir, exist_ok=True)


# ========== ACF plot on original
# plt.figure(figsize=(15, 5))
# plot_acf(y=load_data, name=plot_dir + 'load_var_acf')
plot_acf_full_sub(y=load_data, lags=1000, name=plot_dir + 'load_var_acf_1000')


# ========== ACF plot on differencing load
load_diff = load_data - load_data.shift(1)
load_diff.dropna(inplace=True)
plot_acf_full_sub(y=load_diff, lags=10, name=plot_dir + 'load_diff_acf_10')

load_diff_2 = load_diff - load_diff.shift(1)
load_diff_2.dropna(inplace=True)
plot_acf_full_sub(load_diff_2, lags=10, name=plot_dir + 'load_diff_2_acf_10')

