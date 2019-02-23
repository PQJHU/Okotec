import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from Code.NonLinear_ForecastModels.LSTM_Params import file_path,plot_dir,output_dir
from tabulate import tabulate


ts = pd.read_csv(file_path, delimiter=';', parse_dates=True, index_col=0)

load = ts['l']


# ========== stationary test =============
# ADF, null of unit root
adf_result = adfuller(x=load.values)

print(tabulate([['ADF-stat:', adf_result[0]],['p-value:', adf_result[1]], ['lags:', adf_result[2]]],
               tablefmt='jira', numalign='right', floatfmt= '.2f'))


# KPSS
kpss_result = kpss(x=load.values)
print(tabulate([['kpss-stat:', kpss_result[0]],['p-value:', kpss_result[1]], ['lags:', kpss_result[2]]],
               tablefmt='jira', numalign='right', floatfmt= '.2f'))

# differencing variable
