"""
AR, ARX, ARMA(p,q), ARMAX(p,q,j) models, parameters estimation with OLS
Using SARIMAX to estimator AR models with exogenous variables
"""
import matplotlib as mlp
mlp.use('TKAgg')
import matplotlib.pyplot as plt


import pandas as pd
from Code.NonLinear_ForecastModels.LSTM_Params import file_path,plot_dir,output_dir
import statsmodels.api as sm
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA,ARIMA
from statsmodels.tsa.arima_process import ArmaProcess

# read data
ts = pd.read_csv(file_path, delimiter=';', parse_dates=True, index_col=0)
load = ts['l']

# Simulation data
random.seed(1124)
p,q, d =10, 10, 10 # ARMAX(p,q, d)

X = [random.uniform(0,1) + random.normalvariate(0,1) for x in np.linspace(0,1000)]
plt.plot(X)

arparams = np.array([.75, -.25])
ar = np.r_[1, -arparams]


y = ArmaProcess(ar=[-0.8, -0.2, 0.3, 0.5], ma=[-0.71, 0.35, 0.11]).generate_sample(nsample=35000, scale=1)
plt.plot(y)

# hyperparameters
def get_hyperparams():
    hyperparams = {
        'order':(),

    }


def AR_model_calibrate(**hyperparams):
    pass

def AR_model():
    pass


# import model

X = sm.add_constant(load)

class tsa_ARMAX:

    def __init__(self, ts, exog):
        self.ts = ts
        self.exog = exog


