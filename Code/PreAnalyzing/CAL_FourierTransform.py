import pandas as pd
from Code.PreAnalyzing.ReadData import read_data_leipzig,read_transform_grouped_1st
import numpy as np
from numpy import fft
import matplotlib as mlp
mlp.use('TKAgg')
import matplotlib.pyplot as plt
from scipy.signal import periodogram

leipzig= read_data_leipzig(query_var='cold_air')
cooling = leipzig['Cold (kW)']
air = leipzig['Air (kW)']

sin_func = [np.sin(5*x) for x in np.linspace(-10, 10, 1000)]
plt.plot(sin_func)

sin_trans = fft.fft(sin_func)
plt.plot(sin_trans)

pg_cooling = periodogram(cooling.values)
plt.plot(pg_cooling[0], pg_cooling[1])


pg_air = periodogram(air.values)
plt.plot(pg_air[0], pg_air[1])

