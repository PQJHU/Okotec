import pandas as pd
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import os

root_path = os.getcwd()
data_path = root_path + '/data/'
plot_path = root_path + '/Output/plot/EnergyMarket/'

dayahead_auction = pd.read_csv(data_path + 'auction.csv', index_col=0, parse_dates=True)
dayahead_price = pd.read_csv(data_path + 'dayahead.csv', index_col=0, parse_dates=True)
spot_vwp = pd.read_csv(data_path + 'vwap.csv', index_col=0, parse_dates=True)

spot_vwp.columns





