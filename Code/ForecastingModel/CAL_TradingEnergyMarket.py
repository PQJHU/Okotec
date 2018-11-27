import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt

data_dir = 'data/'
outdata_path = 'Output/Data/'
outplot_path = 'Output/plot/'



import pickle
with open(outdata_path + 'ClusteredLoadsForTrading.pkl', 'rb') as cluter_df_file:
    load_clusters_df = pickle.load(cluter_df_file)


# Import financial market data

spot = pd.read_csv(data_dir + 'vwap.csv', index_col=0, parse_dates=True)
print(spot.columns)
dayahead = pd.read_csv(data_dir + 'dayahead.csv', index_col=0, parse_dates=True)
print(dayahead.columns)

auction_prices = spot['Prices']
dayahead_prices = dayahead['Prices']

plt.plot(auction_prices)
plt.plot(dayahead_prices)

auction_describe = spot.describe()
day_ahead_describe = dayahead.describe()

auction_describe.to_csv(outdata_path + 'auction_descrip.csv')
day_ahead_describe.to_csv(outdata_path + 'dayahead_descip.csv')

auction_prices.mean()
dayahead_prices.mean()