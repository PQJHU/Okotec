import pandas as pd
from Code.NonLinear_ForecastModels.LSTM_Params import file_path,plot_dir,output_dir



# read data
ts = pd.read_csv(file_path, delimiter=';', parse_dates=True, index_col=0)
load = ts['l']


#

# import model




