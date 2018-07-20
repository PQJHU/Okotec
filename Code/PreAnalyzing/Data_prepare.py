import pandas as pd
import os

src = pd.read_csv('Source/last_anonym_2017_vartime.csv', sep=';', index_col=0, parse_dates=True)
print(src)

wp1 = pd.read_csv('data/wp1.csv', sep=' ')
wp2 = pd.read_csv('data/wp2.csv', sep=' ')


class Data_Splitter:

    def __init__(self, data_df):
        self._data = data_df
        # group_1 is P_i, i = {1 - 18 and 43 - 72}
        # group_2 is P_i, i = {19 - 42}
        self.group_1 = self._data.columns[1:19].append(src.columns[43:])
        self.group_2 = self._data.columns[19:43]

    def columns_split(self):
        self.load = self._data.loc[:, 'l']
        self.src_group_1 = self._data.loc[:, self.group_1]
        self.src_group_2 = self._data.loc[:, self.group_2]
        return self.src_group_1, self.src_group_2

    def daily_split(self, split_groupe=False):
        _data_ = self._data.groupby(by=self._data.index.date, axis=0, sort=True)
        return _data_


src_split = Data_Splitter(src)
src_g1, src_g2 = src_split.columns_split()

src_daily = src_split.daily_split()

for ele in src_split.daily_split():
    print(ele)
