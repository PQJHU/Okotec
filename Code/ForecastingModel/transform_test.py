from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# define contrived series
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
series = Series(data)
print(series)
# prepare data for normalization
values = series.values
values = values.reshape((len(values), 1))
# train the normalization
scaler_1 = MinMaxScaler(feature_range=(0, 1))
scaler_1_value = scaler_1.fit(values)
scaler_1_value_ft = scaler_1.fit_transform(values)

print(scaler_1_value_ft)
print('Min: %f, Max: %f' % (scaler_1_value.data_min_, scaler_1_value.data_max_))

# test standardscaler
scaler_2 = StandardScaler()
scaler_2_value = scaler_2.fit(values)
mean = values.mean()
std = values.std()
t1 = (values - mean)/std
scaler_2_value_ft = scaler_2.fit_transform(values)


# normalize the dataset and print
normalized = scaler_1_value.transform(values)
print(normalized)
# inverse transform and print
inversed = scaler_1_value.inverse_transform(normalized)
print(inversed)