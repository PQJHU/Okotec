from Code.NonLinear_ForecastModels.LSTM_Params import *
from Code.NonLinear_ForecastModels import OPR_PreprocessAndLoad

# test for creating model
import keras.models as kmodel
import keras.layers as klayer
import keras.callbacks as kc

# Loading the data:
X_train, y_train, time_frame_train, X_test, y_test, time_frame_test, y_scaler = OPR_PreprocessAndLoad.PreprocessingData(
    file_path=file_path,
    lagged_days=lagged_days,
    sample_perday=sample_perday,
    horizon=horizon,
    split_date=split_date,
    forecast_scheme=forecast_scheme,
    group_up=group_up,
    transform=transform,
    new_exo=new_exo,
    exo_lag=exo_lag)

batch_size = [20]
N_batch = int(X_train.shape[0] / batch_size[0])

X_train = X_train[0:N_batch * batch_size[0]]
y_train = y_train[0:N_batch * batch_size[0]]

units = 10

model_lstm = kmodel.Sequential()
model_lstm.add(klayer.LSTM(units=units,
                           input_shape=(lagged_days * sample_perday, 4),
                           batch_input_shape=(batch_size[0], lagged_days * sample_perday, 4),
                           return_sequences=False,
                           stateful=True
                           ))
# model_lstm.add(klayer.LSTM(50, return_sequences=True, stateful=True))
# model_lstm.add(klayer.LSTM(50, return_sequences=False, stateful=True))
model_lstm.add(klayer.Dropout(0.3))
model_lstm.add(klayer.Dense(horizon * sample_perday))
model_lstm.compile(loss='mse', optimizer='adam', metrics=['mae'])

print(f'LSTM parameter: {model_lstm.summary()}')

early_stop = kc.EarlyStopping(monitor='loss',
                              min_delta=min_delta,
                              patience=4,
                              verbose=2,
                              mode='auto')

history_lstm = model_lstm.fit(X_train, y_train,
                              epochs=50,
                              batch_size=batch_size[0],
                              validation_split=0.3,
                              verbose=2,
                              callbacks=[early_stop],
                              )

model_gru = kmodel.Sequential()
model_gru.add(klayer.GRU(units=units,
                         input_shape=(lagged_days * sample_perday, 4),
                         batch_input_shape=(batch_size[0], lagged_days * sample_perday, 4)))
model_gru.add(klayer.Dropout(0.3))
model_gru.add(klayer.Dense(horizon * sample_perday))
model_gru.compile(loss='mse', optimizer='adam', metrics=['mae'])

history_gru = model_gru.fit(X_train, y_train,
                            epochs=50,
                            batch_size=batch_size[0],
                            validation_split=0.3,
                            verbose=2,
                            callbacks=[early_stop])
print(f'GRU parameter: {model_gru.summary()}')
