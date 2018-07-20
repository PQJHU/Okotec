from keras.models import Sequential
from keras.utils.vis_utils import plot_model

model = Sequential()

from keras.layers import Dense, Activation

model.add(layer=Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
plot_model(model, to_file='test_archi_plot.png', show_shapes=True, show_layer_names=True)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


