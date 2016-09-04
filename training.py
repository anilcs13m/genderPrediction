import sys
import random
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers import Embedding
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.core import Dense, Activation, Dropout
from keras.models import model_from_json
from keras.layers import Merge

from data_helper import loaddata

m_names, f_names = loaddata()

totalEntries = len(m_names) + len(f_names)
maxlen = len(max( m_names , key=len)) + len(max( f_names , key=len))

chars = set(  "".join(m_names) + "".join(f_names)  )
char_index = dict((c, i) for i, c in enumerate(chars))

X = np.zeros((totalEntries , maxlen, len(chars) ), dtype=np.bool)
y = np.zeros((totalEntries , 2 ), dtype=np.bool)

for i, name in enumerate(m_names):
    for t, char in enumerate(name):
        X[i, t, char_index[char]] = 1
    y[i, 0 ] = 1

for i, name in enumerate(f_names):
    for t, char in enumerate(name):
        X[i + len(m_names), t, char_index[char]] = 1
    y[i + len(m_names) , 1 ] = 1


nEpochs = 10
def baseline_model():
    # building model
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))

    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True))

    model.add(Dropout(0.4))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    return model

model=baseline_model()
json_string = model.to_json()

with open("model.json", "w") as file:
	file.write(json_string)
model.fit(X,y, batch_size=32, nb_epoch=nEpochs)
model.save_weights('model_lstm')


def baseline_model1():
    # building model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(maxlen, len(chars))))

    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))

    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    return model

model1=baseline_model1()
json_string = model1.to_json()

with open("model1.json", "w") as file:
    file.write(json_string)
model1.fit(X,y, batch_size=32, nb_epoch=nEpochs)
model1.save_weights('model_lstm1')
