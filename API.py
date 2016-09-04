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

from data_helper import loaddata

m_names, f_names = loaddata()

totalEntries = len(m_names) + len(f_names)
maxlen = len(max( m_names , key=len)) + len(max( f_names , key=len))

chars = set(  "".join(m_names) + "".join(f_names)  )
char_indices = dict((c, i) for i, c in enumerate(chars))

with open("model.json", 'r') as model:
    json_string = model.read()
## model load
model = model_from_json(json_string)
model.load_weights('model_lstm')


with open("model1.json", 'r') as model1:
    json_string = model1.read()
##model load
model1 = model_from_json(json_string)
model1.load_weights('model_lstm1')

def predict(name):
	global maxlen , chars , char_indices
	x = np.zeros((1, maxlen, len(chars)))

	for t, char in enumerate(name):
		x[0, t, char_indices[char]] = 1

	preds1 = model.predict(x, verbose=0)[0]
	preds2 = model1.predict(x, verbose=0)[0]
	preds = 0.4*preds1 + 0.6*preds2 ## two model embeding
	return preds

while True:
	print("Enter Name")
	n = input()
	v = predict(n)
	if v[0] > v[1]:
		print("Male")
	else:
		print("Female")