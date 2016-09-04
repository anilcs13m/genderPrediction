from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.models import model_from_json

import numpy as np
import random
import sys

def loaddata():
	with open("male.txt") as m:
	    m_names = m.readlines()

	with open("female.txt") as f:
	    f_names = f.readlines()

	with open("male1.txt") as mm:
		mread = mm.readlines()
		mname=[]
		for m in mread:
			mname.append(m.split(' ')[0])

	with open("female1.txt") as ff:
		fread = ff.readlines()
		fname=[]
		for f in fread:
			fname.append(f.split(' ')[0])

	comman_names = []

	for f_name in f_names:
		if f_name in m_names:
			comman_names.append(f_name)

	for f_name in fname:
		if f_name in mname:
			comman_names.append(f_name)
	

	m_names = [m.lower() for m in m_names if not m in comman_names]
	f_names = [f.lower() for f in f_names if not f in comman_names]

	mname = [m.lower() for m in mname if not m in comman_names]
	fname = [f.lower() for f in fname if not f in comman_names]

	for f_name in fname:
		if f_name not in f_names:
			f_names.append(f_name)
	
	for m_name in mname:
		if m_name not in m_names:
			m_names.append(m_name)
	
	return m_names,f_names

