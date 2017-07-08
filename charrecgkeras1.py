import numpy as np
import copy
import os
import scipy
from scipy import ndimage
import pickle
import math
import tensorflow
import keras
import matplotlib.pyplot as plt	
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import random
from keras import backend as K
K.set_image_dim_ordering('th')

def char(j):
    if(j<=9):
	return j
    if(j==10):
	return 'A'
    if(j==11):
	return 'B'
    if(j==12):
	return 'C'
    if(j==13):
	return 'D'
    if(j==14):
	return 'E'
    if(j==15):
	return 'F'
    if(j==16):
	return 'G'
    if(j==17):
	return 'H'
    if(j==18):
	return 'I'
    if(j==19):
	return 'J'
    if(j==20):
	return 'K'
    if(j==21):
	return 'L'
    if(j==22):
	return 'M'
    if(j==23):
	return 'N'
    if(j==24):
	return 'O'
    if(j==25):
	return 'P'
    if(j==26):
	return 'Q'
    if(j==27):
	return 'R'
    if(j==28):
	return 'S'
    if(j==29):
	return 'T'
    if(j==30):
	return 'U'
    if(j==31):
	return 'V'
    if(j==32):
	return 'W'
    if(j==33):
	return 'X'
    if(j==34):
	return 'Y'
    if(j==35):
	return 'Z'
    if(j==36):
	return 'a'
    if(j==37):
	return 'b'
    if(j==38):
	return 'c'
    if(j==39):
	return 'd'
    if(j==40):
	return 'e'
    if(j==41):
	return 'f'
    if(j==42):
	return 'g'
    if(j==43):
	return 'h'
    if(j==44):
	return 'i'
    if(j==45):
	return 'j'
    if(j==46):
	return 'k'
    if(j==47):
	return 'l'
    if(j==48):
	return 'm'
    if(j==49):
	return 'n'
    if(j==50):
	return 'o'
    if(j==51):
	return 'p'
    if(j==52):
	return 'q'
    if(j==53):
	return 'r'
    if(j==54):
	return 's'
    if(j==55):
	return 't'
    if(j==56):
	return 'u'
    if(j==57):
	return 'v'
    if(j==58):
	return 'w'
    if(j==59):
	return 'x'
    if(j==60):
	return 'y'
    if(j==61):
	return 'z'
	
	
seed = 128
rng = np.random.RandomState(seed)

output_num_units = 62

epochs = 20
batch_size = 100



dict={}
with open('train_dataset.pickle','rb') as f:
    dict=pickle.load(f)
    train_dataset=dict['train_dataset']
    train_labels=dict['train_labels']
    valid_dataset=dict['valid_dataset']
    valid_labels=dict['valid_labels']
    test_dataset=dict['test_dataset']
    test_labels=dict['test_labels']

per1=np.random.permutation(train_dataset.shape[0])
per2=np.random.permutation(valid_dataset.shape[0])
per3=np.random.permutation(test_dataset.shape[0])
train_dataset=train_dataset[per1,:]
train_labels=train_labels[per1,:]
valid_dataset=valid_dataset[per2,:]
valid_labels=valid_labels[per2,:]
test_dataset=test_dataset[per3,:]
test_labels=test_labels[per3,:]
train_dataset=np.reshape(train_dataset,(train_dataset.shape[0],1,40,40))
test_dataset=np.reshape(test_dataset,(test_dataset.shape[0],1,40,40))
valid_dataset=np.reshape(valid_dataset,(valid_dataset.shape[0],1,40,40))
train_y = keras.utils.np_utils.to_categorical(train_labels)
valid_y = keras.utils.np_utils.to_categorical(valid_labels)
test_y = keras.utils.np_utils.to_categorical(test_labels)

# create model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(1, 40, 40), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(output_num_units, activation='softmax'))


# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model = model.fit(train_dataset, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(valid_dataset, valid_y))

model.save_weights('charrecgweights1.h5')	





