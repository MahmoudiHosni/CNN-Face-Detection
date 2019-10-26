# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:42:46 2018

@author: Tbarki
"""
from keras.layers.core import Dense, Activation, Dropout
import numpy as np 
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import to_categorical
def get_data_set(filepath):
    imgs = []
    labels = []
    print('Start reading files ... ')
    for f in os.listdir(filepath):
        if not (f.endswith('pgm')):
            labels.append(f.split('.')[0])
            print('Reading file: ' + f)
            img = np.asarray(Image.open(filepath+f))
            imgs.append(img)
    print('Reading files finihsed ')
    return np.asarray(imgs), labels

filepath = ('/YALE/faces/')
imgs, labels = get_data_set(filepath)
labels1=[labels[i][-2:] for i in range(len(labels))]
X_train, X_test, y_train, y_test = train_test_split(imgs, labels1, test_size =0.2)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train=y_train[:,1:]
y_test=y_test[:,1:]

classifier = Sequential()
classifier.add(Convolution2D(32, kernel_size=9, input_shape = (243, 320, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (8, 8)))
classifier.add(Convolution2D(32, kernel_size=3, activation = 'relu'))
classifier.add(Convolution2D(32, kernel_size=3, activation = 'relu'))
classifier.add(Convolution2D(16, kernel_size=3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (4, 4)))
classifier.add(Flatten())
classifier.add(Dense( activation = 'relu',units = 248))
classifier.add(Dense( activation = 'relu',units = 128))
classifier.add(Dense(activation = 'softmax',units=15))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50,batch_size=32)
#TRAINING ACCURACY = 100%
#VALIDATION ACCURACY =90.91%
