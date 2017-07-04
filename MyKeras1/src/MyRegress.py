#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop

#(X_train,y_train),(X_test,y_test) = mnist.load_data()
import pickle
import gzip
f = gzip.open('mnist.pkl.gz', 'rb')
data = pickle.load(f)
f.close()
#print data
X_train, y_train, X_test, y_test = data[0][0],data[0][1],data[1][0],data[1][1]
print X_train.shape,y_train.shape,X_test.shape,y_test.shape
print type(X_train),X_train.dtype
# 
X_train = X_train.reshape(X_train.shape[0],-1)/255
X_test = X_test.reshape(X_test.shape[0],-1)/255

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

model = Sequential([Dense(32,input_dim=784),
                    Activation('relu'),
                    Dense(10),
                    Activation('softmax')
                    ])
rmsprop = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

print('Training..................')
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)

print('\nTesting.................')
loss,accuracy = model.evaluate(X_test, y_test, verbose=0)
print('test loss: ', loss)
print('test accuracy: ', accuracy)