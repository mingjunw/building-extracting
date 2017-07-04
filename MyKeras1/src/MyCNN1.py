import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.utils import np_utils
from keras.datasets import mnist

#(x_train,y_train),(x_test,y_test) = mnist.load_data()
import pickle
import gzip
f = gzip.open('mnist.pkl.gz', 'rb')
data = pickle.load(f)
f.close()
#print data
x_train, y_train, x_test, y_test = data[0][0],data[0][1],data[1][0],data[1][1]
print x_train.shape,y_train.shape,x_test.shape,y_test.shape

x_train = x_train.reshape(x_train.shape[0],1,28,28)
x_test = x_test.reshape(x_test.shape[0],1,28,28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /=255
x_test /=255

y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)


model = Sequential()

model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(1,28,28)))
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=10,verbose=1)
score = model.evaluate(x_test, y_test, verbose=0)
print score



