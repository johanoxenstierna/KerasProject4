import matplotlib.pyplot as plt
import numpy as np
# import os
from keras.datasets import mnist

# os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils




batch_size = 128
nb_classes = 10

img_rows, img_cols = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Reshape
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print("One hot encoding: {}".format(Y_train[0, :]))

# # PLOT SOME IMAGES
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(X_train[i, 0], cmap='gray')
#     plt.show()
#     plt.axis('off')

model = Sequential()
model.add(Convolution2D(6, kernel_size=(5,5), input_shape=(1, img_rows, img_cols), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Convolution2D(16, kernel_size=(5, 5), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Convolution2D(120, kernel_size=(5, 5)))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(84)) # this requires 'channels_last' in config file
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

nb_epoch = 1

model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
         verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)














