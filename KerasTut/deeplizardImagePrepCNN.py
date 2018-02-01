

#deeplizard https://www.youtube.com/watch?v=LhEMXbjGV_4&t=301s

import os

import numpy as np
import sklearn


# print(keras.__path__)


from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_path = 'data/train'
valid_path = 'data/valid'
test_path = 'data/test'

stringOnes = 'onesBinary'
stringThrees = 'threesBinary'

# OBS big batch size = works better
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(28, 28),
                                                         classes=[stringOnes, stringThrees], batch_size=10,
                                                         color_mode='grayscale'
                                                         )
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(28, 28),
                                                         classes=[stringOnes, stringThrees], batch_size=20,
                                                         color_mode='grayscale'
                                                         )
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(28, 28),
                                                         classes=[stringOnes, stringThrees], batch_size=20,
                                                         color_mode='grayscale',
                                                         shuffle=False
                                                         )


# def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
#     if type(ims[0]) is np.ndarray:
#         ims = np.array(ims).astype(np.uint8)
#         if (ims.shape[-1] != 3):
#             ims = ims.transpose((0,2,3,1))
#     f = plt.figure(figsize=figsize)
#     cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
#     for i in range(len(ims)):
#         sp = f.add_subplot(rows, cols, i+1)
#         sp.axis('Off')
#         if titles is not None:
#             sp.set_title(titles[i], fontsize=16)
#         plt.imshow(ims[i], interpolation=None if interp else 'none')
#
#
# imgs, labels = next(train_batches)
# plots(imgs, titles=labels)
# plt.show()


# HELLO_WORLD CNN
model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)),
                    Flatten(),
                    Dense(2, activation='softmax')
                    ])

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, validation_data=valid_batches, epochs=10, verbose=2)

pred2 = model.evaluate_generator(test_batches) # returns loss and accuracy





#
#
# # hello AlexNet
# model = Sequential()
# model.add(Conv2D(6, (5, 5), input_shape=(1, 50, 50)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2)))
#
# model.add(Conv2D(16, (5, 5), padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2)))
#
# model.add(Conv2D(120, (5, 5)))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))
#
# # model.add(Flatten())
# # model.add(Dense(2))
# # model.add(Activation('softmax'))
#
# model.add(Flatten())
# model.add(Dense(16)) # this requires 'channels_last' in config file
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# nb_epoch = 30
# model.fit_generator(train_batches, verbose=2, validation_data=valid_batches, epochs=nb_epoch)
# # model.fit(train_batches, epochs=nb_epoch, verbose=1, validation_data=valid_batches)
#



# VGG_16
#
# from KerasTut import vgg16
# vgg16_model = vgg16.VGG16()





