


#deeplizard https://www.youtube.com/watch?v=LhEMXbjGV_4&t=301s

import os

import numpy as np
from numpy.random import seed
seed(1) # forget about reproducibility
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

from keras import callbacks


train_path = 'data3/train'
valid_path = 'data3/valid'
test_path = 'data3/test'

s1 = 'ones'
s2 = 'twos'
s0 = 'zeros'



m_color_mode = 'grayscale'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(61, 85), #x, y
                                                         classes=[s1, s2, s0], batch_size=200,
                                                         color_mode=m_color_mode, shuffle=True
                                                         )
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(61, 85),
                                                         classes=[s1, s2, s0], batch_size=100,
                                                         color_mode=m_color_mode, shuffle=True
                                                         )
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(61, 85),
                                                        classes=[s1, s2, s0], batch_size=200,
                                                         color_mode=m_color_mode, shuffle=False
                                                         )

#
# def plots(ims, figsize=(85, 155), rows=1, interp=False, titles=None):
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


m_callback = callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

# model = Sequential([Conv2D(32, (4, 4), activation='relu', input_shape=(1,61,85)),
#                     Flatten(),
#                     Dense(3, activation='softmax')
#                     ])

model = Sequential([Conv2D(6, (4, 4), activation='relu', input_shape=(1,61,85)),
                    MaxPooling2D(2,2),
                    Conv2D(16, (4, 4), padding='same'),
                    MaxPooling2D(2, 2),
                    Dropout(0.5),
                    Flatten(),
                    Dense(3, activation='softmax')
                    ])





model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches,
                    validation_data=valid_batches,
                    epochs=20,
                    verbose=2, #verbose=1 gives more prints
                    shuffle=False,
                    callbacks=[m_callback]
                    )



pred = model.evaluate_generator(test_batches) # returns loss and accuracy

print(pred)




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





