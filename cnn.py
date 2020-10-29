#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 17:44:19 2020

@author: patxi
"""


import numpy as np
import pandas as pd
import tensorflow as tf
import math
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import InputLayer, Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from shutil import rmtree

from autoencoder import load_UCR_data, normalize



path_UCR = '/home/patxi/Documents/UCRArchive_2018/'
# folder   = 'ChlorineConcentration'
folder   = 'ItalyPowerDemand'
path_UCR = '/home/patxi/Documents/UCRArchive_2018/'
dataset_train, dataset_test = load_UCR_data(path_UCR, folder)




n_select = 1

x_train = dataset_train.iloc[:, 1:].to_numpy()[:, ::n_select]    
x_test  = dataset_test.iloc[:, 1:].to_numpy()[:, ::n_select]  
y_train = dataset_train.iloc[:, 0].to_numpy()
y_test  = dataset_test.iloc[:, 0].to_numpy()

x_train = dataset_test.iloc[:, 1:].to_numpy()[:, ::n_select]    
x_test  = dataset_train.iloc[:, 1:].to_numpy()[:, ::n_select]  
y_train = dataset_test.iloc[:, 0].to_numpy()
y_test  = dataset_train.iloc[:, 0].to_numpy()

print('x_train.shape', x_train.shape)
print('x_test.shape' , x_test.shape)
print('y_train.shape', y_train.shape)
print('y_test.shape' , y_test.shape)


plt.plot(x_train.transpose())
plt.show()
for clase in np.unique(y_train):
    plt.plot(x_train[y_train==clase].transpose())
    plt.show()
x_train = normalize(x_train)
x_test = normalize(x_test)
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test  = le.transform(y_test)

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)
input_dim         = x_train.shape[-1]
number_of_classes = le.classes_.shape[-1]


class CNN:
    """
    """

    model_name = "CNN"

    def __init__(
        self,
        input_dim,
        number_of_classes,
        ):
        """
        Parameters
        ----------
        x_shape : tuple
            Shape of the input dataset: (num_samples, num_timesteps, num_channels)

        """
        self.input_dim         = input_dim
        self.number_of_classes = number_of_classes

    def define_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.input_dim, 1)))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.number_of_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], )
        return model



# prepare data for conv model
x_train = np.expand_dims(x_train, axis=-1)
x_test  = np.expand_dims(x_test, axis=-1)


cnn   = CNN(input_dim, number_of_classes)
model = cnn.define_model()

# fit network
rmtree('./tmp/conv', ignore_errors=True)
model.fit(x_train,
          y_train,
           validation_split=0.2,
          # validation_data=(x_test, y_test),
          epochs=1000,
          batch_size=None,
          verbose=1,
          callbacks=[
              TensorBoard(log_dir='./tmp/conv'),
              EarlyStopping(patience=15),
              ],
          )
# evaluate model
_, accuracy = model.evaluate(x_test, y_test, batch_size=None, verbose=0)
print("TEST ACC: ", accuracy)
print(confusion_matrix(y_test.argmax(axis=1), model.predict(x_test).argmax(axis=1)))
