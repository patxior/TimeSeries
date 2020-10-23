#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 09:13:03 2020

@author: patxi
"""

import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
    

def y():
    x = np.arange(0,25.6,0.2)
    return np.array([math.sin(i) for i in x])+x/10+np.random.rand(len(x))

def normalize(X):
    max_value = X.max()
    min_value = X.min()
    X = X-min_value
    X = X/(max_value-min_value)
    return X

def create_data(n_train=50, n_test=10, show=False):
    x_train = np.array([y() for _ in range(n_train)])
    x_test = np.array([y() for _ in range(n_test)])
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    if show==True:
        plt.plot(x_train.transpose(), color='red', alpha=0.1)
        plt.plot(x_test.transpose(), color='blue', alpha=0.1)
        plt.show()
    return [x_train, x_test]

def define_autoencoder(input_dim, encoded_dim):
    # AUTOENCODER
    input_data = tf.keras.Input(shape=(input_dim,))
    encoded    = tf.keras.layers.Dense(encoded_dim*4, activation='relu')(input_data)
    encoded    = tf.keras.layers.Dense(encoded_dim*2, activation='relu')(encoded)
    encoded    = tf.keras.layers.Dense(encoded_dim, activation='relu')(encoded)
    decoded    = tf.keras.layers.Dense(encoded_dim*2, activation='relu')(encoded)
    decoded    = tf.keras.layers.Dense(encoded_dim*4, activation='relu')(decoded)
    decoded    = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = tf.keras.Model(input_data, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # ENCODER
    encoder = tf.keras.Model(input_data, encoded)
    # DECODER
    encoded_input = tf.keras.Input(shape=(encoded_dim,))
    deco = autoencoder.layers[-3](encoded_input)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    # create the decoder model
    decoder = tf.keras.Model(encoded_input, deco)
    return autoencoder, encoder, decoder



if __name__ == "__main__":
    x_train, x_test = create_data(show=True)
    input_dim   = x_train.shape[-1]
    encoded_dim = 8
    autoencoder, encoder, decoder = define_autoencoder(input_dim, encoded_dim)
    autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=16,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.TensorBoard(log_dir='./tmp/autoencoder'),
                        tf.keras.callbacks.EarlyStopping(patience=5),
                        ],
                    )
    
    plt.plot(autoencoder.predict(x_test).transpose()); plt.show()