#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:16:39 2020

@author: kingsley
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import customLayers as cl

#%% features and target

x = tf.convert_to_tensor(np.random.rand(100000, 2))
y = tf.convert_to_tensor(x[:,0]+x[:,1])

#%% Model definition

input = tf.keras.Input(shape=(2,))
dense = cl.Linear(32)(input)
#dense = tf.keras.layers.Dense(32)(input)
output = tf.keras.layers.Dense(2)(dense)

model = tf.keras.Model(inputs=input, outputs=output)

#%% Compile model

loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['mae'])

#%% Training model

#train model
history = model.fit(x, y, validation_split=0.3, epochs=5)

#plot traning
plt.figure()
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss on Iteration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
