#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:19:12 2020

Investigating making a network learn the cross product

@author: kingsley
"""

#%% Imports

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#%% Create features and targets

n_data = int(1e6)

x = np.random.rand(n_data, 2, 3)
y = np.cross(x[:,0,:], x[:,1,:])

#%% Define and compile model

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten( input_shape=(2,3)),
    tf.keras.layers.Dense(16, activation='sigmoid'),
    tf.keras.layers.Dense(16, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3),
])

loss_fn = tf.keras.losses.MeanSquaredError()
#cosine similarity for only caring about direction:
#loss_fn = tf.keras.losses.cosine_similarity()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['mae'])

print("Model compiled.")

#%% Training model

#train model
history = model.fit(x, y, validation_split=0.3, epochs=25)

#plot traning
plt.figure()
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss on Iteration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()