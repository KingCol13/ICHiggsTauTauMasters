#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:23:33 2020

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

#%% Define model

inputs = tf.keras.Input(shape=(2,3))
#flat = tf.keras.layers.Flatten()(inputs)
#dense = tf.keras.layers.Dense(64, activation='relu',)(flat)
#outputs = tf.keras.layers.Dense(3)(dense)
outputs = tf.linalg.cross(inputs[:,0,:], inputs[:,1,:])

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
print("Model defined.")

#%% Compile model

loss_fn = tf.keras.losses.MeanSquaredError()
#cosine similarity for only caring about direction:
#loss_fn = tf.keras.losses.CosineSimilarity()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['mae'])

print("Model compiled.")
#%% Training model

#train model
history = model.fit(x, y, validation_split=0.3, epochs=2)

#plot traning
plt.figure()
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss on Iteration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()