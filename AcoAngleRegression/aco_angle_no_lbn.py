#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:49:36 2021

@author: kingsley
"""

#%% Initial imports and setup

import uproot 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# stop tensorflow trying to overfill GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
    
#%% Data loading

# loading the tree
tree = uproot.open("ROOTfiles/MVAFILE_AllHiggs_tt_pseudo_phitt.root")["ntuple"]

p1_1 = ["pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1"]
p2_1 = ["pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1"]
p3_1 = ["pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1"]

p1_2 = ["pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2"]
p2_2 = ["pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2"]
p3_2 = ["pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2"]

other_features = [ "ip_x_1", "ip_y_1", "ip_z_1",        #leading impact parameter
                   "ip_x_2", "ip_y_2", "ip_z_2",        #subleading impact parameter
                   "y_1_1", "y_1_2"]    # ratios of energies

target = [    "aco_angle_1"]  #acoplanarity angle
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2"
             ]

leading_prongs = p1_1+p2_1+p3_1
subleading_prongs = p1_2+p2_2+p3_2

df = tree.pandas.df(leading_prongs+subleading_prongs+other_features+target+selectors)

num_data = len(df)

print("Loaded data.")

#%% Selection

#pi-pi
#df = df[(df['mva_dm_1'] == 0) & (df['mva_dm_2'] == 0)]

#rho-rho:
#df = df[(df['mva_dm_1'] == 1) & (df['mva_dm_2'] == 1) & (df['tau_decay_mode_1'] == 1) & (df['tau_decay_mode_2'] == 1)]

#a1-a1 (a1->pi+2pi0)
#df = df[(df['mva_dm_1'] == 2) & (df['mva_dm_2'] == 2) & (df['tau_decay_mode_1'] == 1) & (df['tau_decay_mode_2'] == 1)]

#a1-a1 (a1->3pi))
df = df[(df['mva_dm_1'] == 10) & (df['mva_dm_2'] == 10)]

#a1-a1 combined
#df = df[((df['mva_dm_1'] == 2) | (df['mva_dm_1'] == 10)) & ((df['mva_dm_2'] == 2) | (df['mva_dm_2'] == 10))]

#%% Cleanup

df = df.dropna()

#%% Features and targets
x = tf.convert_to_tensor(df[leading_prongs+subleading_prongs+other_features], dtype=tf.float32)

y = tf.convert_to_tensor(df[target], dtype=tf.float32)

#%% Describe and compile model

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['mae'])

print("Model compiled.")

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

#%% Evaluating model

model.evaluate(x, y)