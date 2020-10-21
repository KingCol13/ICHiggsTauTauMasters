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

import pandas as pd
import uproot

from pylorentz import Momentum4
from pylorentz import Position4
from lbn import LBN, LBNLayer

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
tree = uproot.open("/eos/user/d/dwinterb/SWAN_projects/Masters_CP/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]

# define what variables are to be read into the dataframe
momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2"] #subleading neutral pi 4-momentum

other_features = [ "ip_x_1", "ip_y_1", "ip_z_1",        #leading impact parameter
                   "ip_x_2", "ip_y_2", "ip_z_2",        #subleading impact parameter
                   "y_1_1", "y_1_2"]    # ratios of energies

target = [    "aco_angle_1"]  #acoplanarity angle
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2"
             ]

df = tree.pandas.df(momenta_features+other_features+target+selectors)

df = df[
      (df["tau_decay_mode_1"] == 1) 
    & (df["tau_decay_mode_2"] == 1) 
    & (df["mva_dm_1"] == 1) 
    & (df["mva_dm_2"] == 1)
]

num_data = len(df)

#%% Create momenta and boost

# Create our 4-vectors in the lab frame
pi_1_lab = Momentum4(df["pi_E_1"], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
pi_2_lab = Momentum4(df["pi_E_2"], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])

pi0_1_lab = Momentum4(df["pi0_E_1"], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
pi0_2_lab = Momentum4(df["pi0_E_2"], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])

# Create 4-vectors in the ZMF
zmf_momentum = pi_1_lab + pi_2_lab

pi_1_ZMF = pi_1_lab.boost_particle(-zmf_momentum)
pi_2_ZMF = pi_2_lab.boost_particle(-zmf_momentum)

pi0_1_ZMF = pi0_1_lab.boost_particle(-zmf_momentum)
pi0_2_ZMF = pi0_2_lab.boost_particle(-zmf_momentum)

#%% Calculate other targets
# Find the transverse components
pi0_1_trans = np.cross(pi0_1_ZMF[1:,:].transpose(), pi_1_ZMF[1:, :].transpose())
pi0_2_trans = np.cross(pi0_2_ZMF[1:,:].transpose(), pi_2_ZMF[1:, :].transpose())

# Normalise the lambda vectors
pi0_1_trans = pi0_1_trans/np.linalg.norm(pi0_1_trans, ord=2, axis=1, keepdims=True)
pi0_2_trans = pi0_2_trans/np.linalg.norm(pi0_2_trans, ord=2, axis=1, keepdims=True)

#Calculate Phi_ZMF using dot product and arccos
dot = np.sum(pi0_1_trans*pi0_2_trans, axis=1)
phi_0_shift = np.arccos(dot)

# Calculate O
preO = np.cross(pi0_1_trans, pi0_2_trans).transpose()*np.array(pi_2_ZMF[1:, :])
big_O = np.sum(preO, axis=0)
# Shift Phi based on O's sign
phi_1_shift=np.where(big_O<0, phi_0_shift, 2*np.pi-phi_0_shift)

# Shift phi based on energy ratios
y_1 = np.array(df['y_1_1'])
y_2 = np.array(df['y_1_2'])
y_T = np.array(df['y_1_1']*df['y_1_2'])
phi_2_shift=np.where(y_T<0, phi_1_shift, np.where(phi_1_shift<np.pi, phi_1_shift+np.pi, phi_1_shift-np.pi))

#%% Set features and target

#x = tf.convert_to_tensor([pi_1_lab, pi_2_lab, pi0_1_lab, pi0_2_lab], dtype=np.float32)
x = tf.convert_to_tensor([pi_1_ZMF, pi_2_ZMF, pi0_1_ZMF, pi0_2_ZMF], dtype=np.float32)
x = tf.transpose(x, [2, 0, 1])

y = tf.convert_to_tensor(phi_2_shift)
y = tf.reshape(y, (num_data, 1))

#%% Define model

inputs = tf.keras.Input(shape=(2,3))
outputs = tf.linalg.cross(inputs[:,0,:], inputs[:,1,:], name="CrossProduct")

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
history = model.fit(x, y, validation_split=0.3, epochs=1)

#plot traning
plt.figure()
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss on Iteration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()