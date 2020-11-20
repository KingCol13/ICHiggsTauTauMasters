#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:43:07 2020

@author: kingsley
"""

#%% imports

import uproot 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pylorentz import Momentum4
from pylorentz import Position4
from lbn_modified import LBN, LBNLayer

#%% Data loading

tree = uproot.open("MVAFILE_AllHiggs_tt.root")["ntuple"]

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

neutrino_features = [  "gen_nu_p_1", "gen_nu_p_2",
                       "gen_nu_phi_1", "gen_nu_phi_2",
                       "gen_nu_eta_1", "gen_nu_eta_2"]

met_features = ["met", "metx", "mety"]

df = tree.pandas.df(momenta_features+other_features+target+selectors+neutrino_features+met_features)

df = df[
      (df["tau_decay_mode_1"] == 1) 
    & (df["tau_decay_mode_2"] == 1) 
    & (df["mva_dm_1"] == 1) 
    & (df["mva_dm_2"] == 1)
]

df = df[df["gen_nu_p_1"] > -4000]

#%% Create momenta and boost

# Create our 4-vectors in the lab frame
pi_1_lab = Momentum4(df["pi_E_1"], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
pi_2_lab = Momentum4(df["pi_E_2"], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])

pi0_1_lab = Momentum4(df["pi0_E_1"], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
pi0_2_lab = Momentum4(df["pi0_E_2"], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])

# Create 4-vectors in the ZMF
zmf_momentum = pi_1_lab + pi_2_lab
boost = Momentum4(zmf_momentum[0], -zmf_momentum[1], -zmf_momentum[2], -zmf_momentum[3])

pi_1_ZMF = pi_1_lab.boost_particle(boost)
pi_2_ZMF = pi_2_lab.boost_particle(boost)

pi0_1_ZMF = pi0_1_lab.boost_particle(boost)
pi0_2_ZMF = pi0_2_lab.boost_particle(boost)

print("Boosted particles.")

#%% Calculate other targets
# Find the transverse components
pi0_1_trans = np.cross(pi0_1_ZMF[1:,:].transpose(), pi_1_ZMF[1:, :].transpose())
pi0_2_trans = np.cross(pi0_2_ZMF[1:,:].transpose(), pi_2_ZMF[1:, :].transpose())

# Normalise the lambda vectors
pi0_1_trans = pi0_1_trans/np.linalg.norm(pi0_1_trans, ord=2, axis=1, keepdims=True)
pi0_2_trans = pi0_2_trans/np.linalg.norm(pi0_2_trans, ord=2, axis=1, keepdims=True)

#Calculate Phi_ZMF using dot product and arccos
dot = np.sum(pi0_1_trans*pi0_2_trans, axis=1)
phi_shift_0 = np.arccos(dot)

# Calculate O
preO = np.cross(pi0_1_trans, pi0_2_trans).transpose()*np.array(pi_2_ZMF[1:, :])
big_O = np.sum(preO, axis=0)
# Shift Phi based on O's sign
phi_shift_1=np.where(big_O>0, phi_shift_0, 2*np.pi-phi_shift_0)

# Shift phi based on energy ratios
y_1 = np.array(df['y_1_1'])
y_2 = np.array(df['y_1_2'])
y_tau = np.array(df['y_1_1']*df['y_1_2'])
phi_shift_2=np.where(y_tau<0, phi_shift_1, np.where(phi_shift_1<np.pi, phi_shift_1+np.pi, phi_shift_1-np.pi))

#%% Features and targets

def normalise(x):
    return (x-tf.math.reduce_mean(x, axis=0))/tf.math.reduce_std(x, axis=0)

#add visible product features
x = tf.convert_to_tensor([pi0_1_ZMF, pi_1_ZMF, pi0_2_ZMF, pi_2_ZMF], dtype=tf.float32)
x = tf.transpose(x, [2, 0, 1])
x = tf.reshape(x, (x.shape[0], 16))

#add met features
x = tf.concat([x, tf.convert_to_tensor(df[met_features], dtype=tf.float32)], axis=1)

y = tf.convert_to_tensor(df[neutrino_features], dtype=tf.float32)


#normalise
x = normalise(x)
y = normalise(y)

#%% Building network

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation="relu"),
    #tf.keras.layers.Dense(200, activation="relu"),
    #tf.keras.layers.Dense(200, activation="relu"),
    #tf.keras.layers.Dense(200, activation="relu"),
    tf.keras.layers.Dense(6)
])

loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['mae'])

print("Model compiled.")


#%% Training model

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

#%% Histogram

pred = model(x)[:,0]
true = y[:,1]

plt.figure()
plt.title("Neural Network Performance")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.hist(pred, bins = 100, alpha = 0.5, label="Predicted")
plt.hist(true, bins = 100, alpha = 0.5, label="True")
plt.xlabel("phi_cp_unshifted")
plt.grid()
plt.legend()
plt.show()