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
phi_shift_0 = np.arccos(dot)

# Calculate O
preO = np.cross(pi0_1_trans, pi0_2_trans).transpose()*np.array(pi_2_ZMF[1:, :])
big_O = np.sum(preO, axis=0)
# Shift Phi based on O's sign
phi_shift_1=np.where(big_O<0, phi_shift_0, 2*np.pi-phi_shift_0)

# Shift phi based on energy ratios
y_1 = np.array(df['y_1_1'])
y_2 = np.array(df['y_1_2'])
y_tau = np.array(df['y_1_1']*df['y_1_2'])
phi_shift_2=np.where(y_tau<0, phi_shift_1, np.where(phi_shift_1<np.pi, phi_shift_1+np.pi, phi_shift_1-np.pi))

#%% Set features and target

#x = tf.convert_to_tensor([pi_1_lab, pi_2_lab, pi0_1_lab, pi0_2_lab], dtype=np.float32)
x = tf.convert_to_tensor([pi_1_ZMF, pi_2_ZMF, pi0_1_ZMF, pi0_2_ZMF], dtype=np.float32)
x = tf.transpose(x, [2, 0, 1])

#y = tf.convert_to_tensor(pi0_1_trans, dtype="float32")

y = tf.convert_to_tensor(phi_shift_2, dtype="float32")
y = tf.reshape(y, (num_data, 1))
# Final goal:
#y = tf.convert_to_tensor(phi_shift_2)
#y = tf.reshape(y, (num_data, 1))

#%% Define model

tf_zmf_momenta = tf.keras.Input(shape=(4,4))

#Can I combine these 2 somehow?:
tf_pi0_1_trans = tf.math.l2_normalize(tf.linalg.cross(tf_zmf_momenta[:,2,1:], tf_zmf_momenta[:,0,1:]), axis=1)
tf_pi0_2_trans = tf.math.l2_normalize(tf.linalg.cross(tf_zmf_momenta[:,3,1:], tf_zmf_momenta[:,1,1:]), axis=1)

tf_phi_shift_0 = tf.math.acos(tf.reduce_sum(tf.math.multiply(tf_pi0_1_trans, tf_pi0_2_trans), axis=1))

tf_big_O = tf.math.reduce_sum(tf.math.multiply(tf.linalg.cross(tf_pi0_1_trans, tf_pi0_2_trans), tf_zmf_momenta[:,1,1:]), axis=1)

tf_phi_shift_1 = tf.where(tf_big_O<0, tf_phi_shift_0, 2*np.pi-tf_phi_shift_0)

tf_y_1 = (tf_zmf_momenta[:,0,0] - tf_zmf_momenta[:,2,0])/(tf_zmf_momenta[:,0,0] + tf_zmf_momenta[:,2,0])
yf_y_2 = (tf_zmf_momenta[:,1,0] - tf_zmf_momenta[:,3,0])/(tf_zmf_momenta[:,1,0] + tf_zmf_momenta[:,3,0])

tf_y_tau = tf_y_1*yf_y_2

tf_phi_shift_2 = tf.where(tf_y_tau<0, tf_phi_shift_1, tf.where(tf_phi_shift_1<np.pi, tf_phi_shift_1+np.pi, tf_phi_shift_1-np.pi))

model = tf.keras.Model(inputs=tf_zmf_momenta, outputs=tf_phi_shift_2)
print("Model defined.")

#%% Compile model

loss_fn = tf.keras.losses.MeanSquaredError()
#cosine similarity for only caring about direction:
#loss_fn = tf.keras.losses.CosineSimilarity()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['mae'])

print("Model compiled.")

#%% Evaluate model

model.evaluate(x, y)

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