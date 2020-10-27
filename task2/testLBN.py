#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 01:55:02 2020

@author: kingsley

Trying to make LBN work as expected
"""

#%% Initial imports and setup

import uproot 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pylorentz import Momentum4
from pylorentz import Position4
from klbn import LBN, LBNLayer

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

print("Loaded data.")
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

print("Boosted particles.")
#%% Features and targets

x = tf.convert_to_tensor([pi_1_lab, pi_2_lab, pi0_1_lab, pi0_2_lab], dtype=np.float32)
x = tf.transpose(x, [2, 0, 1])

y = tf.convert_to_tensor([pi_1_ZMF, pi_2_ZMF, pi0_1_ZMF, pi0_2_ZMF], dtype=np.float32)
#weird order from LBN
y = tf.transpose(y, [2, 1, 0])

#%% Building network

#features for LBN output
LBN_output_features = ["E", "px", "py", "pz"]

#define our LBN layer:
myLBNLayer = LBNLayer((4, 4), 4, n_restframes=1, boost_mode=LBN.PRODUCT, features=LBN_output_features, abs_restframe_weights=False, abs_particle_weights=False)

#set the weights
weights = [np.eye(4), np.reshape(np.array([1, 1, 0, 0], dtype=np.float32), (4,1))]
myLBNLayer.set_weights(weights)

#define NN model and compile
model = tf.keras.models.Sequential([
    myLBNLayer,
    tf.keras.layers.Reshape((4,4)),
])

loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['mae'])

print("Model compiled.")

#%% Evaluating model

model.evaluate(x, y)

#%%

def get_m(momentum):
    return np.sqrt(momentum[0]*momentum[0] - momentum[1]*momentum[1] - momentum[2]*momentum[2] - momentum[3]*momentum[3])