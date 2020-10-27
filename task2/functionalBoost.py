#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:27:05 2020

@author: kingsley

Work out the tensorflow maths required to boost particles into a rest frame
"""

#%% Initial imports and setup

import uproot 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

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

#%% Features and targets

x = tf.convert_to_tensor([pi_1_lab, pi_2_lab, pi0_1_lab, pi0_2_lab], dtype=np.float32)
x = tf.transpose(x, [2, 0, 1])

y = tf.convert_to_tensor([pi_1_ZMF, pi_2_ZMF, pi0_1_ZMF, pi0_2_ZMF], dtype=np.float32)
y = tf.transpose(y, [2, 0, 1])
#%% Define model

tf_lab_momenta = tf.keras.Input(shape=(4,4))

tf_zmf_momentum = tf_lab_momenta[:,0] + tf_lab_momenta[:,1]

#optimise this with matrices?
m = tf.math.sqrt(tf_zmf_momentum[:,0]*tf_zmf_momentum[:,0] - tf.math.reduce_sum(tf.square(tf_zmf_momentum[:,1:]), axis=1))
gamma = -tf.expand_dims(tf.expand_dims(tf_zmf_momentum[:,0]/m, axis=-1), -1)
beta = tf.math.sqrt(1 - 1 / (gamma*gamma))

e = tf.expand_dims(tf.concat((tf.ones_like(tf.expand_dims(m, 1)), -tf.linalg.normalize(tf_zmf_momentum[:,1:], axis=1)[0]), axis=1), -1)
e_T = tf.transpose(e, [0, 2, 1])

I = tf.constant(tf.eye(4,4), tf.float32, shape=(1, 4, 4))
U = tf.constant([[-1, 0, 0, 0]] + 3 * [[0, -1, -1, -1]], tf.float32, shape=(1, 4, 4))
                               
Lambda = I + (U + gamma) * ((U + 1) * beta - U) * (e * e_T)

tf_zmf_momenta = tf.transpose(tf.matmul(Lambda, tf_lab_momenta, transpose_b = True), [0, 2, 1])

test_out = e

model = tf.keras.Model(inputs=tf_lab_momenta, outputs=tf_zmf_momenta) #outputs=tf_zmf_momenta)
#%% Compile model

loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['mae'])

#%% Evaluate
