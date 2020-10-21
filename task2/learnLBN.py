# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
#%% Lorentz boosts

# Create our 4-vectors in the lab frame
pi_1 = Momentum4(df["pi_E_1"], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
pi_2 = Momentum4(df["pi_E_2"], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])

IP1 = Momentum4(df["pi0_E_1"], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
IP2 = Momentum4(df["pi0_E_2"], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])

# Create 4-vectors in the ZMF
pi_T4M = pi_1 + pi_2

pi1_ZMF = pi_1.boost_particle(-pi_T4M)
pi2_ZMF = pi_2.boost_particle(-pi_T4M)

IP1_ZMF = IP1.boost_particle(-pi_T4M)
IP2_ZMF = IP2.boost_particle(-pi_T4M)

#%% Calculate other targets
# Find the transverse components
IP1_trans = np.cross(IP1_ZMF[1:,:].transpose(), pi1_ZMF[1:, :].transpose())
IP2_trans = np.cross(IP2_ZMF[1:,:].transpose(), pi2_ZMF[1:, :].transpose())

# Normalise the vectors
IP1_trans = IP1_trans/np.linalg.norm(IP1_trans, ord=2, axis=1, keepdims=True)
IP2_trans = IP2_trans/np.linalg.norm(IP2_trans, ord=2, axis=1, keepdims=True)

#Calculate Phi_ZMF using dot product and arccos
dot = np.sum(IP1_trans*IP2_trans, axis=1)
Phi_ZMF = np.arccos(dot)

# Calculate O
preO = np.cross(IP1_trans, IP2_trans).transpose()*np.array(pi2_ZMF[1:, :])
big_O = np.sum(preO, axis=0)
# Shift Phi based on O's sign
Phi_Shifted=np.where(big_O<0, Phi_ZMF, 2*np.pi-Phi_ZMF)

# Shift phi based on energy ratios
y_1 = np.array(df['y_1_1'])
y_2 = np.array(df['y_1_2'])
y_T = np.array(df['y_1_1']*df['y_1_2'])
Phi_Shifted=np.where(y_T<0, Phi_Shifted, np.where(Phi_Shifted<np.pi, Phi_Shifted+np.pi, Phi_Shifted-np.pi))
#%% Creating features and targets

# Create x and y tensors
x = tf.convert_to_tensor([pi1_ZMF, pi2_ZMF, IP1_ZMF, IP2_ZMF], dtype=np.float32)
x  =  tf.transpose(x, [2, 0, 1])

y_1 = y_1.reshape(num_data, 1)
y_2 = y_2.reshape(num_data, 1)
y = np.concatenate((IP1_trans, IP2_trans, y_1, y_2),axis=1)#, pi2_ZMF[1:].transpose()), axis=1)
y = tf.convert_to_tensor(y, dtype=np.float32)
#y =  tf.transpose(y, [2, 0, 1])
#y = np.array([pi1_ZMF, pi2_ZMF, IP1_ZMF, IP2_ZMF], dtype=np.float32).transpose()
#normalise y:
#y = (y-np.mean(y, axis=0))/np.std(y, axis=0)

#%% Building network

#features for LBN output
LBN_output_features = ["E", "px", "py", "pz"]

#define our LBN layer:
#myLBNLayer = LBNLayer((4, 4), 11, boost_mode=LBN.PAIRS, features=LBN_output_features)

#define NN model and compile
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten( input_shape=(4,4)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(8)
])

loss_fn = tf.keras.losses.MeanSquaredError()
model1.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['mae'])

print("Model compiled.")
#%% Training model

#train model
history = model1.fit(x, y, validation_split=0.3, epochs=25)

#plot traning
plt.figure()
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss on Iteration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()