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

tree = uproot.open("MVAFILE_AllHiggs_tt.root")["ntuple"]

momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2"] #subleading neutral pi 4-momentum

target = [    "aco_angle_1"]  #acoplanarity angle
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2"
             ]

neutrino_features = [  "gen_nu_p_1", "gen_nu_p_2",
                       "gen_nu_phi_1", "gen_nu_phi_2",
                       "gen_nu_eta_1", "gen_nu_eta_2"]

met_features = ["met", "metx", "mety"]#, "metCov00", "metCov01", "metCov10", "metCov11"]

sv_features = ["sv_x_1", "sv_y_1", "sv_z_1",
               "sv_x_2", "sv_y_2", "sv_z_2"]
"""
               "svcov00_1", "svcov01_1", "svcov02_1",
               "svcov10_1", "svcov11_1", "svcov12_1",
               "svcov20_1", "svcov21_1", "svcov22_1",
               "svcov00_2", "svcov01_2", "svcov02_2",
               "svcov10_2", "svcov11_2", "svcov12_2",
               "svcov20_2", "svcov21_2", "svcov22_2"]
"""
    
ip_features = ["ip_x_1", "ip_y_1", "ip_z_1", "ip_x_2", "ip_y_2", "ip_z_2"]
"""
               "ipcov00_1", "ipcov01_1", "ipcov02_1", "ipcov10_1", "ipcov11_1", "ipcov12_1",
               "ipcov20_1", "ipcov21_1", "ipcov22_1", "ipcov00_2", "ipcov01_2", "ipcov02_2",
               "ipcov10_2", "ipcov11_2", "ipcov12_2", "ipcov20_2", "ipcov21_2", "ipcov22_2"]
"""
    
phi_cp_feature = ["gen_phitt"]

df = tree.pandas.df(momenta_features+target+selectors
                    +neutrino_features+met_features+sv_features+sv_features+ip_features
                    +phi_cp_feature)

df = df[
      (df["tau_decay_mode_1"] == 1) 
    & (df["tau_decay_mode_2"] == 1) 
    & (df["mva_dm_1"] == 1) 
    & (df["mva_dm_2"] == 1)
]

#Filter out -9999 values for neutrino momenta
df = df[df["gen_nu_p_1"] > -4000]
df = df[df["gen_nu_p_2"] > -4000]

#%% Create momenta and boost

# Create our 4-vectors in the lab frame
pi_1_lab = Momentum4(df["pi_E_1"], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
pi_2_lab = Momentum4(df["pi_E_2"], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])

pi0_1_lab = Momentum4(df["pi0_E_1"], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
pi0_2_lab = Momentum4(df["pi0_E_2"], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])

rho_1_lab = pi_1_lab + pi0_1_lab
rho_2_lab = pi_2_lab + pi0_2_lab

#Neutrinos
nu_1_lab = Momentum4.e_m_eta_phi(df["gen_nu_p_1"], 0, df["gen_nu_eta_1"], df["gen_nu_phi_1"])
nu_2_lab = Momentum4.e_m_eta_phi(df["gen_nu_p_2"], 0, df["gen_nu_eta_2"], df["gen_nu_phi_2"])

#met, ip, sv
df_len = len(df)
zeros = np.zeros(df_len)
met_lab =  Position4(zeros, df['metx']  ,df['mety']  , zeros)
#sv_1_lab = Position4(zeros, df['sv_x_1'], df['sv_y_1'], df['sv_z_1'])
#sv_2_lab = Position4(zeros, df['sv_x_2'], df['sv_y_2'], df['sv_z_2'])
ip_1_lab = Position4(zeros,  df['ip_x_1'], df['ip_y_1'], df['ip_z_1'])
ip_2_lab = Position4(zeros,  df['ip_x_2'], df['ip_y_2'], df['ip_z_2'])

# Find the boost to visible product ZNF
zmf_momentum = pi_1_lab + pi_2_lab + pi0_1_lab + pi0_2_lab
boost = Momentum4(zmf_momentum[0], -zmf_momentum[1], -zmf_momentum[2], -zmf_momentum[3])

# Calculate 4-vectors in the ZMF
pi_1_ZMF = pi_1_lab.boost_particle(boost)
pi_2_ZMF = pi_2_lab.boost_particle(boost)

pi0_1_ZMF = pi0_1_lab.boost_particle(boost)
pi0_2_ZMF = pi0_2_lab.boost_particle(boost)

nu_1_ZMF = nu_1_lab.boost_particle(boost)
nu_2_ZMF = nu_2_lab.boost_particle(boost)

met_ZMF = met_lab.boost_particle(boost)
#sv_1_ZMF = sv_1_lab.boost_particle(boost)
#sv_2_ZMF = sv_2_lab.boost_particle(boost)
ip_1_ZMF = ip_1_lab.boost_particle(boost)
ip_2_ZMF = ip_2_lab.boost_particle(boost)

print("Boosted particles.")

#%% Features and targets

def normalise(x):
    return (x-tf.math.reduce_mean(x, axis=0))/tf.math.reduce_std(x, axis=0)

#add visible product features
x = tf.convert_to_tensor([pi0_1_lab, pi_1_lab, pi0_2_lab, pi_2_lab], dtype=tf.float32)
x = tf.transpose(x, [2, 0, 1])
x = tf.reshape(x, (x.shape[0], 16))

"""
#invariant masses
mass_features = np.real([pi0_1_lab.m, pi_1_lab.m, pi0_2_lab.m, pi_2_lab.m, rho_1_lab.m, rho_2_lab.m])
mass_features = tf.convert_to_tensor(mass_features, dtype=tf.float32)
mass_features = tf.transpose(mass_features, [1, 0])
x = tf.concat([x, mass_features], axis=1)
"""



#add met features
x = tf.concat([x, tf.convert_to_tensor(df[met_features], dtype=tf.float32)], axis=1)

#add sv features
#x = tf.concat([x, tf.convert_to_tensor(df[sv_features], dtype=tf.float32)], axis=1)

#add ip features
x = tf.concat([x, tf.convert_to_tensor(df[ip_features], dtype=tf.float32)], axis=1)

#y = tf.convert_to_tensor(df[neutrino_features], dtype=tf.float32)
y = tf.convert_to_tensor([nu_1_lab.phi, nu_2_lab.phi], dtype=tf.float32)
y = tf.transpose(y, [1, 0])

#normalise
#x = normalise(x)
#y = normalise(y)

#remove NaNs, TODO: improve this
x = tf.where(tf.math.is_nan(x), 0, x)

#%% Prototype on reduced dataset and split:
"""
prototype_num = 100000
x = x[:prototype_num]
y = y[:prototype_num]
"""

trainFrac = 0.7
numTrain = int(trainFrac*x.shape[0])
x_train = x[:numTrain]
y_train = y[:numTrain]

x_val = x[numTrain:]
y_val = y[numTrain:]

#%% Building network

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(64, activation="relu"),#, kernel_regularizer=tf.keras.regularizers.L1L2(0.01, 0.1)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),#, kernel_regularizer=tf.keras.regularizers.L1L2(0.01, 0.1)),
    #tf.keras.layers.Dense(200, activation="relu"),
    #tf.keras.layers.Dense(200, activation="relu"),
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2)
])

opt = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['mae', 'mse'])

print("Model compiled.")


#%% Training model

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5)

#plot traning
plt.figure()
plt.plot(history.history['mse'], label="Training MSE")
plt.plot(history.history['val_mse'], label="Validation MSE")
plt.title("Loss on Iteration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%% Histograms

res = np.array(model(x_val) - y_val)
plt.figure()
plt.title("NN Predicted Minus True Phi")
plt.xlabel("Phi / Radians")
plt.ylabel("Frequency")
#plt.xlim(-100, 100)
#plt.xlim(-5, 5)
plt.hist(res[:,0], bins = 100, alpha = 0.5, label="phi_1 mean={:.2f}, std={:.2f}".format(np.mean(res[:,0]), np.std(res[:,0])))
plt.hist(res[:,1], bins = 100, alpha = 0.5, label="phi_1 mean={:.2f}, std={:.2f}".format(np.mean(res[:,1]), np.std(res[:,1])))
plt.grid()
plt.legend(loc="upper right")
plt.show()

#%% Scatter plot

plt.figure()
plt.title("Leading Neutrino Pz")
plt.xlabel("Prediction / GeV")
plt.ylabel("True / GeV")
plt.hist2d(res[:,0], y_val[:,0], 1000)
plt.show()


#%% Evaluate

print(tf.math.reduce_mean(tf.math.abs(model(x)-y), axis=0)/tf.math.reduce_std(y, axis=0))
