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

sv_features = ["sv_x_1", "sv_y_1", "sv_z_1", "sv_x_2", "sv_y_2", "sv_z_2"]

ip_features = ["ip_x_1", "ip_y_1", "ip_z_1", "ip_x_2", "ip_y_2", "ip_z_2"]

phi_cp_feature = ["gen_phitt"]

df = tree.pandas.df(momenta_features+other_features+target+selectors
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

#Neutrinos
nu_1_lab = Momentum4.e_m_eta_phi(df["gen_nu_p_1"], 0, df["gen_nu_eta_1"], df["gen_nu_phi_1"])
nu_2_lab = Momentum4.e_m_eta_phi(df["gen_nu_p_2"], 0, df["gen_nu_eta_2"], df["gen_nu_phi_2"])

# Find the boost to ZMF
zmf_momentum = pi_1_lab + pi_2_lab + pi0_1_lab + pi0_2_lab
boost = Momentum4(zmf_momentum[0], -zmf_momentum[1], -zmf_momentum[2], -zmf_momentum[3])

# Calculate 4-vectors in the ZMF
pi_1_ZMF = pi_1_lab.boost_particle(boost)
pi_2_ZMF = pi_2_lab.boost_particle(boost)

pi0_1_ZMF = pi0_1_lab.boost_particle(boost)
pi0_2_ZMF = pi0_2_lab.boost_particle(boost)

nu_1_ZMF = nu_1_lab.boost_particle(boost)
nu_2_ZMF = nu_2_lab.boost_particle(boost)

print("Boosted particles.")

#%% Features and targets

def normalise(x):
    return (x-tf.math.reduce_mean(x, axis=0))/tf.math.reduce_std(x, axis=0)

#add visible product features
x = tf.convert_to_tensor([pi0_1_lab, pi_1_lab, pi0_2_lab, pi_2_lab], dtype=tf.float32)
x = tf.transpose(x, [2, 0, 1])
x = tf.reshape(x, (x.shape[0], 16))

#add met features
x = tf.concat([x, tf.convert_to_tensor(df[met_features], dtype=tf.float32)], axis=1)

#add sv features
x = tf.concat([x, tf.convert_to_tensor(df[sv_features], dtype=tf.float32)], axis=1)

#add ip features
x = tf.concat([x, tf.convert_to_tensor(df[ip_features], dtype=tf.float32)], axis=1)

#y = tf.convert_to_tensor(df[neutrino_features], dtype=tf.float32)
y = tf.convert_to_tensor([nu_1_lab, nu_2_lab], dtype=tf.float32)
y = tf.transpose(y, [2, 0, 1])

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
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Reshape((2, 4))
])

opt = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['mae', 'mse'])

print("Model compiled.")


#%% Training model

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)

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

pred_minus_true = np.array(model(x_val)[:,0,0]) - np.array(y_val[:,0,0])

plt.figure()
plt.title("Neural Network Reconstruction of Leading Neutrino Energy")
plt.xlabel("GeV")
plt.ylabel("Frequency")
plt.xlim(0, 250)
#plt.xlim(-5, 5)
plt.hist(pred_minus_true, bins = 100, alpha = 0.5, label="Predicted Minus True")
plt.grid()
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.title("NN Predicted Minus True Leading Neutrino Momentum")
plt.xlabel("GeV")
plt.ylabel("Frequency")
plt.xlim(-100, 100)
#plt.xlim(-5, 5)
pred_minus_true = np.array(model(x_val)[:,0,1]) - np.array(y_val[:,0,1])
plt.hist(pred_minus_true, bins = 100, alpha = 0.5, label="px mean={:.2f}, std={:.2f}".format(np.mean(pred_minus_true), np.std(pred_minus_true)))
pred_minus_true = np.array(model(x_val)[:,0,2]) - np.array(y_val[:,0,2])
plt.hist(pred_minus_true, bins = 100, alpha = 0.5, label="py mean={:.2f}, std={:.2f}".format(np.mean(pred_minus_true), np.std(pred_minus_true)))
pred_minus_true = np.array(model(x_val)[:,0,3]) - np.array(y_val[:,0,3])
plt.hist(pred_minus_true, bins = 100, alpha = 0.5, label="pz mean={:.2f}, std={:.2f}".format(np.mean(pred_minus_true), np.std(pred_minus_true)))
plt.grid()
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.title("NN Predicted Minus True Subleading Neutrino Momentum")
plt.xlabel("GeV")
plt.ylabel("Frequency")
plt.xlim(-100, 100)
#plt.xlim(-5, 5)
pred_minus_true = np.array(model(x_val)[:,1,1]) - np.array(y_val[:,1,1])
plt.hist(pred_minus_true, bins = 100, alpha = 0.5, label="px mean={:.2f}, std={:.2f}".format(np.mean(pred_minus_true), np.std(pred_minus_true)))
pred_minus_true = np.array(model(x_val)[:,1,2]) - np.array(y_val[:,1,2])
plt.hist(pred_minus_true, bins = 100, alpha = 0.5, label="py mean={:.2f}, std={:.2f}".format(np.mean(pred_minus_true), np.std(pred_minus_true)))
pred_minus_true = np.array(model(x_val)[:,1,3]) - np.array(y_val[:,1,3])
plt.hist(pred_minus_true, bins = 100, alpha = 0.5, label="pz mean={:.2f}, std={:.2f}".format(np.mean(pred_minus_true), np.std(pred_minus_true)))
plt.grid()
plt.legend(loc="upper right")
plt.show()

#%% Higgs mass reconstruction

pred_nu_1_lab = Momentum4(model(x_val)[:,0,0],model(x_val)[:,0,1], model(x_val)[:,0,2], model(x_val)[:,0,3])
pred_nu_2_lab = Momentum4(model(x_val)[:,1,0],model(x_val)[:,1,1], model(x_val)[:,1,2], model(x_val)[:,1,3])

higgs_lab = pred_nu_1_lab+pred_nu_2_lab+pi_1_lab[:,numTrain:]+pi_2_lab[:,numTrain:]+pi0_1_lab[:,numTrain:]+pi0_2_lab[:,numTrain:]

higgs_mass = higgs_lab.m

plt.figure()
plt.title("NN Higgs Mass After Regressing Neutrino Momentum")
plt.xlabel("GeV")
plt.ylabel("Frequency")
plt.axvline(x=125.18, label="125.18GeV")
plt.xlim(0, 300)
plt.hist(higgs_mass, bins = 100, alpha = 0.5, label="mean={:.2f}, std={:.2f}".format(np.mean(higgs_mass), np.std(higgs_mass)))
plt.grid()
plt.legend(loc="upper right")
plt.show()

#%% Scatter plot

plt.figure()
plt.title("Leading Neutrino Pz")
plt.xlabel("Prediction / GeV")
plt.ylabel("True / GeV")
plt.hist2d(model(x_val)[:,0,3], y_val[:,0,3], 1000)
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.show()


#%% Evaluate

print(tf.math.reduce_mean(tf.math.abs(model(x)-y), axis=0)/tf.math.reduce_std(y, axis=0))
