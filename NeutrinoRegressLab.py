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

tree = uproot.open("ROOTfiles/MVAFILE_AllHiggs_tt_pseudo.root")["ntuple"]

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

#%% Clean up data

feature_quantities = momenta_features+met_features+sv_features+ip_features
target_quantities = neutrino_features

#Filter out -9999 values for neutrino momenta
df = df[df["gen_nu_p_1"] > -4000]
df = df[df["gen_nu_p_2"] > -4000]
df = df.dropna(subset=feature_quantities+target_quantities)

#%% train/eval split:

trainFrac = 0.5
df_train, df_eval = np.split(df, [int(trainFrac*len(df))], axis=0)

#%% Features and targets

def normalise(x):
    mu = tf.math.reduce_mean(x, axis=0)
    std = tf.math.reduce_std(x, axis=0)
    return (x-mu)/std, mu, std

#add visible product, met, sv, ip features
x_train = tf.convert_to_tensor(df_train[feature_quantities], dtype=tf.float32)

#y = tf.convert_to_tensor(df[neutrino_features], dtype=tf.float32)
y_train = tf.convert_to_tensor(df_train[neutrino_features], dtype=tf.float32)

#Eval data
x_eval = tf.convert_to_tensor(df_eval[feature_quantities], dtype=tf.float32)
y_eval = tf.convert_to_tensor(df_eval[neutrino_features], dtype=tf.float32)


#normalise
#x_train, x_mu, x_std = normalise(x_train)
y_train, y_mu, y_std = normalise(y_train)
#x_eval, _, _ = normalise(x_eval)
y_eval, _, _ = normalise(y_eval)


#%% Building network

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(64, activation="relu"),#, kernel_regularizer=tf.keras.regularizers.L1L2(0.01, 0.1)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),#, kernel_regularizer=tf.keras.regularizers.L1L2(0.01, 0.1)),
    #tf.keras.layers.Dense(200, activation="relu"),
    #tf.keras.layers.Dense(200, activation="relu"),
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6)
])

opt = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['mae', 'mse'])

print("Model compiled.")


#%% Training model

history = model.fit(x_train, y_train, validation_split=0.2, epochs=5)

#%% plot traning

plt.figure()
plt.plot(history.history['mse'], label="Training MSE")
plt.plot(history.history['val_mse'], label="Validation MSE")
plt.title("Loss on Iteration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%% Histograms

res = np.array((model(x_eval) - y_eval)*y_std + y_mu)

plt.figure()
plt.title("NN Predicted Minus True p")
plt.xlabel("Phi / Radians")
plt.ylabel("Frequency")
#plt.xlim(-100, 100)
plt.hist(res[:,0], bins = 100, alpha = 0.5, label="p_1 mean={:.2f}, std={:.2f}".format(np.mean(res[:,0]), np.std(res[:,0])))
plt.hist(res[:,1], bins = 100, alpha = 0.5, label="p_2 mean={:.2f}, std={:.2f}".format(np.mean(res[:,1]), np.std(res[:,1])))
plt.grid()
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.title("NN Predicted Minus True Phi")
plt.xlabel("Phi / Radians")
plt.ylabel("Frequency")
#plt.xlim(-100, 100)
plt.xlim(-5, 5)
plt.hist(res[:,2], bins = 100, alpha = 0.5, label="phi_1 mean={:.2f}, std={:.2f}".format(np.mean(res[:,2]), np.std(res[:,2])))
plt.hist(res[:,3], bins = 100, alpha = 0.5, label="phi_2 mean={:.2f}, std={:.2f}".format(np.mean(res[:,3]), np.std(res[:,3])))
plt.grid()
plt.legend(loc="upper right")
plt.show()

plt.figure()
plt.title("NN Predicted Minus True Eta")
plt.xlabel("Phi / Radians")
plt.ylabel("Frequency")
#plt.xlim(-100, 100)
plt.xlim(-5, 5)
plt.hist(res[:,4], bins = 100, alpha = 0.5, label="eta_1 mean={:.2f}, std={:.2f}".format(np.mean(res[:,4]), np.std(res[:,4])))
plt.hist(res[:,5], bins = 100, alpha = 0.5, label="eta_2 mean={:.2f}, std={:.2f}".format(np.mean(res[:,5]), np.std(res[:,5])))
plt.grid()
plt.legend(loc="upper right")
plt.show()

#%% Add/remove columns to dataframe

#TODO: fix sv_variables
del df_eval['sv_x_1'], df_eval['sv_y_1'], df_eval['sv_z_1']
del df_eval['sv_x_2'], df_eval['sv_y_2'], df_eval['sv_z_2']

res = np.array(model(x_eval)*y_std + y_mu)
df_eval['reco_nu_p_1'] = res[:,0]
df_eval['reco_nu_p_2'] = res[:,1]
df_eval['reco_nu_phi_1'] = res[:,2]
df_eval['reco_nu_phi_2'] = res[:,3]
df_eval['reco_nu_eta_1'] = res[:,4]
df_eval['reco_nu_eta_2'] = res[:,5]

#%%  Write root file

rootfile = uproot.recreate("example.root")

treeBranches = {column : str(df_eval[column].dtypes) for column in df_eval}
branchDict = {column : np.array(df_eval[column]) for column in df_eval}
tree = uproot.newtree(treeBranches, title="ntuple", compression=uproot.ZLIB(3))

with uproot.recreate("MVAFILE_AllHiggs_tt_reco.root") as f:
    f["ntuple"] = tree
    f["ntuple"].extend(branchDict)