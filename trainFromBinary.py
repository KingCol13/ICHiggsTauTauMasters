#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:51:03 2020

@author: kingsley
"""

#%% imports

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import subprocess

#%%

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
#%% Data selecting

momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2"] #subleading neutral pi 4-momentum

#target = [    "aco_angle_1"]  #acoplanarity angle

"""
target = [  "gen_nu_p_1", "gen_nu_p_2",
                       "gen_nu_phi_1", "gen_nu_phi_2",
                       "gen_nu_eta_1", "gen_nu_eta_2"]
"""
target = ["gen_nu_phi_1", "gen_nu_phi_2"]

met_features = ["met", "metx", "mety"]#, "metcov00", "metcov01", "metcov10", "metcov11"]

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

tau_decay_mode_1 = 1
tau_decay_mode_2 = 1

#%% Build binary dataset

all_variables = momenta_features+met_features+ip_features+target
my_command = ["./makeBinaryDataset"]+[str(tau_decay_mode_1), str(tau_decay_mode_2)]+all_variables

process = subprocess.Popen(my_command,
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()

if(process.returncode!=0):
    raise Exception("makeBinaryDataset failed with stderr: "+stderr.decode("utf-8"))

print("makeBinaryDataset stdout:")
print(stdout.decode("utf-8"))

#%% Set up dataset

batch_size = 32

num_targets = len(target)
num_features = len(all_variables) - num_targets

def decode_example(example):
    return tf.convert_to_tensor(tf.io.decode_raw(example, tf.dtypes.float32))

def tuples_for_keras(batch):
    return (batch[:,:num_features], batch[:,num_features:])

dataset = tf.data.FixedLengthRecordDataset("recordData.dat", len(all_variables)*4).map(decode_example)

#dataset = dataset.map(lambda x: tf.io.decode_raw(x,tf.dtypes.float32))
dataset = dataset.batch(batch_size)

dataset = dataset.map(tuples_for_keras)

#%% Build model

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(num_targets)
])

opt = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['mae', 'mse'])

#%% Training model

history = model.fit(x=dataset, epochs=5)

#plot traning
plt.figure()
plt.plot(history.history['mse'], label="Training MSE")
plt.title("Loss on Iteration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%% Histograms

#Append each batch true values to list then vstack them
it = dataset.as_numpy_iterator()
true_vals = []
for element in it:
    true_vals.append(element[1])

true_vals = np.vstack(true_vals_arr)

res = model.predict(dataset) - true_vals

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
