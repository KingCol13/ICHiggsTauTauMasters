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
import matplotlib.pyplot as plt

from pylorentz import Momentum4
from pylorentz import Position4
from lbn import LBN, LBNLayer
    

#%% Data loading

# loading the tree
tree = uproot.open("ROOTfiles/MVAFILE_AllHiggs_tt_pseudo_phitt.root")["ntuple"]

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

cp_weights = ['wt_cp_sm', 'wt_cp_mm', 'wt_cp_ps']

df = tree.pandas.df(momenta_features+other_features+target+selectors+cp_weights)

num_data = len(df)

print("Loaded data.")

#%% Selection

#pi-pi
#df = df[(df['mva_dm_1'] == 0) & (df['mva_dm_2'] == 0)]

#rho-rho:
#df = df[(df['mva_dm_1'] == 1) & (df['mva_dm_2'] == 1) & (df['tau_decay_mode_1'] == 1) & (df['tau_decay_mode_2'] == 1)]

#a1-a1 (a1->pi+2pi0)
df = df[(df['mva_dm_1'] == 2) & (df['mva_dm_2'] == 2) & (df['tau_decay_mode_1'] == 1) & (df['tau_decay_mode_2'] == 1)]

#a1-a1 (a1->3pi))
#df = df[(df['mva_dm_1'] == 10) & (df['mva_dm_2'] == 10)]

#a1-a1 combined
#df = df[((df['mva_dm_1'] == 2) | (df['mva_dm_1'] == 10)) & ((df['mva_dm_2'] == 2) | (df['mva_dm_2'] == 10))]

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
phi_shift_1=np.where(big_O<0, phi_shift_0, 2*np.pi-phi_shift_0)

# Shift phi based on energy ratios
y_1 = np.array(df['y_1_1'])
y_2 = np.array(df['y_1_2'])
y_tau = np.array(df['y_1_1']*df['y_1_2'])
phi_shift_2=np.where(y_tau<0, phi_shift_1, np.where(phi_shift_1<np.pi, phi_shift_1+np.pi, phi_shift_1-np.pi))

#%% Histogram

#df["aco_angle_1_calc"] = 2*np.pi-phi_shift_2

plt.figure()
plt.title("Binary Discrimination")
plt.xlabel("aco_angle_1 / degrees")
plt.ylabel("Frequency")
plt.hist(df[df['wt_cp_sm']>df['wt_cp_ps']]['aco_angle_1'], 50, label="Even", alpha=0.5)
plt.hist(df[df['wt_cp_sm']<df['wt_cp_ps']]['aco_angle_1'], 50, label="Odd", alpha=0.5)
plt.legend()
plt.grid()
plt.show()