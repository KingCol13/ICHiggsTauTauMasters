#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 2021

@author: kingsley

Graphing discrimination between SM and PS based on aco angle
"""

#%% Initial imports and setup

import uproot 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pylorentz import Momentum4
from pylorentz import Position4
    

#%% Data loading

# loading the tree
tree = uproot.open("ROOTfiles/MVAFILE_AllHiggs_tt_pseudo_phitt.root")["ntuple"]

target = [    "aco_angle_1"]  #acoplanarity angle
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2"
             ]

momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2"] #subleading neutral pi 4-momentum

cp_weights = ['wt_cp_sm', 'wt_cp_mm', 'wt_cp_ps']

neutrinos = ["gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1",
             "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2"]

sv = ["sv_x_1", "sv_y_1", "sv_z_1"]

df = tree.pandas.df(target+selectors+cp_weights+neutrinos+momenta_features+sv)

num_data = len(df)

print("Loaded data.")

#%% Cleanup

df = df[(df['gen_nu_p_1']>-5000) & (df['gen_nu_p_2'] > -5000)]

df = df[(df['gen_nu_p_1'] > -150) & (df['gen_nu_p_1'] < 150)]

#%% Selection

#pi-pi
#df = df[(df['mva_dm_1'] == 0) & (df['mva_dm_2'] == 0)]

#rho-rho:
#df = df[(df['mva_dm_1'] == 1) & (df['mva_dm_2'] == 1) & (df['tau_decay_mode_1'] == 1) & (df['tau_decay_mode_2'] == 1)]

#a1-a1 (a1->pi+2pi0)
#df = df[(df['mva_dm_1'] == 2) & (df['mva_dm_2'] == 2) & (df['tau_decay_mode_1'] == 1) & (df['tau_decay_mode_2'] == 1)]

#a1-a1 (a1->3pi))
df = df[(df['mva_dm_1'] == 10) & (df['mva_dm_2'] == 10)]

#a1-a1 combined
#df = df[((df['mva_dm_1'] == 2) | (df['mva_dm_1'] == 10)) & ((df['mva_dm_2'] == 2) | (df['mva_dm_2'] == 10))]

#%% particles

nu_1 = Momentum4.e_m_eta_phi(df['gen_nu_p_1'], 0, df['gen_nu_eta_1'], df['gen_nu_phi_1'])
nu_2 = Momentum4.e_m_eta_phi(df['gen_nu_p_2'], 0, df['gen_nu_eta_2'], df['gen_nu_phi_2'])

# Create our 4-vectors in the lab frame
pi_1 = Momentum4(df["pi_E_1"], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
pi_2 = Momentum4(df["pi_E_2"], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])

pi0_1 = Momentum4(df["pi0_E_1"], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
pi0_2 = Momentum4(df["pi0_E_2"], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])

tau_1 = pi_1 + pi0_1 + nu_1
tau_2 = pi_2 + pi0_2 + nu_2

higgs = tau_1 + tau_2

#%% Histograms

res0 = np.array(nu_1[1,:])
res1 = np.array(nu_1[2,:])
res2 = np.array(nu_1[3,:])

plt.figure()
plt.title("Leading Neutrino")
plt.xlabel("Momentum / GeV")
plt.ylabel("Frequency")
#plt.xlim(-200, 200)
plt.hist(res0, 50, label="px\nMean: %.2f, std:%.2f"%(res0.mean(), res0.std()), alpha=0.5)
plt.hist(res1, 50, label="py\nMean: %.2f, std:%.2f"%(res1.mean(), res1.std()), alpha=0.5)
plt.hist(res2, 50, label="pz\nMean: %.2f, std:%.2f"%(res2.mean(), res2.std()), alpha=0.5)
plt.legend()
plt.grid()
plt.show()

res0 = tau_1.m
res1 = tau_2.m
res2 = higgs.m

plt.figure()
plt.title("Mass Distributions")
plt.xlabel("Mass/ GeV")
plt.ylabel("Frequency")
plt.hist(res0, 50, label="tau_1 mass\nMean: %.2f, std:%.2f"%(res0.mean(), res0.std()), alpha=0.5)
plt.hist(res1, 50, label="tau_2 mass\nMean: %.2f, std:%.2f"%(res1.mean(), res1.std()), alpha=0.5)
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.title("Higgs mass Distribution")
plt.xlabel("Mass/ GeV")
plt.ylabel("Frequency")
plt.hist(res2, 50, label="Higgs mass\nMean: %.2f, std:%.2f"%(res2.mean(), res2.std()), alpha=0.5)
plt.legend()
plt.grid()
plt.show()