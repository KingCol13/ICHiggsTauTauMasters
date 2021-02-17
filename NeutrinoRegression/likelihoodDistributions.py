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

pions = ["pi_px_1", "pi_py_1", "pi_pz_1", "pi_E_1"]

cp_weights = ['wt_cp_sm', 'wt_cp_mm', 'wt_cp_ps']

neutrinos = ["gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1",
             "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2"]

sv = ["sv_x_1", "sv_y_1", "sv_z_1"]

df = tree.pandas.df(target+selectors+cp_weights+neutrinos+pions+sv)

num_data = len(df)

print("Loaded data.")

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


#%%

nu_1 = Momentum4.e_m_eta_phi(df['gen_nu_p_1'], 0, df['gen_nu_eta_1'], df['gen_nu_phi_1'])
nu_2 = Momentum4.e_m_eta_phi(df['gen_nu_p_2'], 0, df['gen_nu_eta_2'], df['gen_nu_phi_2'])

#%% Histogram

df = df[df['aco_angle_1']>-5000]

plt.figure()
plt.title("Binary Discrimination")
plt.xlabel("aco_angle_1 / degrees")
plt.ylabel("Frequency")
plt.hist(df[df['wt_cp_sm']>df['wt_cp_ps']]['aco_angle_1'], 50, label="Even", alpha=0.5)
plt.hist(df[df['wt_cp_sm']<df['wt_cp_ps']]['aco_angle_1'], 50, label="Odd", alpha=0.5)
plt.legend()
plt.grid()
plt.show()