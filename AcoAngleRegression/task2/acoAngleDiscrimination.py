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
from lbn import LBN, LBNLayer
    

#%% Data loading

# loading the tree
tree = uproot.open("ROOTfiles/MVAFILE_AllHiggs_tt_pseudo_phitt.root")["ntuple"]

target = ["aco_angle_1", "pv_angle"]  #acoplanarity angle
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2"
             ]

cp_weights = ['wt_cp_sm', 'wt_cp_mm', 'wt_cp_ps']

df = tree.pandas.df(target+selectors+cp_weights)

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

df = df[df['pv_angle']>-5000]

plt.figure()
plt.title("Binary Discrimination")
plt.xlabel("pv_angle / degrees")
plt.ylabel("Frequency")
plt.hist(df[df['wt_cp_sm']>df['wt_cp_ps']]['pv_angle'], 50, label="Even", alpha=0.5)
plt.hist(df[df['wt_cp_sm']<df['wt_cp_ps']]['pv_angle'], 50, label="Odd", alpha=0.5)
plt.legend()
plt.grid()
plt.show()