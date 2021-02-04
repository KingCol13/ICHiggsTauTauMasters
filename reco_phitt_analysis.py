#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:25:01 2021

@author: kingsley
"""

#%% Imports

import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% Read data

tree = uproot.open("/mnt/hdd/ROOTFiles/MVAFILE_AllHiggs_tt_pseudo.root")["ntuple"]

selectors = ['mva_dm_1', 'mva_dm_2']

variables = ["aco_angle_1", "gen_phitt", "pseudo_phitt"]

df = tree.pandas.df(variables+selectors)

#%% Selection

df = df[(df['mva_dm_1'] == 1) & (df['mva_dm_2'] == 1)]

#%% Cleanup

df = df.dropna()
df = df[df['aco_angle_1'] > -400]

#%% Fix shift in data
#TODO: remove this

gen_phitt = np.array(df['gen_phitt'])[0:-2]
pseudo_phitt = np.array(df['pseudo_phitt'])[1:-1]*180/np.pi
aco_angle_1 = np.array(df['aco_angle_1'])[0:-2]

pseudo_phitt = np.where(pseudo_phitt>90, pseudo_phitt-180, pseudo_phitt)

#%% Plot histograms

plt.figure()
plt.xlabel("gen_phitt")
plt.ylabel("pseudo_phitt")
plt.hist2d(gen_phitt, pseudo_phitt, 50)
plt.grid()
plt.show()

plt.figure()
plt.xlabel("gen_phitt")
plt.ylabel("aco_angle_1")
plt.hist2d(gen_phitt, aco_angle_1, 50)
plt.grid()
plt.show()

plt.figure()

plt.xlabel("pseudo_phitt")
plt.ylabel("aco_angle_1")
plt.hist2d(pseudo_phitt, aco_angle_1, 50)
plt.grid()
plt.show()