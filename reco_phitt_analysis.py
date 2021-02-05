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

tree = uproot.open("ROOTfiles/MVAFILE_AllHiggs_tt_reco_phitt.root")["ntuple"]

selectors = ['mva_dm_1', 'mva_dm_2']

variables = ["aco_angle_1", "gen_phitt", "reco_phitt" ] #"pseudo_phitt"]

df = tree.pandas.df(variables+selectors)

#%% Selection

df = df[(df['mva_dm_1'] == 1) & (df['mva_dm_2'] == 1)]

#%% Cleanup

df = df.dropna()
df = df[df['aco_angle_1'] > -400]

#%% Fix shift in data
gen_phitt = np.array(df['gen_phitt'])
aco_angle_1 = np.array(df['aco_angle_1'])
new_phitt = np.array(df['reco_phitt'])*180/np.pi
new_phitt = np.where(new_phitt>90, new_phitt-180, new_phitt)

wt_cp_sm = np.array(df['wt_cp_sm'])
wt_cp_mm = np.array(df['wt_cp_mm'])
wt_cp_ps = np.array(df['wt_cp_ps'])

#%% Histograms

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