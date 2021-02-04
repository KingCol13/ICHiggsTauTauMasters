#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:46:19 2021

@author: kingsley
"""


#%% Imports

import uproot 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylorentz import Momentum4

#%% Data loading

tree = uproot.open("/mnt/hdd/ROOTFiles/MVAFILE_AllHiggs_tt_pseudo.root")["ntuple"]

momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1",
              "pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1",
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",        #subleading charged pi 4-momentum
              "pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2",
              "pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2",
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2", #subleading neutral pi 4-momentum
              "gam1_E_1", "gam1_px_1", "gam1_py_1", "gam1_pz_1",
              "gam2_E_1", "gam2_px_1", "gam2_py_1", "gam2_pz_1",
              "gam3_E_1", "gam3_px_1", "gam3_py_1", "gam3_pz_1",
              "gam4_E_1", "gam4_px_1", "gam4_py_1", "gam4_pz_1",
              "gam1_E_2", "gam1_px_2", "gam1_py_2", "gam1_pz_2",
              "gam2_E_2", "gam2_px_2", "gam2_py_2", "gam2_pz_2",
              "gam3_E_2", "gam3_px_2", "gam3_py_2", "gam3_pz_2",
              "gam4_E_2", "gam4_px_2", "gam4_py_2", "gam4_pz_2"]

target = [    "aco_angle_1"]  #acoplanarity angle
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2",
             "tauFlag_1", "tauFlag_2"
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

wt_cp_features = ["wt_cp_sm", "wt_cp_mm", "wt_cp_ps",
                  "pseudo_wt_cp_sm", "pseudo_wt_cp_mm", "pseudo_wt_cp_ps"]

df = tree.pandas.df(momenta_features+target+selectors+wt_cp_features+neutrino_features)
#                    +met_features+sv_features+ip_features
#                    +phi_cp_feature)

#select decay mode
#0=pi, 1=rho, 2=a1(->pi+2pi0), 10=a1(->3pi), 11=3pi+pi0, -1=others
dm=1

# df = df[ #tau_decay_mode==1 sets hadronic decays
#       (df["tau_decay_mode_1"] == 1) 
#     & (df["tau_decay_mode_2"] == 1)
#     & (df["mva_dm_1"] == dm) 
#     & (df["mva_dm_2"] == dm)
# ]

#Filter out -9999 values for neutrino momenta
#df = df[df["gen_nu_p_1"] > -4000]
#df = df[df["gen_nu_p_2"] > -4000]

#%% wt_cp_sm histogram comparison
df2 = df[['wt_cp_sm','pseudo_wt_cp_sm']]
df2 = df2.dropna()

plt.figure()
plt.title("wt_cp_sm gen vs pseudo")
plt.xlabel("wt_cp_sm")
plt.ylabel("pseudo_wt_cp_sm")
plt.hist2d(df2['wt_cp_sm'], df2['pseudo_wt_cp_sm'])
plt.show()

#%% Histograms
df_plot = df[df['tauFlag_1'] == 2]
plt.figure()
plt.hist2d(df_plot['tauFlag_1'], df_plot['tau_decay_mode_1'], bins=40)
plt.title("tauFlag_1=2 vs taU_decay_mode_1 histogram")
plt.xlabel("tauFlag_1")
plt.ylabel('tau_decay_mode_1')
plt.grid()
plt.show()