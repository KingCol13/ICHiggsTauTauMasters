#!/usr/bin/python

#This script will be used to investigate which outputs and which NN structures give the best neutrino regression. 


import sys
#sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
import uproot 
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

import matplotlib as mpl

mpl.use('tkagg')
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/Modules/")
import basic_functions as bf
import configuration_module as conf
import polarimetric_module as polari
import alpha_module as am


#tau_mode1 = 0
#tau_mode2 = 10
#decay_mode1 = 0
#decay_mode2 = 10

if len(sys.argv) == 5:
    tau_mode1 = int(sys.argv[1])
    tau_mode2 = int(sys.argv[2])
    decay_mode1 = int(sys.argv[3])
    decay_mode2 = int(sys.argv[4])


#print('\nWe are regressing the %i, %i channel' %(decay_mode1, decay_mode2))


#for some reason pylorentz is installed somewhere differently ?
#sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
from pylorentz import Momentum4
from pylorentz import Vector4
from pylorentz import Position4


# loading the tree
tree = uproot.open("/home/acraplet/Alie/Masters/MVAFILE_AllHiggs_tt.root")["ntuple"]
#tree = uproot.open("/eos/user/d/dwinterb/SWAN_projects/Masters_CP/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
print("\n Tree loaded\n")


# define what variables are to be read into the dataframe
momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2", #subleading neutral pi 4-momentum
              "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1", #leading neutrino, gen level
              #"nu_px_1", "nu_py_1", "nu_pz_1", "nu_E_1",
              #"nu_px_2", "nu_py_2", "nu_pz_2", "nu_E_2",
              "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2", #subleading neutrino, gen level  
              "pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1",
              "pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1",
              "pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2",
              "pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2"
                ] 

other_features = [ "ip_x_1", "ip_y_1", "ip_z_1",        #leading impact parameter
                   "ip_x_2", "ip_y_2", "ip_z_2",        #subleading impact parameter
                   #"y_1_1", "y_1_2",
                   "gen_phitt", "ip_sig_2", "ip_sig_1"
                 ]    # ratios of energies

target = [ "metx", "mety", "aco_angle_1", "aco_angle_6", "aco_angle_5", "aco_angle_7",  "met", "pv_angle"]
          #acoplanarity angle
    
selectors = [ #"dm_1", "dm_2",
             "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2",
             "rand","wt_cp_ps","wt_cp_sm", "wt_cp_mm"
            ]

additional_info = [ "sv_x_1", "sv_y_1", "sv_z_1",
                    "sv_x_2", "sv_y_2", "sv_z_2",
                    ]

sv_covariance_matrices = ["svcov00_1", "svcov01_1", "svcov02_1",
                       "svcov10_1", "svcov11_1", "svcov12_1", 
                       "svcov20_1", "svcov21_1", "svcov22_1", 
                       "svcov00_2", "svcov01_2", "svcov02_2",
                       "svcov10_2", "svcov11_2", "svcov12_2", 
                       "svcov20_2", "svcov21_2", "svcov22_2", 
    
]

ip_covariance_matrices = ["ipcov00_1", "ipcov01_1", "ipcov02_1",
                       "ipcov10_1", "ipcov11_1", "ipcov12_1", 
                       "ipcov20_1", "ipcov21_1", "ipcov22_1", 
                       "ipcov00_2", "ipcov01_2", "ipcov02_2",
                       "ipcov10_2", "ipcov11_2", "ipcov12_2", 
                       "ipcov20_2", "ipcov21_2", "ipcov22_2", 
    
]

met_covariance_matrices = ["metcov00", 
                           "metcov01", 
                           "metcov10", 
                           "metcov11" ]

variables4=momenta_features+other_features+target+selectors + additional_info + met_covariance_matrices #+ #sv_covariance_matrices #+ covs #copying Kinglsey's way cause it is very clean
print('Check 1')
df4 = tree.pandas.df(variables4)
print('Tree made')

df4 = df4[
      (df4["tau_decay_mode_2"] == 10) 
    #(df4["tau_decay_mode_2"] == 10) 
    & (df4["mva_dm_2"] == 10) 
    #& (df4["mva_dm_2"] == decay_mode2)
    & (df4["gen_nu_p_1"] > -4000)
    & (df4["gen_nu_p_2"] > -4000)
    #& (df4["sv_x_1"] != 0)
    #& (df4["sv_x_2"] != 0)
    
]

#df4 = df4[
        #(df4["dm_1"] == decay_mode1) 
      #& (df4["dm_2"] == decay_mode2) 
    #]
    
df_eval = df4 

treeBranches = {column : str(df_eval[column].dtypes) for column in df_eval}
branchDict = {column : np.array(df_eval[column]) for column in df_eval}
tree = uproot.newtree(treeBranches, title="ntuple", compression=uproot.ZLIB(3))


sys.path.append("/home/acraplet/Alie/Masters/")
with uproot.recreate("MVAFILE_full_X_10.root") as f:
    f["ntuple"] = tree
    f["ntuple"].extend(branchDict)



































