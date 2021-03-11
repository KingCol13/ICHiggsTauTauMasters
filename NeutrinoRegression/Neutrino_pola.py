# This is the python code that will regress and then save the neutrinos best regressed within the root file. 


# For now the best case scenario is 1,0,0,0,0 z 8


print('Hello World')
import sys
#sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
import uproot 
import numpy as np

sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/Modules")
import basic_functions as bf
import configuration_module as conf
import polarimetric_module_checks_week17 as polari

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf



#don't  forget to change name of the file at the end
tau_mode1 = 10
tau_mode2 = 10
decay_mode1 = 10
decay_mode2 = 10


# stop tensorflow trying to overfill GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


#for some reason pylorentz is installed somewhere differently ?
#sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
from pylorentz import Momentum4
from pylorentz import Vector4
from pylorentz import Position4

# loading the tree
tree = uproot.open("/home/acraplet/Alie/Masters/MVAFILE_full_10_10.root")["ntuple"]
#tree = uproot.open("/eos/user/d/dwinterb/SWAN_projects/Masters_CP/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
print("\n Tree loaded\n")


# define what variables are to be read into the dataframe
momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2", #subleading neutral pi 4-momentum
              "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1", #leading neutrino, gen level
              #"gen_vis_p_1", "gen_vis_p_2",
              #"gen_vis_E_1", "gen_vis_E_2",
              #"gen_vis_phi_1", "gen_vis_phi_2",
              #"gen_vis_eta_1", "gen_vis_eta_2",
              
              
              "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2", #subleading neutrino, gen level  
              "pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1",
              "pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1",
              "pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2",
              "pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2"
                ] 

other_features = [ "ip_x_1", "ip_y_1", "ip_z_1",        #leading impact parameter
                   "ip_x_2", "ip_y_2", "ip_z_2",        #subleading impact parameter
                   #"y_1_1", "y_1_2",
                   "ip_sig_1", "ip_sig_2",
                   "gen_phitt",
                   "aco_angle_1", "aco_angle_6", "aco_angle_5", "aco_angle_7", "pv_angle"
                 ]    # ratios of energies

target = [ "met", "metx", "mety", #"aco_angle_1", "aco_angle_6", "aco_angle_5", "aco_angle_7"
         ]  #acoplanarity angle
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2", "rand"
            ]

additional_info = [ "sv_x_1", "sv_y_1", "sv_z_1",
                    "sv_x_2", "sv_y_2", "sv_z_2",
                    "wt_cp_ps", "wt_cp_sm", "wt_cp_mm"
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


variables4= momenta_features + other_features + target + selectors + additional_info + met_covariance_matrices   #+ covs #copying Kinglsey's way cause it is very clean
print('Check 1')
df4 = tree.pandas.df(variables4)

df4 = df4[
      (df4["tau_decay_mode_1"] == tau_mode1) 
    & (df4["tau_decay_mode_2"] == tau_mode2) 
    & (df4["mva_dm_1"] == decay_mode1) 
    & (df4["mva_dm_2"] == decay_mode2)
    #& (df4["gen_nu_p_1"] > -4000)
    #& (df4["gen_nu_p_2"] > -4000)
    #& (df4["sv_x_1"] != 0)
    #& (df4["sv_x_2"] != 0)
    
]

print(len(df4),'This is the length') #up to here we are fine

lenght1 = len(df4)
trainFrac = 0.7
df4 = df4.dropna()

length2 = len(df4)

print((lenght1-length2)/lenght1)

df_train, df_eval = np.split(df4, [int(trainFrac*len(df4))], axis=0)

df_eval = df4

print("panda Data frame created \n")

df4.head()


_, _, _, _, nu1_guessOld, nu2_guessOld, checks = polari.polarimetric_full(df_eval, decay_mode1, decay_mode2, 'sv', 'HiggsM', True)
df_eval['pola0_nu_p_1'] = nu1_guessOld.p
df_eval['pola0_nu_p_2'] = nu2_guessOld.p
df_eval['pola0_nu_phi_1'] = nu1_guessOld.phi
df_eval['pola0_nu_phi_2'] = nu2_guessOld.phi
df_eval['pola0_nu_eta_1'] = nu1_guessOld.eta
df_eval['pola0_nu_eta_2'] = nu2_guessOld.eta
df_eval['pola0_nu_eta_2'] = nu2_guessOld.eta
df_eval['theta_max_level'] = checks


_, _, _, _, nu1_guessOld, nu2_guessOld = polari.polarimetric_full(df_eval, decay_mode1, decay_mode2, 'naive', 'HiggsM')
df_eval['pola1_nu_p_1'] = nu1_guessOld.p
df_eval['pola1_nu_p_2'] = nu2_guessOld.p
df_eval['pola1_nu_phi_1'] = nu1_guessOld.phi
df_eval['pola1_nu_phi_2'] = nu2_guessOld.phi
df_eval['pola1_nu_eta_1'] = nu1_guessOld.eta
df_eval['pola1_nu_eta_2'] = nu2_guessOld.eta


_, _, _, _, nu1_guessOld, nu2_guessOld = polari.polarimetric_full(df_eval, decay_mode1, decay_mode2, 'geo', 'HiggsM')
df_eval['pola2_nu_p_1'] = nu1_guessOld.p
df_eval['pola2_nu_p_2'] = nu2_guessOld.p
df_eval['pola2_nu_phi_1'] = nu1_guessOld.phi
df_eval['pola2_nu_phi_2'] = nu2_guessOld.phi
df_eval['pola2_nu_eta_1'] = nu1_guessOld.eta
df_eval['pola2_nu_eta_2'] = nu2_guessOld.eta


_, _, _, _, nu1_guessOld, nu2_guessOld = polari.polarimetric_full(df_eval, decay_mode1, decay_mode2, 'sv', 'HiggsM_met')
df_eval['pola3_nu_p_1'] = nu1_guessOld.p
df_eval['pola3_nu_p_2'] = nu2_guessOld.p
df_eval['pola3_nu_phi_1'] = nu1_guessOld.phi
df_eval['pola3_nu_phi_2'] = nu2_guessOld.phi
df_eval['pola3_nu_eta_1'] = nu1_guessOld.eta
df_eval['pola3_nu_eta_2'] = nu2_guessOld.eta


_, _, _, _, nu1_guessOld, nu2_guessOld = polari.polarimetric_full(df_eval, decay_mode1, decay_mode2, 'naive', 'HiggsM_met')
df_eval['pola4_nu_p_1'] = nu1_guessOld.p
df_eval['pola4_nu_p_2'] = nu2_guessOld.p
df_eval['pola4_nu_phi_1'] = nu1_guessOld.phi
df_eval['pola4_nu_phi_2'] = nu2_guessOld.phi
df_eval['pola4_nu_eta_1'] = nu1_guessOld.eta
df_eval['pola4_nu_eta_2'] = nu2_guessOld.eta


_, _, _, _, nu1_guessOld, nu2_guessOld = polari.polarimetric_full(df_eval, decay_mode1, decay_mode2, 'geo', 'HiggsM_met')
df_eval['pola5_nu_p_1'] = nu1_guessOld.p
df_eval['pola5_nu_p_2'] = nu2_guessOld.p
df_eval['pola5_nu_phi_1'] = nu1_guessOld.phi
df_eval['pola5_nu_phi_2'] = nu2_guessOld.phi
df_eval['pola5_nu_eta_1'] = nu1_guessOld.eta
df_eval['pola5_nu_eta_2'] = nu2_guessOld.eta



_, _, _, _, nu1_guessOld, nu2_guessOld = polari.polarimetric_full(df_eval, decay_mode1, decay_mode2, 'sv', 'tau_p')
df_eval['pola6_nu_p_1'] = nu1_guessOld.p
df_eval['pola6_nu_p_2'] = nu2_guessOld.p
df_eval['pola6_nu_phi_1'] = nu1_guessOld.phi
df_eval['pola6_nu_phi_2'] = nu2_guessOld.phi
df_eval['pola6_nu_eta_1'] = nu1_guessOld.eta
df_eval['pola6_nu_eta_2'] = nu2_guessOld.eta


_, _, _, _, nu1_guessOld, nu2_guessOld = polari.polarimetric_full(df_eval, decay_mode1, decay_mode2, 'naive', 'tau_p')
df_eval['pola7_nu_p_1'] = nu1_guessOld.p
df_eval['pola7_nu_p_2'] = nu2_guessOld.p
df_eval['pola7_nu_phi_1'] = nu1_guessOld.phi
df_eval['pola7_nu_phi_2'] = nu2_guessOld.phi
df_eval['pola7_nu_eta_1'] = nu1_guessOld.eta
df_eval['pola7_nu_eta_2'] = nu2_guessOld.eta


_, _, _, _, nu1_guessOld, nu2_guessOld = polari.polarimetric_full(df_eval, decay_mode1, decay_mode2, 'geo', 'tau_p')
df_eval['pola8_nu_p_1'] = nu1_guessOld.p
df_eval['pola8_nu_p_2'] = nu2_guessOld.p
df_eval['pola8_nu_phi_1'] = nu1_guessOld.phi
df_eval['pola8_nu_phi_2'] = nu2_guessOld.phi
df_eval['pola8_nu_eta_1'] = nu1_guessOld.eta
df_eval['pola8_nu_eta_2'] = nu2_guessOld.eta

treeBranches = {column : str(df_eval[column].dtypes) for column in df_eval}
branchDict = {column : np.array(df_eval[column]) for column in df_eval}
tree = uproot.newtree(treeBranches, title="ntuple", compression=uproot.ZLIB(3))

with uproot.recreate("MVAFILE_full_10_10_pola12.root") as f:
    f["ntuple"] = tree
    f["ntuple"].extend(branchDict)
