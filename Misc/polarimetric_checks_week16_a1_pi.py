# this is a piece of code to accurately cross check the polarimetric vector method for neutrino approximation.

import sys
#sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
import uproot 
import numpy as np

sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/Modules")
import polarimetric_module_checks_week15 as polari
import alpha_module as am
import basic_functions as bf
import configuration_module as conf


import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from lbn_modified3 import LBN, LBNLayer
import tensorflow as tf
from matplotlib import colors

#working in the a1(3pi)-a1(3pi) channel
tau_mode1 = 10
tau_mode2 = 1
decay_mode1 = 10
decay_mode2 = 1


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
#tree = uproot.open("/home/acraplet/Alie/Masters/MVAFILE_full_10_0_pola2_ippv.root")["ntuple"]
#tree2 = uproot.open("/home/acraplet/Alie/Masters/MVAFILE_full_0_10_ippv.root")["ntuple"]


#tree3 = uproot.open("/home/acraplet/Alie/Masters/MVAFILE_full_10_10_pola12_pv.root")["ntuple"]
tree3 = uproot.open("/home/acraplet/Alie/Masters/MVAFILE_full_10_X_ippv4.root")["ntuple"]

tree4 = uproot.open("/home/acraplet/Alie/Masters/MVAFILE_full_10_X_ippv5.root")["ntuple"]
#tree = uproot.open("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/MVAFILE_AllHiggs_tt_reco_10_10_phitt.root")["ntuple"]
#tree = uproot.open("/eos/user/d/dwinterb/SWAN_projects/Masters_CP/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
print("\n Tree loaded\n")



# define what variables are to be read into the dataframe
momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2",
              #"gen_vis_p_1", "gen_vis_p_2",
              #"gen_vis_E_1", "gen_vis_E_2",
              #"gen_vis_phi_1", "gen_vis_phi_2",
              #"gen_vis_eta_1", "gen_vis_eta_2",
              #subleading neutral pi 4-momentum
              "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1", #leading neutrino, gen level
              "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2", #subleading neutrino, gen level  
              #"reco_nu_p_1", "reco_nu_phi_1", "reco_nu_eta_1",
              #"reco_nu_p_2", "reco_nu_phi_2", "reco_nu_eta_2",
              
              
              #"reco2_nu_p_1", "reco2_nu_phi_1", "reco2_nu_eta_1",
              #"reco2_nu_p_2", "reco2_nu_phi_2", "reco2_nu_eta_2",
              
              "pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1",
              "pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1",
              "pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2",
              "pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2"
                ] 


other_features = [ "ip_x_1", "ip_y_1", "ip_z_1",        #leading impact parameter
                   "ip_x_2", "ip_y_2", "ip_z_2",        #subleading impact parameter
                   #"y_1_1", "y_1_2",
                   "ip_sig_1", "ip_sig_2",
                   "gen_phitt", #"pseudo_phitt",
                   #'pola_nb_shit',
                   #'ippv8_angle',
                   "pv_angle", "ippv_angle",
                   #"ippv2_angle", "ippv8_angle",
                   #"pseudo_ippv_angle", "pseudo2_ippv_angle",
                   "aco_angle_1", "aco_angle_5",
                   #"reco_pv_angle", "reco2_pv_angle", 
                   
                   
                   #"reco_phitt", "reco2_phitt", "pola_phitt", "pola2_phitt"
                   
                 ]    # ratios of energies

target = [ "met", "metx", "mety", #"aco_angle_1", "aco_angle_6", "aco_angle_5", "aco_angle_7"
         ]  #acoplanarity angle
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2", "rand"
            ]

additional_info = [ "sv_x_1", "sv_y_1", "sv_z_1",
                    "sv_x_2", "sv_y_2", "sv_z_2",
                    "wt_cp_ps", "wt_cp_sm", #"wt_cp_mm"
                    ]
met_covariance_matrices = ["metcov00", 
                           "metcov01", 
                           "metcov10", 
                           "metcov11" ]

variables4= momenta_features + other_features + target + selectors + additional_info #+ met_covariance_matrices 
print('Check 1')
#df5 = tree2.pandas.df(variables4)


#df4 = tree.pandas.df(variables4)
#df40 = tree2.pandas.df(variables4)
df3 = tree3.pandas.df(variables4)
df4 = tree4.pandas.df(variables4)

df3 = df3[
     (df3["ippv_angle"] > -4000)
     #& (df3["aco_angle_5"] > -4000)
     &(df3["gen_nu_p_1"] > -4000)
     &(df3["gen_nu_p_2"] > -4000)
     &(df3["ip_sig_2"] > 1.5)
     & (df4["mva_dm_2"] == 0)
    #& (df4["mva_dm_2"] == decay_mode2)
    ]

trainFrac = 0.7
#df_train, df3 = np.split(df3, [int(trainFrac*len(df3))], axis=0)

df4 = df4[
     (df4["ippv_angle"] > -4000)
     #& (df4["aco_angle_5"] > -4000)
     &(df4["gen_nu_p_1"] > -4000)
     &(df4["gen_nu_p_2"] > -4000)
     &(df4["ip_sig_2"] > 1.5)
     &(df4["mva_dm_2"] == 1)
     &(df4["tau_decay_mode_2"] == 1)
    ]

print(df4['ip_x_1'])

def make_y (df, level, ZMF):
    pi = Momentum4(df['pi_E_'+str(level)], df['pi_px_'+str(level)], df['pi_py_'+str(level)], df['pi_pz_'+str(level)])
    
    pi0 = Momentum4(df['pi0_E_'+str(level)], df['pi0_px_'+str(level)], df['pi0_py_'+str(level)], df['pi0_pz_'+str(level)])
    
    pi.boost_particle(ZMF)
    pi0.boost_particle(ZMF)
    
    y = (pi.e - pi0.e)/(pi.e+pi0.e)
    return y


a1, rho = bf.get_vis(df4, decay_mode1, decay_mode2)

df4['y'] = make_y(df4, 2, a1+rho) #* make_y(df4, 1, a1+rho)

df4['ippv2_angle'] = tf.where(df4['y']<=0, tf.where(df4['ippv_angle']<np.pi, df4['ippv_angle']+np.pi, df4['ippv_angle']-np.pi), df4['ippv_angle'])

#
nbins = 15

plt.title('a1-pi/rho channel CP-sensitivity improvements with ip_sig > 1.5')
bf.plot_sm_ps('aco_angle_5', 'r-', df3, nbins, 'a1-pi aco_angle_5 with '+str(nbins)+' bins\na1-pi ZMF')
bf.plot_sm_ps('ippv_angle', 'b-', df3, nbins, 'a1-pi ippv_angle with '+str(nbins)+' bins\n improvement from aco_angle5: '+str(bf.improvement(df3, 'ippv_angle', 'aco_angle_5', nbins))[:5])



bf.plot_sm_ps('aco_angle_1', 'r--', df4, nbins, 'a1-rho ippv aco_angle_1 with '+str(nbins)+' bins\na1-pi ZMF')
bf.plot_sm_ps('ippv_angle', 'b--', df4, nbins, 'a1-rho ippv_angle with '+str(nbins)+' bins\n improvement from aco_angle1: '+str(bf.improvement(df4, 'ippv_angle', 'aco_angle_1', nbins))[:5])
bf.plot_sm_ps('ippv2_angle', 'g--', df4, nbins, 'a1-rho ippv_angle shifted with '+str(nbins)+' bins\n improvement from aco_angle1: '+str(bf.improvement(df4, 'ippv2_angle', 'aco_angle_1', nbins))[:5])

plt.ylabel('ps/sm distributions')
plt.xlabel ('angle (rad)')
plt.grid()
plt.legend()
plt.show()

df0_ps, df0_sm = bf.make_ps_sm(df4)
plt.title('sm-ps separation of ippv_angle in a1-rho channel\nafter y shift')
plt.hist(df0_ps['ippv2_angle'], alpha = 0.7, bins = 100, label ='PS ippv_angle')
plt.hist(df0_sm['ippv2_angle'], alpha = 0.7,  bins = 100, label ='SM ippv_angle')
plt.legend()
plt.xlabel('ippv2_angle')
plt.ylabel('distribution')
plt.grid()
plt.show()
