# this is a piece of code to accurately cross check the alpha method for neutrino approximation.

import sys
#sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
import uproot 
import numpy as np

sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/Modules/")
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
from pylorentz import Momentum4
from pylorentz import Vector4
from pylorentz import Position4

# loading the tree
tree = uproot.open("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/MVAFILE_AllHiggs_tt_10_10_w_gen.root")["ntuple"]



print("\n Tree loaded\n")


# define what variables are to be read into the dataframe
momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2",
              "gen_vis_p_1", "gen_vis_p_2",
              "gen_vis_E_1", "gen_vis_E_2",
              "gen_vis_phi_1", "gen_vis_phi_2",
              "gen_vis_eta_1", "gen_vis_eta_2",
              #subleading neutral pi 4-momentum
              "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1", #leading neutrino, gen level
              "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2", #subleading neutrino, gen level  
              "pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1",
              "pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1",
              "pi2_E_2", "pi2_px_2", "pi2_py_2", "pi2_pz_2",
              "pi3_E_2", "pi3_px_2", "pi3_py_2", "pi3_pz_2"
                ] 

other_features = [ #"ip_x_1", "ip_y_1", "ip_z_1",        #leading impact parameter
                   #"ip_x_2", "ip_y_2", "ip_z_2",        #subleading impact parameter
                   #"y_1_1", "y_1_2",
                   #"ip_sig_1", "ip_sig_2",
                   "gen_phitt",
                   #"aco_angle_1", "pv_angle"
                 ]    # ratios of energies

target = [ "met", "metx", "mety", #"aco_angle_1", "aco_angle_6", "aco_angle_5", "aco_angle_7"
         ]  #acoplanarity angle
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2"
            ]

additional_info = [ "sv_x_1", "sv_y_1", "sv_z_1",
                    "sv_x_2", "sv_y_2", "sv_z_2",
                    "wt_cp_ps", "wt_cp_sm", "wt_cp_mm"
                    ]
met_covariance_matrices = ["metcov00", 
                           "metcov01", 
                           "metcov10", 
                           "metcov11" ]

#covs = sv_covariance_matrices + ip_covariance_matrices + met_covariance_matrices

variables4= momenta_features + other_features + target + selectors + additional_info + met_covariance_matrices 
print('Check 1')
df4 = tree.pandas.df(variables4)

df4 = df4[
      (df4["tau_decay_mode_1"] == tau_mode1) 
    & (df4["tau_decay_mode_2"] == tau_mode2) 
    & (df4["mva_dm_1"] == decay_mode1) 
    & (df4["mva_dm_2"] == decay_mode2)
    & (df4["gen_nu_p_1"] > -4000)
    & (df4["gen_nu_p_2"] > -4000)
    & (df4["pi_E_1"] != 0)
    & (df4["pi_E_2"] != 0)
    #& (df4["sv_x_1"] != 0)
    #& (df4["sv_x_2"] != 0)
    
]



print(len(df4),'This is the length') #up to here we are fine

lenght1 = len(df4)
trainFrac = 0.7
df4 = df4.dropna()



nu_1 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_1"])), df4["gen_nu_eta_1"], df4["gen_nu_phi_1"], df4["gen_nu_p_1"])
nu_2 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_2"])), df4["gen_nu_eta_2"], df4["gen_nu_phi_2"], df4["gen_nu_p_2"])

tau_1_vis, tau_2_vis = bf.get_vis(df4, decay_mode1, decay_mode2)

gen_tau_1_vis, gen_tau_2_vis = bf.get_gen_vis(df4, decay_mode1, decay_mode2)

tau_1 = tau_1_vis + nu_1
tau_2 = tau_2_vis + nu_2
gen_tau_1 = gen_tau_1_vis + nu_1
gen_tau_2 = gen_tau_2_vis + nu_2

reco_alpha_1, reco_alpha_2 = am.alphas(df4, decay_mode1, decay_mode2)
gen_alpha_1, gen_alpha_2 = am.gen_alphas(df4, decay_mode1, decay_mode2)


reco_beta_1 = nu_1.p/tau_1_vis.p 
gen_beta_1 = nu_1.p/gen_tau_1_vis.p
reco_beta_2 = nu_2.p/tau_2_vis.p 
gen_beta_2 = nu_2.p/gen_tau_2_vis.p

gen_dir1 = [gen_tau_1.p_x, gen_tau_1.p_y, gen_tau_1.p_z]
vis1_dir = [gen_tau_1_vis.p_x, gen_tau_1_vis.p_y, gen_tau_1_vis.p_z] 
vis1_dir = [tau_1_vis.p_x, tau_1_vis.p_y, tau_1_vis.p_z] 
gen_angle_1 = np.arccos(bf.dot_product(gen_dir1/bf.norm(gen_dir1), vis1_dir/bf.norm(vis1_dir)))*180/np.pi

gen_dir2 = [gen_tau_2.p_x, gen_tau_2.p_y, gen_tau_2.p_z]
vis2_dir = [gen_tau_2_vis.p_x, gen_tau_2_vis.p_y, gen_tau_2_vis.p_z] 
gen_angle_2 = np.arccos(bf.dot_product(gen_dir2/bf.norm(gen_dir2), vis2_dir/bf.norm(vis2_dir)))*180/np.pi


norm_sv_1 = bf.norm([df4['sv_x_1'], df4['sv_y_1'], df4['sv_z_1']])
norm_sv_1 = np.where(norm_sv_1 == 0, 9999, norm_sv_1)
dir_x_tau1 = df4['sv_x_1']/norm_sv_1
dir_y_tau1 = df4['sv_y_1']/norm_sv_1
dir_z_tau1 = df4['sv_z_1']/norm_sv_1

tau1_dir = [dir_x_tau1, dir_y_tau1, dir_z_tau1]

gen_sv_angle_1 = np.arccos(bf.dot_product(gen_dir1/bf.norm(gen_dir1), tau1_dir))*180/np.pi

bf.plot_2d(gen_angle_1, gen_sv_angle_1, 'angle_recoVis_genTau', 'angle_sv_genTau', (0,12.5))


plt.hist(gen_angle_1, alpha = 0.5, bins = 1000, label = 'Leading tau Mean: %.2f, std:%.2f'%( gen_angle_1.mean(), gen_angle_1.std()))
plt.hist(gen_angle_2, alpha = 0.5, bins = 1000, label = 'subleading tau Mean: %.2f, std:%.2f'%( gen_angle_2.mean(), gen_angle_2.std()))
plt.grid()
plt.xlim(0,1.25)
plt.xlabel('Angle between gen_tau and gen_visible (deg)')
plt.legend()
plt.show()
    
