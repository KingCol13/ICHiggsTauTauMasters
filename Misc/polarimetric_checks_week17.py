# this is a piece of code to accurately cross check the polarimetric vector method for neutrino approximation.

import sys
#sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
import uproot 
import numpy as np

sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/Modules")
import polarimetric_module_checks_week17 as polari
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
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
from pylorentz import Momentum4
from pylorentz import Vector4
from pylorentz import Position4

# loading the tree
tree = uproot.open("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/MVAFILE_AllHiggs_tt_10_10_w_gen.root")["ntuple"]


#tree = uproot.open("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/MVAFILE_AllHiggs_tt_reco_10_10_phitt.root")["ntuple"]
#tree = uproot.open("/eos/user/d/dwinterb/SWAN_projects/Masters_CP/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
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





nu_1_guess1, nu_1_guess2, nu_2_guess1, nu_2_guess2, best_guess_nu1, best_guess_nu2 = polari.polarimetric_full(df4, decay_mode1, decay_mode2, 'sv', 'tau_p')


print(nu_1_guess1)



raise END

print(len(df4),'This is the length') #up to here we are fine

lenght1 = len(df4)
trainFrac = 0.7
df4 = df4.dropna()

nu_1 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_1"])), df4["gen_nu_eta_1"], df4["gen_nu_phi_1"], df4["gen_nu_p_1"])
nu_2 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_2"])), df4["gen_nu_eta_2"], df4["gen_nu_phi_2"], df4["gen_nu_p_2"])

nu_1_guess1, nu_1_guess2, nu_2_guess1, nu_2_guess2, best_guess_nu1, best_guess_nu2 = polari.polarimetric(df4, decay_mode1, decay_mode2)

gen_nu_1_guess1, gen_nu_1_guess2, gen_nu_2_guess1, gen_nu_2_guess2, gen_best_guess_nu1, gen_best_guess_nu2 = polari.gen_polarimetric(df4, decay_mode1, decay_mode2)

nu_1_guess1_theta, nu_1_guess2_theta, nu_2_guess1_theta, nu_2_guess2_theta, best_guess_nu1_theta, best_guess_nu2_theta = polari.polarimetric_theta_max(df4, decay_mode1, decay_mode2)


nu_1_guess1_no, nu_1_guess2_no, nu_2_guess1_no, nu_2_guess2_no, best_guess_nu1_no, best_guess_nu2_no = polari.polarimetric_no_clamping(df4, decay_mode1, decay_mode2)

nu_1_guess1_f, nu_1_guess2_f, nu_2_guess1_f, nu_2_guess2_f, best_guess_nu1_f, best_guess_nu2_f = polari.polarimetric_change_dir(df4, decay_mode1, decay_mode2)

tau_1_vis, tau_2_vis = bf.get_vis(df4, decay_mode1, decay_mode2)
#reco_tau    
tau_1 = nu_1 + tau_1_vis
tau_2 = nu_2 + tau_2_vis



#bf.plot_2d(nu_1.phi, best_guess_nu1_theta.phi, 'gen nus.phi', 'naive nus.phi', (-3, 3), (-3,3))
#bf.plot_2d(nu_1.phi, best_guess_nu1_f.phi, 'gen nus.phi', 'geometric nus.phi', (-3, 3), (-3,3) )

#bf.plot_2d(nu_1.eta, best_guess_nu1_theta.eta, 'gen nus.eta', 'naive nus.eta', (-3, 3), (-3,3))
#bf.plot_2d(nu_1.eta, best_guess_nu1_f.eta, 'gen nus.eta', 'geometric nus.eta', (-3, 3), (-3,3) )

#bf.plot_2d(nu_1.p, best_guess_nu1_theta.p, 'gen nus.p', 'naive nus.p', (0, 100), (0,100))
#bf.plot_2d(nu_1.p, best_guess_nu1_f.p, 'gen nus.p', 'geometric nus.p', (0, 100), (0,100))

#bf.plot_2d((best_guess_nu1_theta+tau_1_vis).phi, tau_1_vis.phi, 'naive nus', 'geometric nus', (-3, 3), (-3,3), (1000, 1000) )

#best_guess_nu1_f+

tau_1_theta = tau_1_vis + best_guess_nu1_theta

tau_1_theta_dir = [tau_1_theta[1], tau_1_theta[2], tau_1_theta[3]]/(bf.norm([tau_1_theta[1], tau_1_theta[2], tau_1_theta[3]]))
tau_1_dir = [tau_1[1], tau_1[2], tau_1[3]]/(bf.norm([tau_1[1], tau_1[2], tau_1[3]]))

plt.title('Geometric Method - Tau direction\nC = 10000 points')
hist = bf.dot_product(tau_1_theta_dir, tau_1_dir )
plt.hist(hist-1, alpha = 0.5, bins = 1000, label = 'Naive')

tau_1_f = tau_1_vis + best_guess_nu1_f

tau_1_f_dir = [tau_1_f[1], tau_1_f[2], tau_1_f[3]]/(bf.norm([tau_1_f[1], tau_1_f[2], tau_1_f[3]]))
#tau_1_dir = [tau_1[1], tau_1[2], tau_1[3]]/(bf.norm([tau_1[1], tau_1[2], tau_1[3]]))

hist = bf.dot_product(tau_1_f_dir, tau_1_dir)
plt.xlabel('Dot product between gen_tau and pola_tau - 1')
plt.hist(hist-1, alpha = 0.5, bins = 1000, label = 'Geometric')
plt.grid()
plt.legend()
plt.show()

plt.title('Neutrino - momenta estimates\nClosest Higgs mass - updated polarimetric method')
plt.subplot(2,2,1)
bf.plot_1d(nu_1.p_z, best_guess_nu1.p_z, 'Gen_nu_1_p', 'first_polarimetric_nu_1.p')
bf.plot_1d(nu_1.p_x, best_guess_nu1.p_x, 'Gen_nu_1', 'first_polarimetric_nu_1.px')
bf.plot_1d(nu_1.p_y, best_guess_nu1.p_y, 'Gen_nu_1.p', 'first_polarimetric_nu1.p')
plt.grid()

plt.subplot(2,2,2)
bf.plot_1d(nu_1.p_z, best_guess_nu1_f.p_z, 'Gen_nu_1_p', 'new_polarimetric_nu_1.p')
bf.plot_1d(nu_1.p_x, best_guess_nu1_f.p_x, 'Gen_nu_1', 'new_polarimetric_nu_1.px')
bf.plot_1d(nu_1.p_y, best_guess_nu1_f.p_y, 'Gen_nu_1.p', 'new_polarimetric_nu1.p')
plt.grid()

plt.subplot(2,2,3)
bf.plot_1d(nu_2.p_z, best_guess_nu2.p_z, 'Gen_nu_2_p', 'first_polarimetric_nu_2.p')
bf.plot_1d(nu_2.p_x, best_guess_nu2.p_x, 'Gen_nu_2', 'first_polarimetric_nu_2.px')
bf.plot_1d(nu_2.p_y, best_guess_nu2.p_y, 'Gen_nu_2.p', 'first_polarimetric_nu2.p')
plt.grid()

plt.subplot(2,2,4)
bf.plot_1d(nu_2.p_z, best_guess_nu2_f.p_z, 'Gen_nu_2_p', 'new_polarimetric_nu_2.p')
bf.plot_1d(nu_2.p_x, best_guess_nu2_f.p_x, 'Gen_nu_2', 'new_polarimetric_nu_2.px')
bf.plot_1d(nu_2.p_y, best_guess_nu2_f.p_y, 'Gen_nu_2.p', 'new_polarimetric_nu2.p')
plt.grid()


plt.show()

raise End



gen_vis_1 = Momentum4.e_eta_phi_p(df4["gen_vis_E_1"],df4["gen_vis_eta_1"],df4["gen_vis_phi_1"],df4["gen_vis_p_1"])
gen_vis_2 = Momentum4.e_eta_phi_p(df4["gen_vis_E_2"],df4["gen_vis_eta_2"],df4["gen_vis_phi_2"],df4["gen_vis_p_2"])

#reco_tau    
tau_1 = nu_1 + tau_1_vis
tau_2 = nu_2 + tau_2_vis

#gen_tau
gen_tau_1 = nu_1 + gen_vis_1
gen_tau_2 = nu_2 + gen_vis_2





# polarimetric reco taus
tau1_guess1 = nu_1_guess1 + tau_1_vis
tau1_guess2 = nu_1_guess2 + tau_1_vis

tau2_guess1 = nu_2_guess1 + tau_2_vis
tau2_guess2 = nu_2_guess2 + tau_2_vis

tau1_guess3 = best_guess_nu1 + tau_1_vis
tau2_guess3 = best_guess_nu2 + tau_2_vis

tau1_guess4 = tf.where(abs(tau1_guess1.p-tau_1.p)<=abs(tau1_guess2.p-tau_1.p), tau1_guess1, tau1_guess2)
tau1_guess4 = Momentum4(tau1_guess4[0], tau1_guess4[1], tau1_guess4[2], tau1_guess4[3])

tau2_guess4 = tf.where(abs(tau2_guess1.p-tau_2.p)<=abs(tau2_guess2.p-tau_2.p), tau2_guess1, tau2_guess2)
tau2_guess4 = Momentum4(tau2_guess4[0], tau2_guess4[1], tau2_guess4[2], tau2_guess4[3])



# theta polarimetric reco taus
tau1_guess1_theta = nu_1_guess1_theta + tau_1_vis
tau1_guess2_theta = nu_1_guess2_theta + tau_1_vis

tau2_guess1_theta  = nu_2_guess1_theta  + tau_2_vis
tau2_guess2_theta  = nu_2_guess2_theta  + tau_2_vis

tau1_guess3_theta  = best_guess_nu1_theta  + tau_1_vis
tau2_guess3_theta  = best_guess_nu2_theta  + tau_2_vis

tau1_guess4_theta  = tf.where(abs(tau1_guess1_theta .p-tau_1.p)<=abs(tau1_guess2_theta .p-tau_1.p), tau1_guess1_theta , tau1_guess2_theta )
tau1_guess4_theta = Momentum4(tau1_guess4_theta[0], tau1_guess4_theta[1], tau1_guess4_theta[2], tau1_guess4_theta[3])

tau2_guess4_theta  = tf.where(abs(tau2_guess1_theta.p-tau_2.p)<=abs(tau2_guess2_theta.p-tau_2.p), tau2_guess1_theta, tau2_guess2_theta)
tau2_guess4_theta = Momentum4(tau2_guess4_theta[0], tau2_guess4_theta[1], tau2_guess4_theta[2], tau2_guess4_theta[3])


nu_1_guess3 = tau1_guess3_theta-tau_1_vis
nu_2_guess3 = tau2_guess3_theta-tau_2_vis

nu_1_guess4 = tau1_guess4_theta-tau_1_vis
nu_2_guess4 = tau2_guess4_theta-tau_2_vis




######### another way to check best solution - met ####
reco4_met_x = nu_1_guess4.p_x + nu_2_guess4.p_x
reco3_met_x = nu_1_guess3.p_x + nu_2_guess3.p_x
reco11_met_x = nu_1_guess1_theta.p_x + nu_2_guess1_theta.p_x
reco22_met_x = nu_1_guess2_theta.p_x + nu_2_guess2_theta.p_x
reco12_met_x = nu_1_guess1_theta.p_x + nu_2_guess2_theta.p_x
reco21_met_x = nu_1_guess2_theta.p_x + nu_2_guess1_theta.p_x

plt.title('Prelimiary checks - can met be used for solution picking\nupdated polarimetric method')
bf.plot_2d(df4['metx']-reco4_met_x, df4['metx']-reco3_met_x, 'metx - reco4_met_x (best tau p)', 'metx - reco4_met_x (best Higgs mass)', (-500, 500), (-500, 500))


reco4_met_y = nu_1_guess4.p_y + nu_2_guess4.p_y
reco3_met_y = nu_1_guess3.p_y + nu_2_guess3.p_y

reco11_met_y = nu_1_guess1_theta.p_y + nu_2_guess1_theta.p_y
reco22_met_y = nu_1_guess2_theta.p_y + nu_2_guess2_theta.p_y
reco12_met_y = nu_1_guess1_theta.p_y + nu_2_guess2_theta.p_y
reco21_met_y = nu_1_guess2_theta.p_y + nu_2_guess1_theta.p_y


##### Now we need to make the selection
m_Higgs = 125.10 #GeV
Higgs11 = tau1_guess1_theta + tau2_guess1_theta
Higgs22 = tau1_guess2_theta + tau2_guess2_theta
Higgs12 = tau1_guess1_theta + tau2_guess2_theta
Higgs21 = tau1_guess2_theta + tau2_guess1_theta
Higgs3 = tau1_guess3_theta + tau2_guess3_theta


def met_choice(Higgs_option1, Higgs_option2, nu_1_option1, nu_2_option1,nu_1_option2, nu_2_option2):
    global reco11_met_y, reco22_met_y,reco12_met_y,reco21_met_y
    global reco11_met_x, reco22_met_x,reco12_met_x,reco21_met_x
    global df4
    global Higgs11, Higgs22, Higgs21, Higgs12 
    
    reco_met_x_option1 = nu_1_option1.p_x + nu_2_option1.p_x
    reco_met_x_option2 = nu_1_option2.p_x + nu_2_option2.p_x
    
    reco_met_y_option1 = nu_1_option1.p_y + nu_2_option1.p_y
    reco_met_y_option2 = nu_1_option2.p_y + nu_2_option2.p_y
    
    diff_Higgs1 = (Higgs_option1.m-m_Higgs)**2
    diff_Higgs2 = (Higgs_option2.m-m_Higgs)**2
    diff_metx1 = (reco_met_x_option1 - df4['metx'])**2
    diff_mety1 = (reco_met_y_option1 - df4['mety'])**2
    diff_metx2 = (reco_met_x_option2 - df4['metx'])**2
    diff_mety2 = (reco_met_y_option2 - df4['mety'])**2
    
    total_diff1 = diff_Higgs1 + 0 * diff_metx1 + 0 * diff_mety1
    total_diff2 = diff_Higgs2 + 0 * diff_metx2 + 0 * diff_mety2
    
    
    print(total_diff1 - total_diff2)
    better_higgs = tf.where(total_diff1 <= total_diff2, Higgs_option1, Higgs_option2)
    
    better_nu1 = tf.where(total_diff1 <= total_diff2, nu_1_option1, nu_1_option2)
    
    better_nu2 = tf.where(total_diff1 <= total_diff2, nu_2_option1, nu_2_option2)
    
    better_higgs = Momentum4(better_higgs[0], better_higgs[1], better_higgs[2], better_higgs[3])
    
    better_nu1 = Momentum4(better_nu1[0], better_nu1[1], better_nu1[2], better_nu1[3])
    
    better_nu2 = Momentum4(better_nu2[0], better_nu2[1], better_nu2[2], better_nu2[3])
    
    return better_higgs, better_nu1, better_nu2
    
    
higgs_best1, nu_1_best1, nu_2_best1 = met_choice(Higgs11, Higgs22, nu_1_guess1_theta, nu_2_guess1_theta, nu_1_guess2_theta, nu_2_guess2_theta)

higgs_best2, nu_1_best2, nu_2_best2 = met_choice(Higgs12, Higgs21, nu_1_guess1_theta, nu_2_guess2_theta, nu_1_guess2_theta, nu_2_guess1_theta)

higgs_best, nu_1_best, nu_2_best = met_choice(higgs_best1, higgs_best2, nu_1_best1, nu_2_best1, nu_1_best2, nu_2_best2)



print(higgs_best.m - Higgs3.m)

plt.plot(higgs_best.m - Higgs3.m, 'x')

#############################################

plt.title('Prelimiary checks - can met be used for solution picking\nupdated polarimetric method')
bf.plot_2d(abs(df4['mety']-reco4_met_y)/df4['metcov11'], abs(df4['mety']-reco3_met_y)/df4['metcov11'], 'abs(mety - reco4_met_y (best tau p))/metcov11', 'abs(mety - reco4_met_y(best Higgs mass))/metcov11', (-2, 2), (-2, 2), (500, 500))

plt.title('Prelimiary checks - can met be used for solution picking\nupdated polarimetric method')
bf.plot_2d(abs(df4['mety']-reco4_met_y), abs(df4['mety']-reco3_met_y), 'abs(mety - reco4_met_y (best tau p))', 'abs(mety - reco4_met_y(best Higgs mass))', (-500, 500), (-500, 500))


'''
For the pi with the PV method you basically just need to know the vector describing the plane spanned by the tau and pi 3-momenta's which is ~ equivalent to the plane spanned by the IP and the pi so i would imagine trying to use the additional constraints from the MET / mass etc to try and improve the estimate of this plan you get just from using the IP and pi vectors - does it make sense?

i mean maybe the point is try to regress directly the PV of the pi rather than trying to get the neutrino because maybe some important angular corrections are not picked up when you regress the neutrino direction

you have the gen information of the tau and tau neutrino right? so you could be able to compute the gen-level PV from this and then try and regress this directly

and P.S in another skype chat Mario showed a plot that suggest that the rest frame is not very important at all, whether you use the higgs rest frame or just the visible rest frame that is so you can just work in the visible rest frame

'''


















raise END


