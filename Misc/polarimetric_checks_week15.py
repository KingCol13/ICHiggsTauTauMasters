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



print(len(df4),'This is the length') #up to here we are fine

lenght1 = len(df4)
trainFrac = 0.7
df4 = df4.dropna()

nu_1 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_1"])), df4["gen_nu_eta_1"], df4["gen_nu_phi_1"], df4["gen_nu_p_1"])
nu_2 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_2"])), df4["gen_nu_eta_2"], df4["gen_nu_phi_2"], df4["gen_nu_p_2"])

nu_1_guess1, nu_1_guess2, nu_2_guess1, nu_2_guess2, best_guess_nu1, best_guess_nu2 = polari.polarimetric(df4, decay_mode1, decay_mode2)

gen_nu_1_guess1, gen_nu_1_guess2, gen_nu_2_guess1, gen_nu_2_guess2, gen_best_guess_nu1, gen_best_guess_nu2 = polari.gen_polarimetric(df4, decay_mode1, decay_mode2)

nu_1_guess1_theta, nu_1_guess2_theta, nu_2_guess1_theta, nu_2_guess2_theta, best_guess_nu1_theta, best_guess_nu2_theta = polari.polarimetric_theta_max(df4, decay_mode1, decay_mode2)


tau_1_vis, tau_2_vis = bf.get_vis(df4, decay_mode1, decay_mode2)

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








plt.title('Quality of neutrino1 reconstruction - BEST CASE SCENARIO\nTau dir = sv')
best_nu1_dir = [best_guess_nu1[1], best_guess_nu1[2], best_guess_nu1[3]]
nu1_dir = [nu_1[1], nu_1[2], nu_1[3]]
dot_product = bf.dot_product(best_nu1_dir/bf.norm(best_nu1_dir), nu1_dir/bf.norm(nu1_dir))
p_diff = (nu_1.p-(tau1_guess4-tau_1_vis).p)
bf.plot_2d(dot_product, p_diff, 'dot_product(guess_nu1_hat, gen_nu1_hat)', '(gen_nu1-guess_nu1.p)', (.7, 1), (0, 500), (400, 800))


plt.title('Quality of tau1 reconstruction\nTau dir = theta*a1')
best_nu1_dir_theta = [best_guess_nu1_theta[1], best_guess_nu1_theta[2], best_guess_nu1_theta[3]]
best_tau1_dir_theta = [tau1_guess4_theta[1], tau1_guess4_theta[2], tau1_guess4_theta[3]]
nu1_dir = [nu_1[1], nu_1[2], nu_1[3]]
tau1_dir = [tau_1[1], tau_1[2], tau_1[3]]
dot_product_theta = np.clip(bf.dot_product(best_tau1_dir_theta /bf.norm(best_tau1_dir_theta), tau1_dir/bf.norm(tau1_dir)), -2, 2)
nu_1_guess4 = tau1_guess4_theta-tau_1_vis
p_diff_theta = (tau_1.p-(tau1_guess4).p)
bf.plot_2d(dot_product_theta, p_diff_theta, 'dot_product(guess_tau1_modif_hat, gen_tau1_hat)', '(gen_tau1.p-guess_tau1_modif.p)', (0.9999, 1), (-200, 200), (1000, 600))




plt.title('Quality of neutrino2 reconstruction - BEST CASE SCENARIO\nTau dir = sv')
best_nu2_dir = [best_guess_nu2[1], best_guess_nu2[2], best_guess_nu2[3]]
nu2_dir = [nu_2[1], nu_2[2], nu_2[3]]
dot_product = bf.dot_product(best_nu2_dir/bf.norm(best_nu2_dir), nu2_dir/bf.norm(nu2_dir))
p_diff = (nu_2.p-(tau2_guess4-tau_2_vis).p)
bf.plot_2d(dot_product, p_diff, 'dot_product(guess_nu2_hat, gen_nu2_hat)', '(gen_nu2-guess_nu2.p)', (.7, 1), (0, 500), (400, 800))


plt.title('Quality of neutrino2 reconstruction - BEST CASE SCENARIO\nTau dir = theta*a1')
best_nu2_dir_theta = [best_guess_nu2_theta[1], best_guess_nu2_theta[2], best_guess_nu2_theta[3]]

nu2_dir = [nu_2[1], nu_2[2], nu_2[3]]

dot_product_theta = np.clip(bf.dot_product(best_nu2_dir_theta /bf.norm(best_nu2_dir_theta), nu2_dir/bf.norm(nu2_dir)), 0.9, 2)
nu_2_guess4_theta = tau2_guess4_theta-tau_2_vis
p_diff = (nu_2.p-(nu_2_guess4_theta).p)
bf.plot_2d(dot_product_theta, p_diff_theta, 'dot_product(guess_nu2_modif_hat, gen_nu2_hat)', '(gen_nu2.p-guess_nu2_modif.p)', (0.998, 1), (-200, 200), (1000, 600))




plt.title('Quality of tau2 reconstruction\nTau dir = theta*a1')
best_tau2_dir_theta = [tau2_guess4_theta[1], tau2_guess4_theta[2], tau2_guess4_theta[3]]
tau2_dir = [tau_2[1], tau_2[2], tau_2[3]]

dot_product_theta = np.clip(bf.dot_product(best_tau2_dir_theta /np.clip(bf.norm(best_tau2_dir_theta), 0.0000000001, 10000000), tau2_dir/bf.norm(tau2_dir)), -2, 2)

p_diff = (tau_2.p-(tau2_guess4).p)
bf.plot_2d(dot_product_theta, p_diff_theta, 'dot_product(guess_tau2_modif_hat, gen_tau2_hat)', '(gen_tau2.p-guess_tau2_modif.p)', (0.9999, 1), (-200, 200), (1000, 600))



plt.title('Fractionnal distance to gen neutrino in momentum')
p_diff = (nu_1.p-nu_1_guess4.p)
plt.hist(np.clip(p_diff, -30, 30), bins = 500)
p_diff_no_modif = (nu_1.p-nu_guess4.p)
plt.hist(np.clip(p_diff_no_modif, -30, 30), bins = 500)
plt.xlabel('(nu_1.p-best_guess_nu1.p)')
plt.grid()
plt.show()
plt.title('Dot product between gen nu direction and polarimetric nu direction')
plt.hist(dot_product, bins = 200)
plt.grid()
plt.show()


plt.title('Quality of neutrino reconstruction\nTau dir = sv')
best_nu1_dir = [best_guess_nu1[1], best_guess_nu1[2], best_guess_nu1[3]]
nu1_dir = [nu_1[1], nu_1[2], nu_1[3]]
dot_product = bf.dot_product(best_nu1_dir/bf.norm(best_nu1_dir), nu1_dir/bf.norm(nu1_dir))
p_diff = (nu_1.p-best_guess_nu1.p)/nu_1.p
bf.plot_2d(dot_product, p_diff, 'dot_product(guess_nu1_hat, gen_nu1_hat)', '(gen_nu1-guess_nu1.p)/(nu_1.p)', (.7, 1), (0, 500), (400, 10000))

plt.title('Quality of neutrino2 reconstruction\nTau dir = sv')
best_nu2_dir = [best_guess_nu2[1], best_guess_nu2[2], best_guess_nu2[3]]
nu2_dir = [nu_2[1], nu_2[2], nu_2[3]]
dot_product2 = bf.dot_product(best_nu2_dir/bf.norm(best_nu2_dir), nu2_dir/bf.norm(nu2_dir))
p_diff2 = (nu_2.p-best_guess_nu2.p)/nu_2.p
bf.plot_2d(dot_product2, p_diff2, 'dot_product(guess_nu2_hat, gen_nu2_hat)', '(gen_nu2-guess_nu2.p)/(nu_2.p)', (.7, 1), (0, 500), (400, 10000))


plt.title('Fractionnal distance to gen neutrino in momentum\nTau dir = sv')
plt.hist(np.clip(p_diff, -30, 30), bins = 500, alpha = 0.5, label = 'Leading Tau neutrino')
plt.hist(np.clip(p_diff2, -30, 30), bins = 500, alpha = 0.5, label = 'Subleading Tau neutrino')
plt.xlabel('(nu.p-best_guess_nu.p)/nu.p')
plt.legend()
plt.grid()
plt.show()

plt.title('Dot product between gen nu direction and polarimetric nu direction\nTau dir = sv')
plt.hist(dot_product, bins = 200, alpha = 0.5, label = 'Leading Tau neutrino')
plt.hist(dot_product2, bins = 200, alpha = 0.5, label = 'Subleading Tau neutrino')
plt.legend()
plt.grid()
plt.show()


#Very nice - to present the repartition of data accross the range
p_diff = bf.dot_product(best_nu1_dir/bf.norm(best_nu1_dir), nu1_dir/bf.norm(nu1_dir))
p_diff2 = bf.dot_product(best_nu2_dir/bf.norm(best_nu2_dir), nu2_dir/bf.norm(nu2_dir))
p_diff3 = bf.dot_product(best_nu1_dir_theta/bf.norm(best_nu1_dir_theta), nu1_dir/bf.norm(nu1_dir))
p_diff4 = bf.dot_product(best_nu2_dir_theta/bf.norm(best_nu2_dir_theta), nu2_dir/bf.norm(nu2_dir))

plt.title('Repartition of dot_product(gen_nu, best_guess_nu)')
x = np.linspace(0, 1, 100)
y = np.quantile(p_diff, x)
y2 = np.quantile(p_diff2, x)

y3 = np.quantile(p_diff3, x)
y4 = np.quantile(p_diff4, x)
plt.plot([-1000, 1000], [0.975, 0.975], 'b--', alpha = 0.5, label = '95percent enclosed')
plt.plot([-1000, 1000], [0.025, 0.025], 'b--', alpha = 0.5)
plt.plot([-1000, 1000], [0.835, 0.835], 'r--', alpha = 0.5, label = '67percent enclosed')
plt.plot([-1000, 1000], [0.165, 0.165], 'r--', alpha = 0.5)
plt.grid()
plt.ylabel('Fraction of data below that value')
plt.xlabel('dot_product(gen_nu, best_guess_nu)')


plt.plot(y, x, 'k', label = 'Leading nu 95 percent between %.3f and %.3f\n67 percent between %.3f and %.3f'%(np.quantile(p_diff, 0.025), np.quantile(p_diff, 0.975), np.quantile(p_diff, 0.165), np.quantile(p_diff, 0.835)))
plt.plot(y2, x, 'g', label = 'Subleading nu 95 percent between %.3f and %.3f\n67 percent between %.3f and %.3f'%(np.quantile(p_diff2, 0.025), np.quantile(p_diff2, 0.975), np.quantile(p_diff2, 0.165), np.quantile(p_diff2, 0.835)))

plt.plot(y3, x, 'k--', label = 'Leading nu MODIFIED 95 percent between %.3f and %.3f\n67 percent between %.3f and %.3f'%(np.quantile(p_diff3, 0.025), np.quantile(p_diff3, 0.975), np.quantile(p_diff3, 0.165), np.quantile(p_diff3, 0.835)))
plt.plot(y4, x, 'g--', label = 'Subleading nu MODIFIED 95 percent between %.3f and %.3f\n67 percent between %.3f and %.3f'%(np.quantile(p_diff4, 0.025), np.quantile(p_diff4, 0.975), np.quantile(p_diff4, 0.165), np.quantile(p_diff4, 0.835)))
plt.legend()
plt.show()

#Very nice - to present the repartition of data accross the range
p_diff = (nu_1.p-best_guess_nu1.p)/nu_1.p
p_diff2 = (nu_2.p-best_guess_nu2.p)/nu_2.p


p_diff3 = (nu_1.p-best_guess_nu1_theta.p)/nu_1.p
p_diff4 = (nu_2.p-best_guess_nu2_theta.p)/nu_2.p

plt.title('Repartition of (gen_nu.p  - best_guess_nu.p)/gen_nu.p data')
x = np.linspace(0, 1, 100)
y = np.quantile(p_diff, x)
y2 = np.quantile(p_diff2, x)
y3 = np.quantile(p_diff3, x)
y4 = np.quantile(p_diff4, x)
plt.plot([-1000, 1000], [0.975, 0.975], 'b--', alpha = 0.5, label = '95percent enclosed')
plt.plot([-1000, 1000], [0.025, 0.025], 'b--', alpha = 0.5)
plt.plot([-1000, 1000], [0.835, 0.835], 'r--', alpha = 0.5, label = '67percent enclosed')
plt.plot([-1000, 1000], [0.165, 0.165], 'r--', alpha = 0.5)
plt.grid()
plt.ylabel('Fraction of data below that value')
plt.xlabel('dot_product(gen_nu, best_guess_nu)')

plt.plot(y, x, 'k', label = 'Leading nu 95 percent between %.3f and %.3f\n67 percent between %.3f and %.3f'%(np.quantile(p_diff, 0.025), np.quantile(p_diff, 0.975), np.quantile(p_diff, 0.165), np.quantile(p_diff, 0.835)))
plt.plot(y2, x, 'g', label = 'Subleading nu 95 percent between %.3f and %.3f\n67 percent between %.3f and %.3f'%(np.quantile(p_diff2, 0.025), np.quantile(p_diff2, 0.975), np.quantile(p_diff2, 0.165), np.quantile(p_diff2, 0.835)))

plt.plot(y3, x, 'k--', label = 'Leading nu MODIFIED 95 percent between %.3f and %.3f\n67 percent between %.3f and %.3f'%(np.quantile(p_diff3, 0.025), np.quantile(p_diff3, 0.975), np.quantile(p_diff3, 0.165), np.quantile(p_diff3, 0.835)))
plt.plot(y4, x, 'g--', label = 'Subleading nu MODIFIED 95 percent between %.3f and %.3f\n67 percent between %.3f and %.3f'%(np.quantile(p_diff4, 0.025), np.quantile(p_diff4, 0.975), np.quantile(p_diff4, 0.165), np.quantile(p_diff4, 0.835)))
plt.legend()
plt.show()


raise END








































# polarimetric gen taus
gen_tau1_guess1 = gen_nu_1_guess1 + gen_vis_1
gen_tau1_guess2 = gen_nu_1_guess2 + gen_vis_1

gen_tau2_guess1 = gen_nu_2_guess1 + gen_vis_2
gen_tau2_guess2 = gen_nu_2_guess2 + gen_vis_2

gen_tau1_guess3 = gen_best_guess_nu1 + gen_vis_1
gen_tau2_guess3 = gen_best_guess_nu2 + gen_vis_2

gen_tau1_guess4 = tf.where(abs(gen_tau1_guess1.p-gen_tau_1.p)<=abs(gen_tau1_guess2.p-gen_tau_1.p), gen_tau1_guess1, gen_tau1_guess2)
gen_tau1_guess4 = Momentum4(gen_tau1_guess4[0], gen_tau1_guess4[1], gen_tau1_guess4[2], gen_tau1_guess4[3])

gen_tau2_guess4 = tf.where(abs(gen_tau2_guess1.p-gen_tau_2.p)<=abs(gen_tau2_guess2.p-gen_tau_2.p), gen_tau2_guess1, gen_tau2_guess2)
gen_tau2_guess4 = Momentum4(gen_tau2_guess4[0], gen_tau2_guess4[1], gen_tau2_guess4[2], gen_tau2_guess4[3])


fraction_same = np.where(tau1_guess4.p - tau1_guess3.p==0, 1, 0)
print(sum(fraction_same)/len(fraction_same))

bf.plot_2d(gen_tau1_guess1.m, gen_tau1_guess2.m, 'gen_tau1_guess1.m', 'gen_tau1_guess2.m', (0, 1000), (0,1000))







### below: the graphs
plt.title('Check of the tau p reconstruction', fontsize = 'xx-large')
plt.subplot(2,2,1)
hist = np.array(gen_tau_1.p - tau1_guess4.p)
plt.hist(hist, bins = 1000, label = 'Mean: %.2f std: %.2f'%(hist.mean(), hist.std()))
plt.grid(alpha = 0.5)
plt.xlim(-250, 250)
plt.xlabel('gen_tau1.p - reco_polarim_tau1_closest.p')
plt.legend()
#
plt.subplot(2,2,2)
hist = np.array(gen_tau_2.p - tau2_guess4.p)
plt.hist(hist, bins = 1000, label = 'Mean: %.2f std: %.2f'%(hist.mean(), hist.std()))
plt.grid(alpha = 0.5)
plt.xlim(-250, 250)
plt.xlabel('gen_tau2.p - reco_polarim_tau2_closest.p')
plt.legend()
#
plt.subplot(2,2,3)
hist = np.array(gen_tau_1.p - gen_tau1_guess4.p)
plt.hist(hist, bins = 1000, label = 'Mean: %.2f std: %.2f'%(hist.mean(), hist.std()))
plt.grid(alpha = 0.5)
plt.xlim(-250, 250)
plt.xlabel('gen_tau1.p - gen_polarim_tau1_closest.p')
plt.legend()
#
plt.subplot(2,2,4)
hist = np.array(gen_tau_2.p - gen_tau2_guess4.p)
plt.hist(hist, bins = 1000, label = 'Mean: %.2f std: %.2f'%(hist.mean(), hist.std()))
plt.grid(alpha = 0.5)
plt.xlim(-250, 250)
plt.xlabel('gen_tau2.p - gen_polarim_tau2_closest.p')
plt.legend()
#
plt.show()
raise End

bf.plot_1d(gen_tau1_guess1.p, tau1_guess1.p, 'gen_tau1_guess1.p', 'tau1_guess1.p')



















## check whether SV is a good tau direction
gen_vis_1_dir = [gen_vis_1.p_x, gen_vis_1.p_y, gen_vis_1.p_z]
gen_vis_1_dir = gen_vis_1_dir/bf.norm(gen_vis_1_dir)

reco_vis_1_dir = [tau_1_vis.p_x, tau_1_vis.p_y, tau_1_vis.p_z]
reco_vis_1_dir = reco_vis_1_dir/bf.norm(reco_vis_1_dir)

gen_tau1_dir = [gen_tau_1.p_x, gen_tau_1.p_y, gen_tau_1.p_z]
gen_tau1_dir = gen_tau1_dir/bf.norm(gen_tau1_dir)


reco_tau1_dir = [tau_1.p_x, tau_1.p_y, tau_1.p_z]
reco_tau1_dir = reco_tau1_dir/bf.norm(reco_tau1_dir)


#checking the angle between a1 and tau1 gen level
gen_theta = bf.dot_product(gen_tau1_dir, gen_vis_1_dir)
reco_theta = bf.dot_product(reco_tau1_dir, reco_vis_1_dir)

norm_sv_1 = bf.norm([df4['sv_x_1'], df4['sv_y_1'], df4['sv_z_1']])
#remove the 0 lenght data otherwise we have nans
norm_sv_1 = np.where(norm_sv_1 == 0, 9999, norm_sv_1)
dir_x_tau1 = df4['sv_x_1']/norm_sv_1
dir_y_tau1 = df4['sv_y_1']/norm_sv_1
dir_z_tau1 = df4['sv_z_1']/norm_sv_1
sv_dir_1 = [dir_x_tau1, dir_y_tau1, dir_z_tau1]

polari_theta = bf.dot_product(sv_dir_1, gen_vis_1_dir)



#these are fine, the theta we calculated are fine
gen_theta_angle = np.arccos(gen_theta)
reco_theta_angle = np.arccos(reco_theta)
polari_theta_angle = np.arccos(polari_theta)


m_tau = 1.77686

#check theta max, maybe as a function of theta_gen
theta_max = np.arcsin((m_tau**2 - tau_1_vis.m**2)/(2*m_tau*tau_1_vis.p))
gen_theta_max = np.arcsin((m_tau**2 - gen_vis_1.m**2)/(2*m_tau*gen_vis_1.p))

reco_theta_angle = tf.where(reco_theta_angle >= theta_max, theta_max, reco_theta_angle)
polari_theta_angle = tf.where(polari_theta_angle >= theta_max, theta_max, polari_theta_angle)

gen_theta_angle = tf.where(gen_theta_angle >= gen_theta_max, gen_theta_max, gen_theta_angle)

















#now calculate p_tau:
#m_vis = tau_1_vis.m
#theta_GJ = polari_theta_angle
#minus_b = (m_vis**2 + m_tau**2) * tau_1_vis.p * np.cos(theta_GJ)
#two_a = 2*(m_vis**2 + tau_1_vis.p2 * (np.sin(theta_GJ))**2)
#b_squared_m_four_ac = (m_vis**2 + tau_1_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_1_vis.p2*(np.sin(theta_GJ))**2)
#b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)

#6) have the two solutions
#sol1_1 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
#sol2_1 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a

#closest_sol = np.where(abs(sol1_1-gen_tau_1.p)<=abs(sol2_1-gen_tau_1.p), sol1_1, sol2_1 )

#check solutions - which one is best 
#tau1_sol_1 = Momentum4(np.sqrt(sol1_1**2+m_tau**2), dir_x_tau1*sol1_1, dir_y_tau1*sol1_1, dir_z_tau1*sol1_1)
#tau1_sol_2 = Momentum4(np.sqrt(sol2_1**2+m_tau**2), dir_x_tau1*sol2_1, dir_y_tau1*sol2_1, dir_z_tau1*sol2_1) 

#nu_1_guess1 = tau1_sol_1 - tau_1_vis
#nu_1_guess2 = tau1_sol_2 - tau_1_vis















plt.title('Checking the angle between tau and visible - gen and sv approximation', fontsize = 'x-large')
plt.plot([0,2*np.pi*(180/np.pi)], [0,2*np.pi*(180/np.pi)], 'k--')
plt.hist2d(gen_theta_angle*(180/np.pi), polari_theta_angle*(180/np.pi), bins = (1500, 2000), norm=colors.LogNorm())
plt.xlabel('gen_theta_tau-a1(deg)')
plt.ylabel('sv_theta_tau-a1(deg)')
plt.colorbar()
plt.grid(alpha = 0.5)
plt.xlim(0,2.0)
plt.ylim(0,3.0)
plt.show()

hist = (np.array(gen_theta_angle) - np.array(polari_theta_angle))*(180/np.pi)
plt.title('Checking the angle between tau and visible - gen and sv approximation', fontsize = 'x-large')
plt.hist(hist, bins = 500, label = 'Mean diff: %.2f, std: %.2f'%(hist.mean(), hist.std()))
plt.xlabel('gen_theta_tau-a1 - sv_theta_tau-a1 (deg)')
plt.grid(alpha = 0.5)
plt.legend()
plt.show()

#raise end



plt.title('Checking the momenta of reco_tau and polarimetric_reco_tau', fontsize = 'x-large')
plt.hist2d(tau_1.p, closest_sol, bins = 1000, norm=colors.LogNorm())
plt.xlabel('reco_tau_1.p(Gev)')
plt.ylabel('reco_polarimetric_closest_sol(GeV)')
plt.xlim(0, 1000)
plt.ylim(0, 1000)
plt.plot([0, 1000], [0,1000], 'k--')
plt.grid(alpha = 0.5)
plt.show()



plt.title('Checking the momenta of gen_tau and best polarimetric_reco_tau', fontsize = 'x-large')
hist = (gen_tau_1.p - closest_sol)/gen_tau_1.p
plt.hist((gen_tau_1.p - closest_sol)/gen_tau_1.p, bins = 1000, label = 'Mean: %.2f, std: %.2f'%(np.array(hist).mean(), np.array(hist).std()))
plt.xlabel('(gen_tau.p - reco_polarimetric_closest_sol)/gen_tau.p')
plt.ylabel('Occurence')
plt.legend(prop ={'size':10})
plt.grid(alpha = 0.5)
plt.show()






plt.title('Checking the angle between tau and visible', fontsize = 'x-large')
plt.plot([0,2*np.pi*(180/np.pi)], [0,2*np.pi*(180/np.pi)], 'k--')
plt.hist2d(reco_theta_angle*(180/np.pi), theta_max*(180/np.pi), bins = 1000, norm=colors.LogNorm())
plt.xlabel('reco_theta_tau-a1(deg)')
plt.ylabel('theta_GJ_max(deg)')
plt.grid(alpha = 0.5)
plt.show()





### below: the graphs
plt.title('GEN Polarimetric method giving best Higgs mass - neutrino 1', fontsize = 'xx-large')
plt.subplot(2,2,1)
plt.hist2d(np.array(nu_1.p_x), np.array(gen_best_guess_nu1.p_x),  bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([-100, 100], [-100, 100], 'k--')
#plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_nu1_px')
plt.ylabel('gen_polarimetric_nu1_px')
plt.colorbar() 
plt.subplot(2,2,2)
plt.hist2d(np.array(nu_1.p_y), np.array(gen_best_guess_nu1.p_y), bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([-100, 100], [-100, 100], 'k--')
#plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_nu1_px')
plt.ylabel('gen_polarimetric_nu1_px')
plt.colorbar() 
plt.subplot(2,2,3)
plt.hist2d(np.array(nu_1.p_z), np.array(gen_best_guess_nu1.p_z), bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([-300, 300], [-100, 100], 'k--')
#plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_nu1_pz')
plt.ylabel('gen_polarimetric_nu1_pz')
plt.colorbar() 
plt.subplot(2,2,4)
plt.hist2d(np.array(nu_1.p), np.array(gen_best_guess_nu1.p), bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([0,500], [0, 500], 'k--')
#plt.xlim(0, 500)
plt.ylim(0, 500)
plt.xlabel('gen_nu1_p')
plt.ylabel('gen_polarimetric_nu1_p')
plt.colorbar() 
plt.show()


raise End

















plt.title('Checking the angle between tau and visible - sin?', fontsize = 'x-large')
plt.plot([0,2*np.pi*(180/np.pi)], [0,2*np.pi*(180/np.pi)], 'k--')
plt.hist2d(reco_theta_angle*(180/np.pi), theta_max*(180/np.pi), bins = 1000, norm=colors.LogNorm())
plt.xlabel('reco_theta_tau-a1(deg)')
plt.ylabel('theta_GJ_max(deg)')
plt.grid(alpha = 0.5)
plt.show()




plt.title('Checking the angle between tau and visible - sin?', fontsize = 'x-large')
plt.plot([0,2*np.pi*(180/np.pi)], [0,2*np.pi*(180/np.pi)], 'k--')
plt.hist2d(gen_theta_angle*(180/np.pi), reco_theta_angle*(180/np.pi), bins = 1000, norm=colors.LogNorm())
plt.xlabel('gen_theta_tau-a1(deg)')
plt.ylabel('reco_theta_tau-a1(deg)')
plt.grid(alpha = 0.5)












plt.title('Checking the angle between tau and visible', fontsize = 'x-large')
plt.plot([0,2*np.pi*(180/np.pi)], [0,2*np.pi*(180/np.pi)], 'k--')
plt.hist2d(gen_theta_angle*(180/np.pi), reco_theta_angle*(180/np.pi), bins = 1000, norm=colors.LogNorm())
plt.xlabel('gen_theta_tau-a1(deg)')
plt.ylabel('reco_theta_tau-a1(deg)')
plt.grid(alpha = 0.5)

plt.ylim(1.54*(180/np.pi),1.58*(180/np.pi))
plt.xlim(1.54*(180/np.pi),1.58*(180/np.pi))
plt.show()







plt.title('Checking the tau direction', fontsize = 'x-large')
perp_sv = bf.dot_product(gen_tau1_dir, sv_dir_1)
angle_gen_sv = np.arccos(np.array(perp_sv))*(180/np.pi)
plt.hist(angle_gen_sv, bins = 1000, label = 'Mean:%.2f '%angle_gen_sv.mean())
plt.xlabel('angle between gen_tau and sv_tau (deg)', fontsize = 'x-large')
#plt.xlim(-15, 15)
plt.legend(prop = {'size' : 16})
plt.grid()
plt.show()



plt.title('Checking the tau direction', fontsize = 'x-large')
perp_sv = bf.cross_product(gen_tau1_dir, sv_dir_1)
plt.hist2d(gen_tau_1.e, bf.norm(perp_sv), bins = 1000, norm=colors.LogNorm() )
plt.ylabel('Norm of gen_tau_dir_unit x sv_tau_dir_unit', fontsize = 'x-large')
plt.colorbar()
plt.xlabel('Gen tau energy (GeV)')

plt.xlim(0, 800)
#plt.legend(prop = {'size' : 16})
plt.grid()
plt.show()











### Check the tau direction - maybe we are in wrong frame
# do the cross product between the momenta if parallel, should be 0


reco_tau1_dir = np.array([tau_1.p_x, tau_1.p_y, tau_1.p_z])
reco_tau1_dir = reco_tau1_dir/bf.norm(reco_tau1_dir)


perp = bf.cross_product(gen_tau1_dir, reco_tau1_dir)
plt.plot(bf.norm(perp), 'x', label = 'Mean:%.2e '%np.array(bf.norm(perp)).mean())
plt.ylabel('Norm of gen_tau_dir_unit x reco_tau_dir_unit', fontsize = 'x-large')
plt.legend(prop = {'size' : 16})
plt.show()












#polarimetric tau with best mass - useless cause they have same mass...
m_tau = 1.77686
nu1_best_tau_m = np.where(abs(tau1_guess1.m-m_tau)<abs(tau1_guess2.m-m_tau), nu_1_guess1, nu_1_guess2)

nu1_best_tau_m = Momentum4(nu1_best_tau_m[0], nu1_best_tau_m[1], nu1_best_tau_m[2], nu1_best_tau_m[3])

tau1_best_mass = nu1_best_tau_m + tau_1_vis








### below: the graphs
plt.title('GEN Polarimetric method giving best Higgs mass - neutrino 1', fontsize = 'xx-large')
plt.subplot(2,2,1)
plt.hist2d(np.array(nu_1.p_x), np.array(gen_best_guess_nu1.p_x),  bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([-100, 100], [-100, 100], 'k--')
#plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_nu1_px')
plt.ylabel('gen_polarimetric_nu1_px')
plt.colorbar() 
plt.subplot(2,2,2)
plt.hist2d(np.array(nu_1.p_y), np.array(gen_best_guess_nu1.p_y), bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([-100, 100], [-100, 100], 'k--')
#plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_nu1_px')
plt.ylabel('gen_polarimetric_nu1_px')
plt.colorbar() 
plt.subplot(2,2,3)
plt.hist2d(np.array(nu_1.p_z), np.array(gen_best_guess_nu1.p_z), bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([-300, 300], [-100, 100], 'k--')
#plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_nu1_pz')
plt.ylabel('gen_polarimetric_nu1_pz')
plt.colorbar() 
plt.subplot(2,2,4)
plt.hist2d(np.array(nu_1.p), np.array(gen_best_guess_nu1.p), bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([0,500], [0, 500], 'k--')
#plt.xlim(0, 500)
plt.ylim(0, 500)
plt.xlabel('gen_nu1_p')
plt.ylabel('gen_polarimetric_nu1_p')
plt.colorbar() 
plt.show()



















raise End

def overall_check(guess_1, guess_2, true):
    p_x_diff1 = abs(guess_1[1]-true[1])
    p_x_diff2 = abs(guess_2[1]-true[1])
    p_y_diff1 = abs(guess_1[2]-true[2])
    p_y_diff2 = abs(guess_2[2]-true[2])
    p_z_diff1 = abs(guess_1[3]-true[3])
    p_z_diff2 = abs(guess_2[3]-true[3])
    w = 0
    if p_x_diff1 < p_x_diff2 and p_y_diff1 < p_y_diff2 and  p_z_diff1 < p_z_diff2 :
        w = 1
        #print('1 overall best')
        return[np.sqrt(guess_1[1]**2+guess_1[2]**2+guess_1[3]**2), guess_1[1], guess_1[2], guess_1[3]]
    
    if p_x_diff1 > p_x_diff2 and p_y_diff1 > p_y_diff2 and  p_z_diff1 > p_z_diff2 :
        w = 1
        #print('2 overall best')
        return [np.sqrt(guess_2[1]**2+guess_2[2]**2+guess_2[3]**2), guess_2[1], guess_2[2], guess_2[3]]
    
    
    if p_x_diff1+p_y_diff1+p_z_diff1 < p_x_diff2+p_y_diff2+p_z_diff2 and w!=1:
        w = 1
        return [np.sqrt(guess_1[1]**2+guess_1[2]**2+guess_1[3]**2), guess_1[1], guess_1[2], guess_1[3]]
    
    if w == 0:
        return [np.sqrt(guess_2[1]**2+guess_2[2]**2+guess_2[3]**2), guess_2[1], guess_2[2], guess_2[3]]
    
    #if p_x_diff1 < p_x_diff2 :
        #if p_y_diff1 < p_y_diff2 :
            #return Momentum4( np.sqrt(guess_1[1]+guess_1[2]+guess_1[3]), guess_1[1], guess_1[2], guess_1[3])
        #if p_y_diff1 > p_y_diff2 :
            #if p_z_diff1 < p_z_diff2 :
                #return Momentum4( np.sqrt(guess_1[1]+guess_1[2]+guess_1[3]), guess_1[1], guess_1[2], guess_1[3])
            #if p_z_diff1 > p_z_diff2 :
                #return Momentum4( np.sqrt(guess_1[1]+guess_1[2]+guess_1[3]), guess_1[1], guess_1[2], guess_1[3])




# check the best ever polarimetric method:
closest_guesses = []

for i in range(100000):
    closest_guesses.append(overall_check(np.transpose(gen_nu_1_guess1)[i], np.transpose(gen_nu_1_guess2)[i], np.transpose(nu_1)[i]))


closest_guesses = np.transpose(closest_guesses)
closest_guesses = Momentum4(closest_guesses[0], closest_guesses[1], closest_guesses[2], closest_guesses[3])

























#############################################
plt.hist2d(nu_1.p[:100000], closest_guesses.p, 100, norm=colors.LogNorm())
plt.xlabel('gen_nu1.px')
plt.plot([0, 2000], [0, 2000], 'k--')
plt.ylim(0,400)
plt.xlim(0,400)
plt.ylabel('closest_gen_polarimetric_nu1.p')
plt.colorbar()
plt.show()



diff_x = np.array(nu_1.p_x[:100000]-closest_guesses.p_x)
diff_y = np.array(nu_1.p_y[:100000]-closest_guesses.p_y)
diff_z = np.array(nu_1.p_z[:100000]-closest_guesses.p_z)

plt.hist(diff_x, alpha = 0.5, bins = 1000, label = 'px Mean diff: %.2f, std: %.2f'%(diff_x.mean(), diff_x.std()))
plt.hist(diff_y, alpha = 0.5, bins = 1000, label = 'py Mean diff: %.2f, std: %.2f'%(diff_y.mean(), diff_y.std()))
plt.hist(diff_z, alpha = 0.5, bins = 1000, label = 'pz Mean diff: %.2f, std: %.2f'%(diff_z.mean(), diff_z.std()))
plt.grid()
plt.xlim(-100, 100)
plt.xlabel('nu_p{x,y,z} - closest_gen_polarimetric_nu1.p{x,y,z}')
plt.legend(prop = {'size':16})
plt.show()



plt.hist2d(nu_1.p, gen_best_guess_nu1.p, 100, norm=colors.LogNorm())
plt.xlabel('gen_nu1.px')
plt.ylabel('gen_polarimetric_nu1.p')
plt.colorbar()
plt.show()














######historgrams 


plt.hist2d(tau_2.m, tau_2_vis.e, 100, norm=colors.LogNorm())
plt.xlabel('pseudo_tau_2.m')
plt.ylabel('reco_visible2.E')
plt.colorbar()
plt.show()

plt.hist2d(tau_1.m, tau_1_vis.e, 100, norm=colors.LogNorm())
plt.xlabel('pseudo_tau_1.m')
plt.ylabel('reco_visible1.E')
plt.colorbar()
plt.show()




plt.plot(tau1_guess1.m, tau1_guess2.m, 'x')
plt.xlabel('tau1_guess1.m')
plt.ylabel('tau1_guess2.m')
plt.grid()
plt.show()

plt.plot(tau1_guess1.p, tau1_guess2.p, 'x')
plt.plot([0, 2000], [0, 2000], 'k--')
plt.xlabel('tau1_guess1.p')
plt.ylabel('tau1_guess2.p')
plt.grid()
plt.show()





### below: the graphs
plt.title('GEN Polarimetric method giving best Higgs mass - neutrino 1', fontsize = 'xx-large')
plt.subplot(2,2,1)
plt.hist2d(np.array(nu_1.p_x), np.array(gen_best_guess_nu1.p_x),  bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([-100, 100], [-100, 100], 'k--')
#plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_nu1_px')
plt.ylabel('gen_polarimetric_nu1_px')
plt.colorbar() 
plt.subplot(2,2,2)
plt.hist2d(np.array(nu_1.p_y), np.array(gen_best_guess_nu1.p_y), bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([-100, 100], [-100, 100], 'k--')
#plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_nu1_px')
plt.ylabel('gen_polarimetric_nu1_px')
plt.colorbar() 
plt.subplot(2,2,3)
plt.hist2d(np.array(nu_1.p_z), np.array(gen_best_guess_nu1.p_z), bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([-300, 300], [-100, 100], 'k--')
#plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_nu1_pz')
plt.ylabel('gen_polarimetric_nu1_pz')
plt.colorbar() 
plt.subplot(2,2,4)
plt.hist2d(np.array(nu_1.p), np.array(gen_best_guess_nu1.p), bins=(100, 800), norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.plot([0,500], [0, 500], 'k--')
#plt.xlim(0, 500)
plt.ylim(0, 500)
plt.xlabel('gen_nu1_p')
plt.ylabel('gen_polarimetric_nu1_p')
plt.colorbar() 
plt.show()




### below: the graphs
plt.title('visible decay product reco', fontsize = 'xx-large')
plt.subplot(2,2,1)
plt.hist2d(np.array(gen_vis_1.p_x), np.array(tau_1_vis.p_x), bins = 100, norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_vis1_px')
plt.ylabel('reco_vis1_px')
plt.colorbar() 
plt.subplot(2,2,2)
plt.hist2d(np.array(gen_vis_1.p_y), np.array(tau_1_vis.p_y), bins = 100, norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_vis1_py')
plt.ylabel('reco_vis_1_py')
plt.colorbar() 
plt.subplot(2,2,3)
plt.hist2d(np.array(gen_vis_1.p_z), np.array(tau_1_vis.p_z), bins = 100, norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.xlim(-300, 300)
plt.ylim(-300, 300)
plt.xlabel('gen_vis_1_pz')
plt.ylabel('reco_vis_1_pz')
plt.colorbar() 
plt.subplot(2,2,4)
plt.hist2d(np.array(gen_vis_1.e), np.array(tau_1_vis.e), bins = 100, norm=colors.LogNorm())
plt.grid(alpha = 0.5)
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.xlabel('gen_vis_1.E')
plt.ylabel('reco_vis_1.E')
plt.colorbar() 
plt.show()


plt.hist(np.clip((np.array(gen_vis_1.p_z) - np.array(tau_1_vis.p_z))/np.array(gen_vis_1.p_z), -0.5, 0.5), bins = 1000)
plt.xlabel('(gen_vis_pz-reco_vis1_pz)/gen_vis_pz')
plt.grid()
plt.show()
