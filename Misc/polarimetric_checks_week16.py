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
tau_mode3 = 1
decay_mode3 = 0

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


tree3 = uproot.open("/home/acraplet/Alie/Masters/MVAFILE_full_10_10_pola12_pv.root")["ntuple"]
#tree3 = uproot.open("/home/acraplet/Alie/Masters/MVAFILE_full_10_X.root")["ntuple"]

#tree = uproot.open("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/MVAFILE_AllHiggs_tt_reco_10_10_phitt.root")["ntuple"]
#tree = uproot.open("/eos/user/d/dwinterb/SWAN_projects/Masters_CP/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
print("\n Tree loaded\n")



# define what variables are to be read into the dataframe
momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              #"pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              #"pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2",
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


other_features = [ #"ip_x_1", "ip_y_1", "ip_z_1",        #leading impact parameter
                   #"ip_x_2", "ip_y_2", "ip_z_2",        #subleading impact parameter
                   #"y_1_1", "y_1_2",
                   "ip_sig_1", "ip_sig_2",
                   "gen_phitt", #"pseudo_phitt",
                   #'pola_nb_shit',
                   'pola5_pv_angle',
                   "pv_angle", #"ippv_angle",
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

df3 = df3[
     (df3["pv_angle"] > -4000)
     & (df3["aco_angle_1"] > -4000)
     &(df3["tau_decay_mode_2"] == 10) 
     &(df3["mva_dm_2"] == 10) 
     &(df3["tau_decay_mode_1"] == 10) 
     &(df3["mva_dm_1"] == 10) 
     &(df3["gen_nu_p_1"] > -4000)
     &(df3["gen_nu_p_2"] > -4000)
    #& (df4["mva_dm_2"] == decay_mode2)
     &(df3['sv_x_1'] != 0)
     &(df3['sv_x_2'] != 0)
    ]

#trainFrac = 0.7
#df_train, df3 = np.split(df3, [int(trainFrac*len(df3))], axis=0)

def make_ps_sm(df4):
    df_ps = df4[
        (df4["rand"]<df4["wt_cp_ps"]/2)     #a data frame only including the pseudoscalars
    ]

    df_sm = df4[
        (df4["rand"]<df4["wt_cp_sm"]/2)     #data frame only including the scalars
    ]
    return df_ps, df_sm

   
def improvement(df, variable_new, variable_old, nbins = 4):
    df_ps, df_sm = make_ps_sm(df)
    #bins_array = np.linspace(0, 2*np.pi, nbins+1)
    #print(bins_array)
    a = df[variable_old]
    list_bins = []
    for i in range (nbins):
        list_bins.append(np.quantile(a, (i+1)*1/nbins))
        print((i+1)*1/nbins)
    bins_array = np.array(list_bins)
    
    #print(bins_array)
    
    hist_sm_new= np.histogram(df_sm[variable_new], bins = bins_array)
    hist_ps_new = np.histogram(df_ps[variable_new], bins = bins_array)
    
    hist_sm_old= np.histogram(df_sm[variable_old], bins = bins_array)
    hist_ps_old = np.histogram(df_ps[variable_old], bins = bins_array)
    
    sep_new = np.array(hist_sm_new[0]/hist_ps_new[0])
    sep_old = np.array(hist_sm_old[0]/hist_ps_old[0])
    
    if variable_old == 'aco_angle_1':
        sep_old = np.array(hist_ps_old[0]/hist_sm_old[0])
    print(min(sep_old), min(sep_new), sep_old.mean())
    
     #
    return -(min(sep_old)-min(sep_new))/(min(sep_old)-1)#(1-min(sep_new))/(1-min(sep_old))
        

nbins = 4

print('a1-a1 channel improvement bins = 4', improvement(df3, 'pv_angle', 'aco_angle_1', nbins))
#print('a1-a1 channel improvement bins = 4', improvement(df3, 'pola5_pv_angle', 'pv_angle', nbins))


def sm_ps(angle, marker, df, nbins, name = 0):
    df_ps, df_sm = make_ps_sm(df)
    bins_array = np.linspace(0, 2*np.pi , nbins+1)
    zeros = np.linspace(0, 0, nbins+1)+1

    a = df[angle]
    list_bins = [0]
    for i in range (nbins):
        list_bins.append(np.quantile(a, (i+1)*1/nbins))
    bins_array = np.array(list_bins)

    hist_sm = np.histogram(df_sm[angle], bins = bins_array)
    hist_ps = np.histogram(df_ps[angle], bins = bins_array)
    if angle == 'aco_angle_1' or angle == 'aco_angle_5':
        #name = '1/' + angle
        plt.plot(bins_array[:-1], hist_ps[0]/hist_sm[0], marker, alpha = 0.7, label = name)
    else:
        if name == 0:
            name = angle
        plt.plot(bins_array[:-1], hist_sm[0]/hist_ps[0], marker, alpha = 0.7, label = name)
    plt.plot(bins_array[:-1], zeros[:-1], 'k-')

#sm_ps('pv_angle', 'b--', df3, nbins, 'pv_angle with 4 bins')
#sm_ps('aco_angle_1', 'r--', df3, nbins, 'aco_angle_1 with 4 bins')

nbins = 4
plt.title('a1-a1 channel CP-sensitivity improvements')
sm_ps('aco_angle_1', 'r-', df3, nbins, 'aco_angle_1 with '+str(nbins)+' bins')
plt.xlabel ('angle (rad)')
sm_ps('pv_angle', 'b-', df3, nbins, 'pv_angle with '+str(nbins)+' bins\n improvement from aco_angle1: '+str(improvement(df3, 'pv_angle', 'aco_angle_1', nbins))[:5])

sm_ps('pola5_pv_angle', 'g-', df3, nbins, 'pola5_pv_angle with '+str(nbins)+' bins\n improvement from aco_angle_1: '+str(improvement(df3, 'pola5_pv_angle', 'aco_angle_1', nbins))[:5])

plt.ylabel('ps/sm distributions')
plt.grid()
plt.legend()
plt.show()


raise END

df4 = df4[(df4["mva_dm_1"] == 10) 
    & (df4["mva_dm_2"] == 0)
    & (df4["aco_angle_5"] > -4000)
    & (df4["ippv_angle"] > -4000)
    ]

df40 = df40[(df40["mva_dm_1"] == 0) 
    & (df40["mva_dm_2"] == 10)
    & (df40["aco_angle_5"] > -4000)
    & (df40["ippv_angle"] > -4000)
    ]




df6 = df4[
      #(df4["tau_decay_mode_1"] == tau_mode1) 
   #& (df4["tau_decay_mode_2"] == tau_mode2) 
     (df4["mva_dm_1"] == 10) 
    & (df4["mva_dm_2"] == 0)
    & (df4["aco_angle_5"] > -4000)
    & (df4["ippv_angle"] > -4000)
    & (df4["ip_sig_2"] > 1.5)
    #& (df4["pola_nb_shit"] == 0)
    #& (df4["gen_nu_p_2"] > -4000)
    #& (df4["pi_E_1"] != 0)
    #& (df4["pi_E_2"] != 0)
    #& (df4["sv_x_1"] != 0)
    #& (df4["sv_x_2"] != 0)
]

df7 = df4[
      #(df4["tau_decay_mode_1"] == tau_mode1) 
   #& (df4["tau_decay_mode_2"] == tau_mode2) 
     (df4["mva_dm_1"] == 10) 
    & (df4["mva_dm_2"] == 0)
    & (df4["aco_angle_5"] > -4000)
    & (df4["ippv_angle"] > -4000)
    & (df4["ip_sig_2"] < 1.5)
    #& (df4["pola_nb_shit"] == 0)
    #& (df4["gen_nu_p_2"] > -4000)
    #& (df4["pi_E_1"] != 0)
    #& (df4["pi_E_2"] != 0)
    #& (df4["sv_x_1"] != 0)
    #& (df4["sv_x_2"] != 0)
]
df60 = df40[
      #(df4["tau_decay_mode_1"] == tau_mode1) 
   #& (df4["tau_decay_mode_2"] == tau_mode2) 
     (df40["mva_dm_1"] == 0) 
    & (df40["mva_dm_2"] == 10)
    & (df40["aco_angle_5"] > -4000)
    & (df40["ippv_angle"] > -4000)
    & (df40["ip_sig_1"] > 1.5)
    #& (df4["pola_nb_shit"] == 0)
    #& (df4["gen_nu_p_2"] > -4000)
    #& (df4["pi_E_1"] != 0)
    #& (df4["pi_E_2"] != 0)
    #& (df4["sv_x_1"] != 0)
    #& (df4["sv_x_2"] != 0)
]

df70 = df40[
      #(df4["tau_decay_mode_1"] == tau_mode1) 
   #& (df4["tau_decay_mode_2"] == tau_mode2) 
     (df40["mva_dm_1"] == 0) 
    & (df40["mva_dm_2"] == 10)
    & (df40["aco_angle_5"] > -4000)
    & (df40["ippv_angle"] > -4000)
    & (df40["ip_sig_1"] < 1.5)
    #& (df4["pola_nb_shit"] == 0)
    #& (df4["gen_nu_p_2"] > -4000)
    #& (df4["pi_E_1"] != 0)
    #& (df4["pi_E_2"] != 0)
    #& (df4["sv_x_1"] != 0)
    #& (df4["sv_x_2"] != 0)
]




def make_ps_sm(df4):
    df_ps = df4[
        (df4["rand"]<df4["wt_cp_ps"]/2)     #a data frame only including the pseudoscalars
    ]

    df_sm = df4[
        (df4["rand"]<df4["wt_cp_sm"]/2)     #data frame only including the scalars
    ]
    return df_ps, df_sm


df_ps, df_sm = make_ps_sm(df4) 

df6_ps, df6_sm = make_ps_sm(df6) 
df7_ps, df7_sm = make_ps_sm(df7) 

df0_ps, df0_sm = make_ps_sm(df40) 

df60_ps, df60_sm = make_ps_sm(df60) 
df70_ps, df70_sm = make_ps_sm(df70) 




def sm_ps_hist(angle, df_sm, df_ps):
    name_sm = angle + "_sm" 
    name_ps = angle + "_ps"
    plt.hist(df_ps[angle], bins = 50, alpha = 0.7, label = name_ps)
    plt.hist(df_sm[angle], bins = 50, alpha = 0.7, label = name_sm)
    plt.grid()
    
def sm_ps(angle, marker, df_ps, df_sm, name = 0):
    bins_array = np.linspace(0, 2*np.pi, 16)
    zeros = np.linspace(0, 0, 16)+1
    hist_sm = np.histogram(df_sm[angle], bins = bins_array)
    hist_ps = np.histogram(df_ps[angle], bins = bins_array)
    if angle == 'aco_angle_5':
        #name = '1/' + angle
        plt.plot(bins_array[:-1], hist_ps[0]/hist_sm[0], marker, alpha = 0.7, label = name)
    else:
        if name == 0:
            name = angle
        plt.plot(bins_array[:-1], hist_sm[0]/hist_ps[0], marker, alpha = 0.7, label = name)
    plt.plot(bins_array[:-1], zeros[:-1], 'k-')
    
    
def improvement(df_ps, df_sm, variable_new, variable_old, nbins = 4):
    bins_array = np.linspace(0, 2*np.pi, nbins)
    hist_sm_new= np.histogram(df_sm[variable_new], bins = bins_array)
    hist_ps_new = np.histogram(df_ps[variable_new], bins = bins_array)
    
    hist_sm_old= np.histogram(df_sm[variable_old], bins = bins_array)
    hist_ps_old = np.histogram(df_ps[variable_old], bins = bins_array)
    
    sep_new = np.array(hist_sm_new[0]/hist_ps_new[0])
    sep_old = np.array(hist_sm_old[0]/hist_ps_old[0])
    
    if variable_old == 'aco_angle_5':
        sep_old = np.array(hist_ps_old[0]/hist_sm_old[0])
    print(min(sep_old), min(sep_new))
    return -(min(sep_old)-min(sep_new))/(min(sep_old)-1)
        

print(len(df6['ippv_angle'])/len(df4['ippv_angle']))
print('a1-pi channel improvement bins = 15', improvement(df6_ps, df6_sm, 'ippv_angle', 'aco_angle_5', 15))
print('pi-a1 channel improvement bins = 15',improvement(df60_ps, df60_sm, 'ippv_angle', 'aco_angle_5', 15))


plt.title('a1-pi channel ippv method', fontsize = 'xx-large')
#sm_ps('aco_angle_5', 'm-.', df7_sm, df7_ps, 'a1-pi aco_angle_5 IP < 1.5')
sm_ps('aco_angle_5', 'm--', df_sm, df_ps, 'a1-pi aco_angle_5 no IP cut')
sm_ps('aco_angle_5', 'm-', df6_sm, df6_ps, 'a1-pi aco_angle_5 IP > 1.5')

#sm_ps('ippv_angle', 'r-.', df7_sm, df7_ps, 'a1-pi ippv_angle IP < 1.5')
sm_ps('ippv_angle', 'r--', df_sm, df_ps, 'a1-pi ippv_angle no IP_sig cut')
sm_ps('ippv_angle', 'r-', df6_sm, df6_ps, 'a1-pi ippv_angle no IP_sig > 1.5')
#sm_ps('aco_angle_5', 'b-.', df70_sm, df70_ps, 'pi-a1 aco_angle_5 IP < 1.5')
sm_ps('aco_angle_5', 'b--', df0_sm, df0_ps, 'pi-a1 aco_angle_5 no IP cut')
sm_ps('aco_angle_5', 'b-', df60_sm, df60_ps, 'pi-a1 aco_angle_5 IP > 1.5')
#sm_ps('ippv_angle', 'c-.', df70_sm, df70_ps, 'pi-a1 ippv_angle no IP_sig < 1.5')
sm_ps('ippv_angle', 'c--', df0_sm, df0_ps, 'pi-a1 ippv_angle no IP_sig cut')
sm_ps('ippv_angle', 'c-', df60_sm, df60_ps, 'pi-a1 ippv_angle no IP_sig > 1.5')
#sm_ps('ippv_angle', 'b-.', df7_sm, df7_ps, 'pi-a1 ippv_angle no IP_sig < 1.5')
#sm_ps('ippv2_angle', '-', df_sm, df_ps, 'pi-a1 ippv2_angle - 1 shift')
#sm_ps('ippv2_angle', '-', df5_sm, df5_ps, 'pi-a1 ippv2_angle - 0 shifts')
#sm_ps('ippv_angle', '--', df_sm, df_ps, 'pi-a1 ippv_angle - 1 shift')
#sm_ps('pseudo_ippv_angle', 'g-', df_sm, df_ps, 'pi-a1 pseudo_ippv_angle')
#sm_ps('pseudo2_ippv_angle', 'm-', df_sm, df_ps, 'pi-a1 pseudo_ippv_angle')
#sm_ps('ippv2_angle', 'g-', df_sm, df_ps, 'pi-a1 ippv2_angle')
#sm_ps('ippv8_angle', 'r-.', df_sm, df_ps, 'pi-a1 ippv8_angle')
#sm_ps('aco_angle_5', 'r--', df_sm, df_ps, 'a1-pi aco_angle_5')
#sm_ps('ippv_angle', 'r-', df_sm, df_ps, 'a1-pi ippv_angle')
plt.xlabel('angle(rad)', fontsize = 'x-large')
plt.ylabel('ps-sm separation', fontsize = 'x-large')
plt.grid()
plt.legend(prop = {'size' : 11})
plt.show()



plt.title('Relevant ip_sig in a1-pi and pi-a1 channel', fontsize = 'xx-large')
plt.hist(df4['ip_sig_2'], bins = 100, alpha = 0.7, label = 'a1-pi ip_sig_2 F(ip_sig>1.5)=%.2f'%(len(df6['ippv_angle'])/len(df4['ippv_angle'])))
plt.hist(df40['ip_sig_1'], bins = 100, alpha = 0.7, label = 'pi-a1 ip_sig_1 F(ip_sig>1.5)=%.2f'%(len(df60['ippv_angle'])/len(df40['ippv_angle'])))
plt.grid()
plt.xlabel('ip_sig', fontsize = 'x-large')
plt.ylabel('Occurences', fontsize = 'x-large')
plt.legend(prop = {'size' : 12})
plt.show()



raise END
#plt.subplot(2,2,3)
#plt.title('pi-a1 channel aco_angle_5')
#sm_ps_hist('aco_angle_5', df5_sm, df5_ps)
#plt.ylabel('Occurences pi-a1')
#plt.ylim(0,3500)
#plt.legend(loc=8)
#plt.xlabel('aco_angle_5')
plt.subplot(2,2,3)
#plt.title('pi-a1 channel new pv method')
sm_ps_hist('ippv2_angle',df_sm, df_ps)
plt.xlabel('ippv2_angle')
plt.ylim(0,3500)
plt.legend(loc=8)
plt.subplot(2,2,4)
#plt.title('pi-a1 channel new pv method')
sm_ps_hist('ippv8_angle',df_sm, df_ps)
plt.xlabel('ippv8_angle')
plt.ylim(0,3500)
plt.legend(loc=8)


plt.subplot(2,2,1)
plt.title('a1-pi channels aco_angle_5')
sm_ps_hist('aco_angle_5', df_sm, df_ps)
plt.ylabel('Occurences a1 - pi')
plt.ylim(0,4000)
plt.legend(loc=8)
plt.xlabel('aco_angle_5')
plt.subplot(2,2,2)
plt.title('a1-pi channels new ippv method')
sm_ps_hist('ippv_angle',df_sm, df_ps)
plt.xlabel('ippv_angle')
plt.ylim(0,4000)
plt.legend(loc=8)
plt.show()


raise end







    
plt.figure()
#sm_ps('pv_angle', 'r-')
#sm_ps('pseudo_pv_angle', '--')
sm_ps('aco_angle_5', '--')
#sm_ps('pola0_pv_angle', '--')
#sm_ps('pola1_pv_angle', '-')
#sm_ps('pola2_pv_angle', 'm-')
sm_ps('pola3_pv_angle', 'g-')
#sm_ps('pola4_pv_angle', '-')
sm_ps('pola5_pv_angle', 'r-')
#sm_ps('pola6_pv_angle', '-')
#sm_ps('pola7_pv_angle', 'm-')
sm_ps('pola8_pv_angle', 'b-.')
plt.ylabel('SM/PS distributions')
plt.title('Improvements to the polarimetric Method')
plt.xlabel('pv_angle (rad)')
plt.grid()
plt.legend()
plt.show()

nu_1 = Momentum4(np.zeros(len(df4["gen_nu_phi_1"])), df4["gen_nu_eta_1"], df4["gen_nu_phi_1"], df4["gen_nu_p_1"])

nu_2 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_2"])), df4["gen_nu_eta_2"], df4["gen_nu_phi_2"], df4["gen_nu_p_2"])

gen_metx = nu_1.p_x + nu_2.p_y

bf.plot_2d(df4["metx"], np.clip((gen_metx - df4["metx"])/gen_metx, -100, 100), 'reco_metx', '(gen_metx-reco_metx)/gen_metx (clipped -100, 100)', (-500, 500), (-500, 500), (1000, 1000) )

plt.title('Reco_metx - gen_metx')
plt.hist(df4["metx"]-gen_metx, bins = 1000, label = 'mean %.2f' %(np.array(df4["metx"]-gen_metx).mean()))
plt.xlabel('Reco_metx - gen_metx')
plt.legend()
plt.grid()
plt.show()

raise End

def sm_ps_hist(angle, df_sm, df_ps):
    name_sm = angle + "_sm" 
    name_ps = angle + "_ps"
    plt.hist(df_ps[angle], bins = 50, alpha = 0.7, label = name_ps)
    
    plt.hist(df_sm[angle], bins = 50, alpha = 0.7, label = name_sm)
    plt.grid()

plt.subplot(2,2,1)
plt.title('a1-pi channel')
sm_ps_hist('aco_angle_1', df5_sm, df5_ps)
plt.ylabel('Occurences')
plt.ylim(0,250)
plt.legend()
plt.xlabel('aco_angle_1')
plt.subplot(2,2,2)
plt.title('a1-pi channel')
sm_ps_hist('pola0_pv_angle',df5_sm, df5_ps)
plt.xlabel('pv_angle_lab_frame')
plt.ylim(0,250)
plt.legend()
plt.show()

#plt.subplot(2,2,3)
#plt.title('1 tau needing shift')
#plt.ylabel('Occurences')
#plt.xlabel('pv_angle')
#sm_ps_hist('pv_angle',df1_sm, df1_ps)
#plt.subplot(2,2,4)
#plt.title('2 taus needing shift')
#sm_ps_hist('pv_angle',df2_sm, df2_ps)
#plt.xlabel('pv_angle')
plt.show()


plt.subplot(2,2,1)
plt.title('All - with best possible tau shift')
sm_ps_hist('pola5_pv_angle', df_sm, df_ps)
plt.ylabel('Occurences')
plt.subplot(2,2,2)
plt.title('No tau needing shift')
sm_ps_hist('pola5_pv_angle',df0_sm, df0_ps)
plt.subplot(2,2,3)
plt.title('1 tau shifted')
plt.ylabel('Occurences')
plt.xlabel('pola5_pv_angle')
sm_ps_hist('pola5_pv_angle',df1_sm, df1_ps)
plt.subplot(2,2,4)
plt.title('2 taus shifted')
sm_ps_hist('pola5_pv_angle',df2_sm, df2_ps)
plt.xlabel('pola5_pv_angle')
plt.show()

print('\nFraction of events without need for tau shift: %.4f' % float(len(df0)/len(df4)))

print('\nFraction of events with need for 1 tau shift:  %.4f' % float(len(df1)/len(df4)))

print('\nFraction of events with need for 2 tau shifts:  %.4f' % float(len(df2)/len(df4)))


def sm_ps_theta(angle, marker, df_sm, df_ps, nb_shift, name = 0):
    bins_array = np.linspace(0, 2*np.pi, 21)
    zeros = np.linspace(0, 0, 21)+1
    hist_sm = np.histogram(df_sm[angle], bins = bins_array)
    hist_ps = np.histogram(df_ps[angle], bins = bins_array)
    if nb_shift != 4 and name == 0:
        name = angle + " "+ str(nb_shift)+ " shift(s)"
    if nb_shift == 4 and name == 0:
        name = angle + " full"
    if angle == 'aco_angle_1':
        plt.plot(bins_array[:-1], hist_ps[0]/hist_sm[0], marker, alpha = 0.5, label = name)
    if angle != 'aco_angle_1':
        plt.plot(bins_array[:-1], hist_sm[0]/hist_ps[0], marker, alpha = 0.5, label = name)
    #plt.plot(bins_array[:-1], zeros[:-1], 'k-')
    
#pola5_
plt.title('Comparision in a1a1 channel: decay products frame or Higgs rest frame', fontsize = 'x-large')
plt.ylabel('SM/PS distributions', fontsize = 'x-large')
plt.xlabel('pv_angle (rad)', fontsize = 'x-large')
#sm_ps_theta('pola0_pv_angle', 'm--', df0_sm, df0_ps, 0)
#sm_ps_theta('pola0_pv_angle', 'g--', df1_sm, df1_ps, 1)
#sm_ps_theta('pola0_pv_angle', 'r--', df2_sm, df2_ps, 2)
sm_ps_theta('pola0_pv_angle', 'b-', df_sm, df_ps, 4, 'Initial pv_angle in Higgs rest frame')
sm_ps_theta('pola5_pv_angle', 'b--', df_sm, df_ps, 4, 'pv_angle after re-shift in Higgs rest frame')

sm_ps_theta('aco_angle_1', 'g-', df_sm, df_ps, 4, 'aco_angle a1-a1')

sm_ps_theta('aco_angle_1', 'r--', df5_sm, df5_ps, 4, 'aco_angle a1-pi')
#sm_ps_theta('pola2_pv_angle', 'm-', df0_sm, df0_ps, 0)
#sm_ps_theta('pola2_pv_angle', 'g-', df1_sm, df1_ps, 1)
#sm_ps_theta('pola2_pv_angle', 'r-', df2_sm, df2_ps, 2)
sm_ps_theta('pola0_pv_angle', 'r-', df5_sm, df5_ps, 4, 'Initial pv_angle in visible decay products frame - a1-pi')
plt.grid()
plt.legend()
plt.show()

nu_2 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_2"])), df4["gen_nu_eta_2"], df4["gen_nu_phi_2"], df4["gen_nu_p_2"])

nu_2_guess = Momentum4.m_eta_phi_p(np.zeros(len(df4["pola5_nu_phi_2"])), df4["pola5_nu_eta_2"], df4["pola5_nu_phi_2"], df4["pola5_nu_p_2"])


bf.plot_2d(nu_2.p_z, nu_2_guess.p_z, 'gen_nu2_pz', 'polarimetric_nu2_pz', (-200, 200), (-200, 200), (300, 300))





raise End

def plot_1D(variable, number, level):
    
    mean_std = []
    exec('mean_std.append(np.array(gen_nu_%s.%s - %s_nu_%s.%s).mean())'%(number, variable, level, number, variable))
    exec('mean_std.append(np.array(gen_nu_%s.%s - %s_nu_%s.%s).std())'%(number, variable, level, number, variable))
    exec('plt.hist(gen_nu_%s.%s - %s_nu_%s.%s, bins = 100, alpha = 0.3, label = "gen-%s nu%s_%s, mean=%.2f, std=%.2f")'%(number, variable, level, number, variable, level, number, variable, mean_std[0], mean_std[1]))

gen_nu_1 = Momentum4.m_eta_phi_p(df4['gen_nu_p_1']*0, df4['gen_nu_eta_1'], df4['gen_nu_phi_1'], df4['gen_nu_p_1'])
gen_nu_2 = Momentum4.m_eta_phi_p(df4['gen_nu_p_2']*0, df4['gen_nu_eta_2'], df4['gen_nu_phi_2'], df4['gen_nu_p_2'])


pola3_nu_1 = Momentum4.m_eta_phi_p(df4['pola3_nu_p_1']*0, df4['pola3_nu_eta_1'], df4['pola3_nu_phi_1'], df4['pola3_nu_p_1'])
pola3_nu_2 = Momentum4.m_eta_phi_p(df4['pola3_nu_p_2']*0, df4['pola3_nu_eta_2'], df4['pola3_nu_phi_2'], df4['pola3_nu_p_2'])



plot_1D('p_x', '1', 'pola3')
plot_1D('p_y', '1', 'pola3')
plot_1D('p_z', '1', 'pola3')
plot_1D('p_x', '2', 'pola3')
plot_1D('p_y', '2', 'pola3')
plot_1D('p_z', '2', 'pola3')
plt.show()



plt.title('PS and SM distributions of pv angles')
plt.subplot(2,2,1)
plt.hist(df_sm['pv_angle'], alpha = 0.5, bins = 50, label = 'pv_angle root')
plt.hist(df_ps['pv_angle'], alpha = 0.5, bins = 50, label = 'pv_angle root')
plt.xlabel('pv_angle (rad)')
plt.ylim(0, 1500)
plt.grid()
plt.legend()

plt.subplot(2,2,2)
plt.hist(df_sm['pola2_pv_angle'], alpha = 0.5, bins = 50, label = 'sm pola2 pv-angle')
plt.hist(df_ps['pola2_pv_angle'], alpha = 0.5, bins = 50, label = 'ps pola2 pv-angle')
plt.xlabel('pv_angle (rad)')
plt.ylim(0, 2000)
plt.grid()
plt.legend()

plt.subplot(2,2,3)
plt.hist(df_sm['pola3_pv_angle'], alpha = 0.5, bins = 50, label = 'sm pola3 pv-angle')
plt.hist(df_ps['pola3_pv_angle'], alpha = 0.5, bins = 50, label = 'ps pola3 pv-angle')
plt.xlabel('pv_angle (rad)')
plt.ylim(0, 2000)
plt.grid()
plt.legend()

plt.subplot(2,2,4)
plt.hist(df_sm['pola4_pv_angle'], alpha = 0.5, bins = 50, label = 'sm pola4 pv-angle')
plt.hist(df_ps['pola4_pv_angle'], alpha = 0.5, bins = 50, label = 'ps pola4 pv-angle')
plt.xlabel('pv_angle (rad)')
plt.ylim(0, 2000)
plt.grid()
plt.legend()


plt.show()


bf.plot_2d(df4["pola3_pv_angle"], df4["pola2_pv_angle"], 'pv-angle root', 'pola2_pv-angle', (0,2*np.pi), (0,2*np.pi))
plt.show()



#convert for same range as pv_angle
#df4["gen_phitt"] = 2*df4["gen_phitt"]*(2*np.pi/360) + np.pi
#df4["pseudo_phitt"] = 2*df4["pseudo_phitt"]*(2*np.pi/360) + np.pi
#df4["reco_phitt"] = 2*df4["reco_phitt"]*(2*np.pi/360) + np.pi
#df4["reco2_phitt"] = 2*df4["reco2_phitt"]*(2*np.pi/360) + np.pi
#df4["pola_phitt"] = 2*df4["pola_phitt"]*(2*np.pi/360) + np.pi
#df4["pola2_phitt"] = 2*df4["pola2_phitt"]*(2*np.pi/360) + np.pi





gen_nu_1 = Momentum4.m_eta_phi_p(df4['gen_nu_p_1']*0, df4['gen_nu_eta_1'], df4['gen_nu_phi_1'], df4['gen_nu_p_1'])

gen_nu_2 = Momentum4.m_eta_phi_p(df4['gen_nu_p_2']*0, df4['gen_nu_eta_2'], df4['gen_nu_phi_2'], df4['gen_nu_p_2'])

reco_nu_1 = Momentum4.m_eta_phi_p(df4['reco_nu_p_1']*0, df4['reco_nu_eta_1'], df4['reco_nu_phi_1'], df4['reco_nu_p_1'])

reco_nu_2 = Momentum4.m_eta_phi_p(df4['reco_nu_p_2']*0, df4['reco_nu_eta_2'], df4['reco_nu_phi_2'], df4['reco_nu_p_2'])

pola_nu_1 = Momentum4.m_eta_phi_p(df4['pola_nu_p_1']*0, df4['pola_nu_eta_1'], df4['pola_nu_phi_1'], df4['pola_nu_p_1'])

pola_nu_2 = Momentum4.m_eta_phi_p(df4['pola_nu_p_2']*0, df4['pola_nu_eta_2'], df4['pola_nu_phi_2'], df4['pola_nu_p_2'])


def plot_1D(variable, number, level):
    
    mean_std = []
    exec('mean_std.append(np.array(gen_nu_%s.%s - %s_nu_%s.%s).mean())'%(number, variable, level, number, variable))
    exec('mean_std.append(np.array(gen_nu_%s.%s - %s_nu_%s.%s).std())'%(number, variable, level, number, variable))
    exec('plt.hist(gen_nu_%s.%s - %s_nu_%s.%s, bins = 100, alpha = 0.3, label = "gen-%s nu%s_%s, mean=%.2f, std=%.2f")'%(number, variable, level, number, variable, level, number, variable, mean_std[0], mean_std[1]))
    
plot_1D('p_x', '1', 'pola')
plot_1D('p_y', '1', 'pola')
plot_1D('p_z', '1', 'pola')
plot_1D('p_x', '2', 'pola')
plot_1D('p_y', '2', 'pola')
plot_1D('p_z', '2', 'pola')
plt.legend(prop = {'size': 12})
plt.title('Quality of Neutrino reconstruction with the polarimetric vector approximation', fontsize = 'xx-large')
plt.xlabel('Difference in Momenta to the true neutrino momentum (GeV)', fontsize = 'x-large')
plt.grid()
plt.show()




############ phitt  #####################################
#plt.title('PS and SM distributions of angles')
#plt.subplot(2,2,1)
#plt.hist(df_sm['reco2_phitt'], alpha = 0.5, bins = 50, label = 'sm reco2 phitt')
#plt.hist(df_ps['reco2_phitt'], alpha = 0.5, bins = 50, label = 'ps reco2 phitt')
#plt.xlabel('reco2_phitt')
#plt.ylim(0, 1500)
#plt.grid()
#plt.legend()

#plt.subplot(2,2,2)
#plt.hist(df_sm['reco_phitt'], alpha = 0.5, bins = 50, label = 'sm reco phitt')
#plt.hist(df_ps['reco_phitt'], alpha = 0.5, bins = 50, label = 'ps reco phitt')
#plt.xlabel('reco_phitt')
#plt.ylim(0, 1500)
#plt.grid()
#plt.legend()

#plt.subplot(2,2,3)
#plt.hist(df_sm['pola2_phitt'], alpha = 0.5, bins = 50, label = 'sm pola2 phitt')
#plt.hist(df_ps['pola2_phitt'], alpha = 0.5, bins = 50, label = 'ps pola2 phitt')
#plt.xlabel('pola2_phitt')
#plt.ylim(0, 1500)
#plt.grid()
#plt.legend()
#plt.subplot(2,2,4)
#plt.hist(df_sm['pola_phitt'], alpha = 0.5, bins = 50, label = 'sm pola phitt')
#plt.hist(df_ps['pola_phitt'], alpha = 0.5, bins = 50, label = 'ps pola phitt')
#plt.xlabel('pola_phitt')
#plt.ylim(0, 1500)
#plt.grid()
#plt.legend()
#plt.show()

############## phitt  #####################################
#plt.title('PS and SM distributions of angles')
#plt.subplot(2,2,1)
#plt.hist(df_sm['gen_phitt'], alpha = 0.5, bins = 50, label = 'sm gen phitt')
#plt.hist(df_ps['gen_phitt'], alpha = 0.5, bins = 50, label = 'ps gen phitt')
#plt.xlabel('gen_phitt')
#plt.ylim(0, 1500)
#plt.grid()
#plt.legend()

#plt.subplot(2,2,2)
#plt.hist(df_sm['pseudo_phitt'], alpha = 0.5, bins = 50, label = 'sm pseudo phitt')
#plt.hist(df_ps['pseudo_phitt'], alpha = 0.5, bins = 50, label = 'ps pseud phitt')
#plt.xlabel('pseudo_phitt')
#plt.ylim(0, 1500)
#plt.grid()
#plt.legend()

#plt.subplot(2,2,3)
#plt.hist(df_sm['reco_phitt'], alpha = 0.5, bins = 50, label = 'sm reco phitt')
#plt.hist(df_ps['reco_phitt'], alpha = 0.5, bins = 50, label = 'ps reco phitt')
#plt.xlabel('reco_phitt')
#plt.ylim(0, 1500)
#plt.grid()
#plt.legend()
#plt.subplot(2,2,4)
#plt.hist(df_sm['pola_phitt'], alpha = 0.5, bins = 50, label = 'sm pola phitt')
#plt.hist(df_ps['pola_phitt'], alpha = 0.5, bins = 50, label = 'ps pola phitt')
#plt.xlabel('pola_phitt')
#plt.ylim(0, 1500)
#plt.grid()
#plt.legend()
#plt.show()




############## now check (thanks Hugo) the histogram differences ##########


bins_array = np.linspace(0, 2*np.pi, 21)
zeros = np.linspace(0, 0, 21)
#hist_sm_gen_phitt = np.histogram(df_sm['gen_phitt'], bins = bins_array)
#hist_ps_gen_phitt = np.histogram(df_ps['gen_phitt'], bins = bins_array)

#hist_sm_pseudo_phitt = np.histogram(df_sm['pseudo_phitt'], bins = bins_array)
#hist_ps_pseudo_phitt = np.histogram(df_ps['pseudo_phitt'], bins = bins_array)

#hist_sm_pola_phitt = np.histogram(df_sm['pola_phitt'], bins = bins_array)
#hist_ps_pola_phitt = np.histogram(df_ps['pola_phitt'], bins = bins_array)

#hist_sm_pola2_phitt = np.histogram(df_sm['pola2_phitt'], bins = bins_array)
#hist_ps_pola2_phitt= np.histogram(df_ps['pola2_phitt'], bins = bins_array)

#hist_sm_reco_phitt = np.histogram(df_sm['reco_phitt'], bins = bins_array)
#hist_ps_reco_phitt = np.histogram(df_ps['reco_phitt'], bins = bins_array)

#hist_sm_reco2_phitt = np.histogram(df_sm['reco2_phitt'], bins = bins_array)
#hist_ps_reco2_phitt = np.histogram(df_ps['reco2_phitt'], bins = bins_array)


hist_sm_gen = np.histogram(df_sm['pv_angle'], bins = bins_array)
hist_ps_gen = np.histogram(df_ps['pv_angle'], bins = bins_array)

hist_sm_pseudo = np.histogram(df_sm['pseudo_pv_angle'], bins = bins_array)
hist_ps_pseudo = np.histogram(df_ps['pseudo_pv_angle'], bins = bins_array)

hist_sm_pola = np.histogram(df_sm['pola_pv_angle'], bins = bins_array)
hist_ps_pola = np.histogram(df_ps['pola_pv_angle'], bins = bins_array)

hist_sm_pola2 = np.histogram(df_sm['pola2_pv_angle'], bins = bins_array)
hist_ps_pola2 = np.histogram(df_ps['pola2_pv_angle'], bins = bins_array)

hist_sm_pola3 = np.histogram(df_sm['pola3_pv_angle'], bins = bins_array)
hist_ps_pola3 = np.histogram(df_ps['pola3_pv_angle'], bins = bins_array)

hist_sm_pola4 = np.histogram(df_sm['pola4_pv_angle'], bins = bins_array)
hist_ps_pola4 = np.histogram(df_ps['pola4_pv_angle'], bins = bins_array)


hist_sm_reco = np.histogram(df_sm['reco_pv_angle'], bins = bins_array)
hist_ps_reco = np.histogram(df_ps['reco_pv_angle'], bins = bins_array)

hist_sm_reco2 = np.histogram(df_sm['reco2_pv_angle'], bins = bins_array)
hist_ps_reco2 = np.histogram(df_ps['reco2_pv_angle'], bins = bins_array)


hist_sm_aco = np.histogram(df_sm['aco_angle_1'], bins = bins_array)
hist_ps_aco = np.histogram(df_ps['aco_angle_1'], bins = bins_array)





########### and now comapre the two !

plt.title('sm/ps separation in phitt and pv angle\n%i bins'%(len(bins_array)-1), fontsize = 'xx-large')
#plt.plot(bins_array[:-1], hist_sm_pseudo_phitt[0]/hist_ps_pseudo_phitt[0], 'r-', label = 'pseudo phitt')
plt.plot(bins_array[:-1], hist_sm_pseudo[0]/hist_ps_pseudo[0], 'r-', label = 'pseudo pv_angle')
plt.plot(bins_array[:-1], (hist_sm_aco[0]/hist_ps_aco[0]),  '-', color = 'darkorange', label = 'gen aco_angle1')
#plt.plot(bins_array[:-1], hist_sm_gen[0]/hist_ps_gen[0], '-', label = 'gen')

#plt.plot(bins_array[:-1], hist_sm_pola_phitt[0]/hist_ps_pola_phitt[0], 'g-', label = 'pola phitt')
plt.plot(bins_array[:-1], hist_sm_gen[0]/hist_ps_gen[0], 'r-.', label = 'root pv_angle')
plt.plot(bins_array[:-1], hist_sm_pola[0]/hist_ps_pola[0], 'g-', label = 'pola pv_angle')
plt.plot(bins_array[:-1], hist_sm_pola4[0]/hist_ps_pola4[0], 'g-.', label = 'pola4 pv_angle')

#plt.plot(bins_array[:-1], hist_sm_reco_phitt[0]/hist_ps_reco_phitt[0], 'b-', label = 'reco phitt')
plt.plot(bins_array[:-1], hist_sm_reco[0]/hist_ps_reco[0], 'b--', label = 'reco pv_angle')
plt.plot(bins_array[:-1], 1+zeros[:-1], 'k-', label = 'No separation')


plt.grid()
plt.xlabel('angle (rad)', fontsize = 'x-large')
plt.legend()
plt.ylabel('sm/ps distributions', fontsize = 'xx-large')
plt.show()




plt.title('Pseudo-scalar-SM separation for CP sensitive angles\ncalculated with different neutrinos estimations', fontsize = 'xx-large')
plt.plot(bins_array[:-1], hist_sm_aco[0] - hist_ps_aco[0], '-', color = 'limegreen', label = 'Acoplanary angle(perfect neutrinos)')

plt.plot(bins_array[:-1], hist_sm_pseudo[0] - hist_ps_pseudo[0], '-', color = 'cornflowerblue', label = 'PV angle(perfect neutrinos)')
plt.plot(bins_array[:-1], hist_sm_pola[0] - hist_ps_pola[0], '--', color = 'cornflowerblue', label = 'PV angle(polarimetric neutrinos)')
plt.plot(bins_array[:-1], hist_sm_reco[0] - hist_ps_reco[0], '-.', color = 'cornflowerblue', label = 'PV angle (NN regressed neutrinos)')

plt.plot(bins_array[:-1], hist_sm_pseudo_phitt[0] - hist_ps_pseudo_phitt[0], '-', color = 'darkorange', label = 'phitt(perfect neutrinos)')
#plt.plot(bins_array[:-1], hist_sm_gen[0] - hist_ps_gen[0], '-', label = 'gen')
plt.plot(bins_array[:-1], hist_sm_pola_phitt[0] - hist_ps_pola_phitt[0], '--', color = 'darkorange', label = 'phitt(polarimetric neutrinos)')
plt.plot(bins_array[:-1], hist_sm_reco_phitt[0] - hist_ps_reco_phitt[0], '-.', color = 'darkorange', label = 'phitt(NN regressed neutrinos)')

plt.plot(bins_array[:-1], zeros[:-1], 'k-', label = 'No separation')
plt.grid()
plt.xlabel('Angle (rad)', fontsize = 'x-large')
plt.legend()
plt.ylabel('Difference between sm and ps distributions', fontsize = 'xx-large')
plt.show()





raise END

















#bf.plot_2d(df4["pv_angle"], df4["pseudo_pv_angle"], 'gen_pv_angle', 'pseudo_pv_angle', (-7, 7), (-7,7), (100,100))

plt.title('PS and SM distributions of angles')
plt.subplot(2,2,1)
plt.hist(df_sm['reco2_pv_angle'], alpha = 0.5, bins = 50, label = 'sm reco2 pv_angle')
plt.hist(df_ps['reco2_pv_angle'], alpha = 0.5, bins = 50, label = 'ps reco2 pv_angle')
plt.xlabel('reco2_pv_angle')
plt.ylim(0, 1500)
plt.grid()
plt.legend()

plt.subplot(2,2,2)
plt.hist(df_sm['reco_pv_angle'], alpha = 0.5, bins = 50, label = 'sm reco pv_angle')
plt.hist(df_ps['reco_pv_angle'], alpha = 0.5, bins = 50, label = 'ps reco pv_angle')
plt.xlabel('reco_pv_angle')
plt.ylim(0, 1500)
plt.grid()
plt.legend()

plt.subplot(2,2,3)
plt.hist(df_sm['pola2_pv_angle'], alpha = 0.5, bins = 50, label = 'sm pola2 pv_angle')
plt.hist(df_ps['pola2_pv_angle'], alpha = 0.5, bins = 50, label = 'ps pola2 pv_angle')
plt.xlabel('pola2_pv_angle')
plt.ylim(0, 1500)
plt.grid()
plt.legend()
plt.subplot(2,2,4)
plt.hist(df_sm['pola_pv_angle'], alpha = 0.5, bins = 50, label = 'sm pola pv_angle')
plt.hist(df_ps['pola_pv_angle'], alpha = 0.5, bins = 50, label = 'ps pola pv_angle')
plt.xlabel('pola_pv_angle')
plt.ylim(0, 1500)
plt.grid()
plt.legend()
plt.show()


############## now check (thanks Hugo) the histogram differences ##########

plt.title('sm-ps separation in pv_angle\n%i bins'%(len(bins_array)-1), fontsize = 'xx-large')
plt.plot(bins_array[:-1], hist_sm_pseudo[0] - hist_ps_pseudo[0], '-', label = 'pseudo')
plt.plot(bins_array[:-1], hist_sm_gen[0] - hist_ps_gen[0], '-', label = 'gen')

plt.plot(bins_array[:-1], hist_sm_pola[0] - hist_ps_pola[0], '-', label = 'pola')
plt.plot(bins_array[:-1], hist_sm_pola2[0] - hist_ps_pola2[0], '-', label = 'pola2')

plt.plot(bins_array[:-1], hist_sm_reco[0] - hist_ps_reco[0], '-', label = 'reco')
plt.plot(bins_array[:-1], hist_sm_reco2[0] - hist_ps_reco2[0], '-', label = 'reco2')


plt.plot(bins_array[:-1], zeros[:-1], 'k-', label = 'No separation')


plt.grid()
plt.xlabel('pv_angle', fontsize = 'x-large')
plt.legend()
plt.ylabel('sm-ps distributions', fontsize = 'xx-large')

plt.show()

plt.title('sm/ps separation in pv_angle\n%i bins'%(len(bins_array)-1), fontsize = 'xx-large')
plt.plot(bins_array[:-1], hist_sm_pseudo[0]/hist_ps_pseudo[0], '-', label = 'pseudo')
plt.plot(bins_array[:-1], hist_sm_gen[0]/hist_ps_gen[0], '-', label = 'gen')

plt.plot(bins_array[:-1], hist_sm_pola[0]/hist_ps_pola[0], '-', label = 'pola')
plt.plot(bins_array[:-1], hist_sm_pola2[0]/hist_ps_pola2[0], '-', label = 'pola2')

plt.plot(bins_array[:-1], hist_sm_reco[0]/hist_ps_reco[0], '-', label = 'reco')
plt.plot(bins_array[:-1], hist_sm_reco2[0]/hist_ps_reco2[0], '-', label = 'reco2')


plt.plot(bins_array[:-1], 1+zeros[:-1], 'k-', label = 'No separation')


plt.grid()
plt.xlabel('pv_angle', fontsize = 'x-large')
plt.legend()
plt.ylabel('sm/ps distributions', fontsize = 'xx-large')

plt.show()



plt.title('sm-ps separation in phitt\n%i bins'%(len(bins_array)-1), fontsize = 'xx-large')
plt.plot(bins_array[:-1], hist_sm_pseudo_phitt[0] - hist_ps_pseudo_phitt[0], '-', label = 'pseudo')
#plt.plot(bins_array[:-1], hist_sm_gen[0] - hist_ps_gen[0], '-', label = 'gen')

plt.plot(bins_array[:-1], hist_sm_pola_phitt[0] - hist_ps_pola_phitt[0], '-', label = 'pola')
plt.plot(bins_array[:-1], hist_sm_pola2_phitt[0] - hist_ps_pola2_phitt[0], '-', label = 'pola2')

plt.plot(bins_array[:-1], hist_sm_reco_phitt[0] - hist_ps_reco_phitt[0], '-', label = 'reco')
plt.plot(bins_array[:-1], hist_sm_reco2_phitt[0] - hist_ps_reco2_phitt[0], '-', label = 'reco2')


plt.plot(bins_array[:-1], zeros[:-1], 'k-', label = 'No separation')


plt.grid()
plt.xlabel('phitt', fontsize = 'x-large')
plt.legend()
plt.ylabel('sm-ps distributions', fontsize = 'xx-large')

plt.show()

plt.title('sm/ps separation in phitt\n%i bins'%(len(bins_array)-1), fontsize = 'xx-large')
plt.plot(bins_array[:-1], hist_sm_pseudo_phitt[0]/hist_ps_pseudo_phitt[0], '-', label = 'pseudo')
#plt.plot(bins_array[:-1], hist_sm_gen[0]/hist_ps_gen[0], '-', label = 'gen')

plt.plot(bins_array[:-1], hist_sm_pola_phitt[0]/hist_ps_pola_phitt[0], '-', label = 'pola')
plt.plot(bins_array[:-1], hist_sm_pola2_phitt[0]/hist_ps_pola2_phitt[0], '-', label = 'pola2')

plt.plot(bins_array[:-1], hist_sm_reco_phitt[0]/hist_ps_reco_phitt[0], '-', label = 'reco')
plt.plot(bins_array[:-1], hist_sm_reco2_phitt[0]/hist_ps_reco2_phitt[0], '-', label = 'reco2')


plt.plot(bins_array[:-1], 1+zeros[:-1], 'k-', label = 'No separation')


plt.grid()
plt.xlabel('phitt', fontsize = 'x-large')
plt.legend()
plt.ylabel('sm/ps distributions', fontsize = 'xx-large')

plt.show()

#plt.hist(df_sm['pseudo_phitt'], alpha = 0.5, bins = 50, label = 'sm pseudo pv_angle')
#plt.hist(df_ps['pseudo_pv_angle'], alpha = 0.5, bins = 50, label = 'ps pseudo pv_angle')
#plt.xlabel('pseudo_pv_angle')
