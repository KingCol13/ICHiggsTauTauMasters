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
import basic_functions as bf
mpl.use('tkagg')
import matplotlib.pyplot as plt
from lbn_modified3 import LBN, LBNLayer
import tensorflow as tf
import configuration_module as conf


tau_mode1 = 1
tau_mode2 = 1
decay_mode1 = 1
decay_mode2 = 1 

if len(sys.argv) == 5:
    tau_mode1 = int(sys.argv[1])
    tau_mode2 = int(sys.argv[2])
    decay_mode1 = int(sys.argv[3])
    decay_mode2 = int(sys.argv[4])


print('\nWe are regressing the %i, %i channel' %(decay_mode1, decay_mode2))


#for some reason pylorentz is installed somewhere differently ?
#sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
from pylorentz import Momentum4
from pylorentz import Vector4
from pylorentz import Position4
import polarimetric_module as polari
import alpha_module as am

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

target = [ "met", "metx", "mety", #"aco_angle_1", "aco_angle_6", "aco_angle_5", "aco_angle_7"
         ]  #acoplanarity angle
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2","rand","wt_cp_ps","wt_cp_sm",
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

covs = sv_covariance_matrices + ip_covariance_matrices + met_covariance_matrices

variables4=(momenta_features+other_features+target+selectors +additional_info + covs) #copying Kinglsey's way cause it is very clean
print('Check 1')
df4 = tree.pandas.df(variables4)

df4 = df4[
      (df4["tau_decay_mode_1"] == tau_mode1) 
    & (df4["tau_decay_mode_2"] == tau_mode2) 
    & (df4["mva_dm_1"] == decay_mode1) 
    & (df4["mva_dm_2"] == decay_mode2)
    & (df4["gen_nu_p_1"] > -4000)
    & (df4["gen_nu_p_2"] > -4000)
    #& (df4["sv_x_1"] != 0)
    #& (df4["sv_x_2"] != 0)
    
]


#remove the nans
df4 = df4.dropna()

print(0.7*len(df4),'This is the length') #up to here we are fine

#df_ps = df4[
      #(df4["rand"]<df4["wt_cp_ps"]/2)     #a data frame only including the pseudoscalars
#]

#df_sm = df4[
      #(df4["rand"]<df4["wt_cp_sm"]/2)     #data frame only including the scalars
#]

print("panda Data frame created \n")

df4.head()

def cross_product(vector3_1,vector3_2):
    if len(vector3_1)!=3 or len(vector3_1)!=3:
        print('These are not 3D arrays !')
    x_perp_vector=vector3_1[1]*vector3_2[2]-vector3_1[2]*vector3_2[1]
    y_perp_vector=vector3_1[2]*vector3_2[0]-vector3_1[0]*vector3_2[2]
    z_perp_vector=vector3_1[0]*vector3_2[1]-vector3_1[1]*vector3_2[0]
    return np.array([x_perp_vector,y_perp_vector,z_perp_vector])

def dot_product(vector1,vector2):
    if len(vector1)!=len(vector2):
        raise Arrays_of_different_size
    prod=0
    for i in range(len(vector1)):
        prod=prod+vector1[i]*vector2[i]
    return prod


def norm(vector):
    if len(vector)!=3:
        print('This is only for a 3d vector')
    return np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)

def remove9999 (Momenta4, leading):
    if leading == 1:
        nu_ref = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_1"])), df4["gen_nu_eta_1"], df4["gen_nu_phi_1"], df4["gen_nu_p_1"])
    if leading == 2:
        nu_ref = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_2"])), df4["gen_nu_eta_2"], df4["gen_nu_phi_2"], df4["gen_nu_p_2"])
    
    array = np.array(Momenta4).T
    array = array[nu_ref.p_z != 9999]
    array = array.T
    return Momentum4(array[0], array[1], array[2], array[3])


def checks(x_array, length, model):
    regressed_array = model({"lab_frame": x_array})

    #length = 0#numTrain

    E_1 = norm([regressed_array[:, i_nu1_px],regressed_array[:, i_nu1_py], regressed_array[:, i_nu1_pz]])
    E_2 = norm([regressed_array[:, i_nu2_px],regressed_array[:, i_nu2_py], regressed_array[:, i_nu2_pz]])


    nu_1_regressed = Momentum4(E_1,regressed_array[:, i_nu1_px],regressed_array[:, i_nu1_py], regressed_array[:, i_nu1_pz])
    nu_2_regressed = Momentum4(E_2,regressed_array[:, i_nu2_px],regressed_array[:, i_nu2_py], regressed_array[:, i_nu2_pz])


    print('All means')

    hist1 = nu_1_regressed.p_x
    hist2 = bf.Mom4_to_tf(nu_1.p_x)[length:]
    hist_a = np.array(hist2-hist1) #
    hist1 = nu_2_regressed.p_x
    hist2 = bf.Mom4_to_tf(nu_2.p_x)[length:]
    hist_b = np.array(hist2-hist1) #
    hist1 = nu_1_regressed.p_y
    hist2 = bf.Mom4_to_tf(nu_1.p_y)[length:]
    hist_c = np.array(hist2-hist1) #
    hist1 = nu_2_regressed.p_y
    hist2 = bf.Mom4_to_tf(nu_2.p_y)[length:]
    hist_l = np.array(hist2-hist1) #
    hist1 = nu_1_regressed.p_z
    hist2 = bf.Mom4_to_tf(nu_1.p_z)[length:]
    hist_d = np.array(hist2-hist1) #
    hist1 = nu_2_regressed.p_z
    hist2 = bf.Mom4_to_tf(nu_2.p_z)[length:]
    hist_e = np.array(hist2-hist1) #
    hist1 = nu_1_regressed.phi
    hist2 = bf.Mom4_to_tf(nu_1.phi)[length:]
    hist_f = np.array(hist2-hist1) #
    hist1 = nu_2_regressed.phi
    hist2 = bf.Mom4_to_tf(nu_2.phi)[length:]
    hist_g = np.array(hist2-hist1) #
    hist1 = nu_1_regressed.eta
    hist2 = bf.Mom4_to_tf(nu_1.eta)[length:]
    hist_h = np.array(hist2-hist1) #
    hist1 = nu_2_regressed.eta
    hist2 = bf.Mom4_to_tf(nu_2.eta)[length:]
    hist_i = np.array(hist2-hist1) #
    hist1 = nu_1_regressed.p
    hist2 = bf.Mom4_to_tf(nu_1.p)[length:]
    hist_j = np.array(hist2-hist1) #
    hist1 = nu_2_regressed.p
    hist2 = bf.Mom4_to_tf(nu_2.p)[length:]
    hist_k = np.array(hist2-hist1) #

    return [hist_a.mean(), hist_b.mean(), hist_c.mean(), hist_d.mean(), hist_l.mean(), hist_e.mean(), 
            hist_f.mean(), hist_g.mean(), hist_h.mean(), hist_i.mean(), hist_j.mean(), hist_k.mean()], [hist_a.std(), hist_b.std(), hist_c.std(), hist_d.std(), hist_l.std(), hist_e.std(), 
            hist_f.std(), hist_g.std(), hist_h.mean(), hist_i.std(), hist_j.std(), hist_k.std()]


def one_d(val):
    return tf.constant(val, shape = df4["pi_px_1"].shape, dtype = np.float32)

###############################################################################################################################
#Need to remonve the nans

def remove_nan (variable):
    return tf.where(tf.math.is_nan(variable), 0*df4[ "pi_E_1"], variable)

#for i in list(df4):
    #if i != 'tau_decay_mode_1' and i != 'tau_decay_mode_2' and i != 'mva_dm_1' and i != 'mva_dm_2' and i!= 'rand' and i[:6] != 'metcov':
        #print(i[:6])
        #exec('df4["%s"] = remove_nan(df4["%s"])'%(i, i))



###############################################################################################################################

# the loss functions
#
#need them there, redefine later
ratio_all = 1
ratio_p = 0
ratio_phi = 0
ratio_tau = 0
ratio_H = 0


def one_d_traning(val, shape_array):
    return tf.constant(val, shape = shape_array.shape, dtype = np.float32)

def loss_D_p (y_true, y_pred):
    global ratio_all
    #calculting the difference between the the components, need to add the smearing of detector eventually
    target_components = [i_nu1_px, i_nu1_py, i_nu1_pz, i_nu2_px, i_nu2_py, i_nu2_pz]
    target_components_diff_list = []
    for i in target_components: target_components_diff_list.append((y_true[:,i]-y_pred[:,i])**2)
    dxyz = 0
    for d in target_components_diff_list: dxyz+=d
    return ratio_all * dxyz #tone it down cause it takes too much space

def energy_nu (y_true, y_pred, number):
    if number == 1:
        return tf.sqrt(y_pred[:, i_nu1_px]**2 + y_pred[:, i_nu1_py]**2 + y_pred[:, i_nu1_pz]**2)
    if number == 2:
        return tf.sqrt(y_pred[:, i_nu2_px]**2 + y_pred[:, i_nu2_py]**2 + y_pred[:, i_nu2_pz]**2)

def loss_mass_tau(y_true, y_pred):
    #now we try only to use the y_pred for neutrino info, this is I guess their way of 
    #only training for neutrino info whilst keeping nice structure
    #we are always assuming m=0
    # note, we are taking y_tau as exact, we could choose not to...
    global ratio_tau
    E1 = y_true[:, i_tau1_e] + energy_nu(y_true, y_pred, 1) 
    P1_squared = (y_true[:, i_tau1_px] + y_pred[:, i_nu1_px])**2 + (y_true[:, i_tau1_py] + y_pred[:, i_nu1_py])**2 + (y_true[:, i_tau1_pz] + y_pred[:, i_nu1_pz])**2
    R1 = (E1**2 - P1_squared -  m_tau_squared)/m_Higgs_squared
    
    E2 = y_true[:, i_tau2_e] + energy_nu(y_true, y_pred, 2) 
    P2_squared = (y_true[:, i_tau2_px] + y_pred[:, i_nu2_px])**2 + (y_true[:, i_tau2_py] + y_pred[:, i_nu2_py])**2 + (y_true[:, i_tau2_pz] + y_pred[:, i_nu2_pz])**2
    R2 = (E2**2 - P2_squared - m_tau_squared)/m_Higgs_squared
    return ratio_tau * (tf.math.abs(R1) + tf.abs(R2))


def loss_mass_Higgs(y_true, y_pred):
    global ratio_H
    print(tf.convert_to_tensor(y_pred[:, 0]).shape, 'this is shape')
    EH = y_true[:, i_tau1_e] + energy_nu(y_true, y_pred, 1) + y_true[:, i_tau2_e] + energy_nu(y_true, y_pred, 2)
    px_H = y_true[:, i_tau1_px] + y_true[:, i_tau2_px] + y_pred[:, i_nu1_px] + y_pred[:, i_nu2_px]
    py_H = y_true[:, i_tau1_py] + y_true[:, i_tau2_py] + y_pred[:, i_nu1_py] + y_pred[:, i_nu2_py]
    pz_H = y_true[:, i_tau1_pz] + y_true[:, i_tau2_pz] + y_pred[:, i_nu1_pz] + y_pred[:, i_nu2_pz]
#      y_true[:, i_gen_mass]**2
    return ratio_H* tf.abs((EH**2-px_H**2-py_H**2-pz_H**2 -m_Higgs_squared)/m_Higgs_squared)


def loss_phi(y_true, y_pred):
    global ratio_phi
    phi_diff_1 = (tf.math.atan2(y_pred[:, i_nu1_py],y_pred[:, i_nu1_px])-tf.math.atan2(y_true[:, i_nu1_py],y_true[:, i_nu1_px]))**2
    phi_diff_2 = (tf.math.atan2(y_pred[:, i_nu2_py],y_pred[:, i_nu2_px])-tf.math.atan2(y_true[:, i_nu2_py],y_true[:, i_nu2_px]))**2

    
    return ratio_phi*tf.convert_to_tensor(phi_diff_1+phi_diff_2) #tone it up


def loss_p(y_true, y_pred):
    global ratio_p
    delta_p1 = (energy_nu(y_true,y_pred, 1) - tf.sqrt(y_true[:, i_nu1_px]**2 + y_true[:, i_nu1_py]**2 + y_true[:, i_nu1_pz]**2))**2
    delta_p2 = (energy_nu(y_true,y_pred, 2) - tf.sqrt(y_true[:, i_nu2_px]**2 + y_true[:, i_nu2_py]**2 + y_true[:, i_nu2_pz]**2))**2
    
    return ratio_p*tf.convert_to_tensor(delta_p1+delta_p2) #tone it up
    



def loss_fn(y_true, y_pred):

    dxyz = loss_D_p(y_true, y_pred)
    #dmet = loss_dmet(y_true, y_pred)
    #dPTtaus = loss_dPTtaus(y_true, y_pred)
    dphi = loss_phi(y_true,y_pred)
    dtau = loss_mass_tau(y_true, y_pred)
    dHiggs = loss_mass_Higgs(y_true, y_pred)
    dp = loss_p(y_true, y_pred)
    #dM = loss_dM_had(y_true, y_pred)

    return dxyz + dtau + dHiggs + dphi + dp


###############################################################################################################################
#define the usefull 4 momenta
nu_1 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_1"])), df4["gen_nu_eta_1"], df4["gen_nu_phi_1"], df4["gen_nu_p_1"])
nu_2 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_2"])), df4["gen_nu_eta_2"], df4["gen_nu_phi_2"], df4["gen_nu_p_2"])

configuration = conf.Configurations(df4, decay_mode1, decay_mode2)

tau_1_vis = configuration.tau_1_vis
tau_2_vis = configuration.tau_2_vis
###############################################################################################################################

# Now need to define the traning etc...

ref = [#smear_px,py                      #0
       #one_d(1.776),                      #1
       #df4["metx"],                   #2
       #df4["mety"],                   #3
       bf.Mom4_to_tf(tau_1_vis.e),       #4
       bf.Mom4_to_tf(tau_1_vis.p_x),     #5
       bf.Mom4_to_tf(tau_1_vis.p_y),     #6
       bf.Mom4_to_tf(tau_1_vis.p_z),     #7
       bf.Mom4_to_tf(tau_2_vis.e),       #8
       bf.Mom4_to_tf(tau_2_vis.p_x),     #9 
       bf.Mom4_to_tf(tau_2_vis.p_y),     #10
       bf.Mom4_to_tf(tau_2_vis.p_z),     #11
       #one_d(125),                    #12
       #bf.Mom4_to_tf(nu_1.e),            #13       corresponding to          #0
       bf.Mom4_to_tf(nu_1.p_x),          #14                                 #1
       bf.Mom4_to_tf(nu_1.p_y),          #15                                 #2
       bf.Mom4_to_tf(nu_1.p_z),          #16                                 #3
       #bf.Mom4_to_tf(nu_2.e),            #17                                 #4
       bf.Mom4_to_tf(nu_2.p_x),          #18                                 #5
       bf.Mom4_to_tf(nu_2.p_y),          #19                                 #6
       bf.Mom4_to_tf(nu_2.p_z),          #20                                 #7
]


ref = tf.transpose(ref)

#i_smear_px = 0
#i_tau_mass = 0
#i_smeared_met_px = 1
#i_smeared_met_py = 2
i_tau1_e = 0
i_tau1_px = 1
i_tau1_py = 2
i_tau1_pz = 3
i_tau2_e = 4
i_tau2_px = 5
i_tau2_py = 6
i_tau2_pz = 7
#i_gen_mass = 9
#i_nu1_e = 12
i_nu1_px = 8
i_nu1_py = 9
i_nu1_pz = 10
#i_nu2_e = 16
i_nu2_px = 11
i_nu2_py = 12
i_nu2_pz = 13

####Very important, the batch size must be defined beforehand to get rid of two trainable params !


m_Higgs_squared = 125**2
m_tau_squared = 1.776**2

nu_1_training = np.array(nu_1)[0]




sv_cov = [df4["svcov00_1"], df4["svcov01_1"],df4["svcov02_1"],df4["svcov10_1"], 
          df4["svcov11_1"], df4["svcov12_1"], df4["svcov20_1"], df4["svcov21_1"], 
          df4["svcov22_1"], df4["svcov00_2"], df4["svcov01_2"], df4["svcov02_2"],
          df4["svcov10_2"], df4["svcov11_2"], df4["svcov12_2"], df4["svcov20_2"],
          df4["svcov21_2"], df4["svcov22_2"],]

ip_cov = [df4["ipcov00_1"], df4["ipcov01_1"],df4["ipcov02_1"],df4["ipcov10_1"], 
          df4["ipcov11_1"], df4["ipcov12_1"], df4["ipcov20_1"], df4["ipcov21_1"], 
          df4["ipcov22_1"],df4["ipcov00_2"],df4["ipcov01_2"], 
          df4["ipcov02_2"],df4["ipcov10_2"],df4["ipcov11_2"],df4["ipcov12_2"],
          df4["ipcov20_2"],df4["ipcov21_2"], df4["ipcov22_2"],]


met_cov = [df4["metcov00"], df4["metcov01"],df4["metcov10"], df4["metcov11"] ]

################################################## A ###############################################################


y = ref

def test_scenario(scenario_nb, list_inputs, trainFrac = 0.7, nodes1 = 300, nodes2 = 300, drop = 0.1, E_size = 25, B_size = 50, ratio_all = 1/10000, ratio_p = 10/1500, ratio_phi = 4, ratio_tau = 10, ratio_H = 4, NN_structure = 'z'):
    
    global decay_mode1, decay_mode2, table2
    
    x = list_inputs
    numTrain = int(trainFrac*x.shape[0])
    x_train = x[:numTrain]
    y_train = y[:numTrain]
    x_val = x[numTrain:]
    y_val = y[numTrain:]

#now define the NN structure, depends on the input

    if NN_structure == 'z':
        NN_structure_nb = 400
        nodes1 = 300 
        nodes2 = 300
        drop = 0.1
    
    if NN_structure == 'b':
        nodes1 = 100 
        nodes2 = 100
        drop = 0.1
        
    if NN_structure == 'd':
        NN_structure_nb = 200
        nodes1 = 300 
        nodes2 = 300
        drop = 0.2
        
    if NN_structure == 'e':
        nodes1 = 300 
        nodes2 = 300
        drop = 0
    
    if NN_structure == 'f':
        nodes1 = 64
        nodes2 = 64
        drop = 0
    
    if NN_structure == 'z' or NN_structure == 'b' or NN_structure == 'd' or NN_structure == 'e' or NN_structure == 'f':
        input_1 = tf.keras.Input(shape = x.shape, name="lab_frame")
        x2 = tf.keras.layers.Dense(nodes1, activation = 'relu', name="learning")(input_1)
        x3 = tf.keras.layers.Dense(nodes2, activation = 'relu', name="learning2")(x2)
        x4 = tf.keras.layers.Dropout(drop, name="dropout2")(x3)
        output = tf.keras.layers.Dense(14, name="output")(x4)

    if NN_structure == 'a':
        NN_structure_nb = 100
        nodes1 = 300
        nodes2 = 300
        nodes3 = 300
        drop = 0.1
        
    if NN_structure == 'c':
        nodes1 = 100
        nodes2 = 100
        nodes3 = 100
        drop = 0.1
        
    if NN_structure == 'g':
        nodes1 = 64
        nodes2 = 100
        nodes3 = 64
        drop = 0.1
    
    if NN_structure == 'a' or NN_structure == 'c' or NN_structure == 'g':
        input_1 = tf.keras.Input(shape = x.shape, name="lab_frame")
        x2 = tf.keras.layers.Dense(nodes1, activation = 'relu', name="learning")(input_1)
        x1 = tf.keras.layers.Dense(nodes2, activation = 'relu', name="learning3")(x2)
        x3 = tf.keras.layers.Dense(nodes3, activation = 'relu', name="learning2")(x1)
        x4 = tf.keras.layers.Dropout(drop, name="dropout2")(x3)
        output = tf.keras.layers.Dense(14, name="output")(x4)
        
    if NN_structure == 'h':
        NN_structure_nb = 300
        input_1 = tf.keras.Input(shape = x.shape, name="lab_frame")
        x2 = tf.keras.layers.Dense(300, activation = 'relu', name="learning")(input_1)
        x1 = tf.keras.layers.Dense(300, activation = 'relu', name="learning3")(x2)
        x3 = tf.keras.layers.Dense(300, activation = 'relu', name="learning2")(x1)
        x5 = tf.keras.layers.Dense(300, activation = 'relu', name="learning4")(x3)
        x4 = tf.keras.layers.Dropout(0.2, name="dropout2")(x5)
        output = tf.keras.layers.Dense(14, name="output")(x4)

    model = tf.keras.Model(
        inputs=[input_1],
        outputs=[output],
    )

    model.summary()
    model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae', loss_D_p, loss_phi, loss_p, loss_mass_tau, loss_mass_Higgs])#, loss_mass_Higgs, loss_mass_tau, loss_D_p])

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
        epochs=E_size,
        batch_size = B_size)

    mean_x_val, std_x_val = checks(x_val, numTrain, model)
    
    loss_val = sum(loss_fn(y_val, model({"lab_frame": x_val})))/len(y_val)
    average_val_p = (std_x_val[0] + std_x_val[1] + std_x_val [2] + std_x_val[3] + std_x_val[4] + std_x_val[5])/6
    print(std_x_val[0], loss_val)

        
    
        
    
    #table hosting only the results
    table2 = list(table2)
    table2.append(np.array([ratio_all, ratio_p, ratio_phi, ratio_tau, ratio_H, loss_val, scenario_nb, NN_structure_nb,  average_val_p]))
    table2 = np.array(table2)
    
    print(table2)

    np.savetxt('ResultsCoarse_dm_%i_%i_loss_%i_%i_%i_%i_%i.csv'%(decay_mode1, decay_mode2, ratio_all, ratio_p, ratio_phi, ratio_tau, ratio_H), table2, delimiter=',', fmt = '%.2f')
        
    return 1



trainFrac_in = 0.5
nodes1_in = 300
nodes2_in = 300
drop_in = 0.1
E_size_in = 25
B_size_in = 500#2**10


#ratio_all = 0
#ratio_p = 0
#ratio_phi = 1
#ratio_tau = 0
#ratio_H = 0
#table2 = []

#for i in range(1, 27):
    #for j in ['z', 'a', 'd', 'h']:
        #exec('done = test_scenario(%s, configuration.x%s, trainFrac_in, nodes1_in, nodes2_in, drop_in, E_size_in, B_size_in, ratio_all, ratio_p, ratio_phi, ratio_tau, ratio_H, "%s" )'%(i,i,j))
        
#ratio_all = 0
#ratio_p = 1
#ratio_phi = 0
#ratio_tau = 0
#ratio_H = 0
#table2 = []


#for i in range(1, 27):  
    #for j in ['z', 'a', 'd', 'h']:
        #exec('done = test_scenario(%s, configuration.x%s, trainFrac_in, nodes1_in, nodes2_in, drop_in, E_size_in, B_size_in, ratio_all, ratio_p, ratio_phi, ratio_tau, ratio_H, "%s" )'%(i,i,j))
        
ratio_all = 1
ratio_p = 1
ratio_phi = 1
ratio_tau = 1
ratio_H = 1
table2 = []


for i in range(1, 27):  
    for j in ['z', 'a', 'd', 'h']:
        exec('done = test_scenario(%s, configuration.x%s, trainFrac_in, nodes1_in, nodes2_in, drop_in, E_size_in, B_size_in, ratio_all, ratio_p, ratio_phi, ratio_tau, ratio_H, "%s" )'%(i,i,j))
        

ratio_all = 1/10000
ratio_p = 10/1500
ratio_phi = 4
ratio_tau = 10
ratio_H = 4
table2 = []

for i in range(1, 27):  
    for j in ['z', 'a', 'd', 'h']:
        exec('done = test_scenario(%s, configuration.x%s, trainFrac_in, nodes1_in, nodes2_in, drop_in, E_size_in, B_size_in, ratio_all, ratio_p, ratio_phi, ratio_tau, ratio_H, "%s" )'%(i,i,j))
        


ratio_all = 1
ratio_p = 0
ratio_phi = 0
ratio_tau = 0
ratio_H = 0
table2 = []

for i in range(1, 27):    
     for j in ['z', 'a', 'd', 'h']:
        exec('done = test_scenario(%s, configuration.x%s, trainFrac_in, nodes1_in, nodes2_in, drop_in, E_size_in, B_size_in, ratio_all, ratio_p, ratio_phi, ratio_tau, ratio_H, "%s" )'%(i,i,j))
        



















































