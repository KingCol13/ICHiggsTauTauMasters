# This is the python code that will regress and then save the neutrinos best regressed within the root file. 


# For now the best case scenario is 1,0,0,0,0 z 8


print('Hello World')
import sys
#sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
import uproot 
import numpy as np
import polarimetric_module as polari
import alpha_module as am
import basic_functions as bf
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from lbn_modified3 import LBN, LBNLayer
import tensorflow as tf

import configuration_module as conf

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
tree = uproot.open("/home/acraplet/Alie/Masters/MVAFILE_tt.root")["ntuple"]
#tree = uproot.open("/eos/user/d/dwinterb/SWAN_projects/Masters_CP/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
print("\n Tree loaded\n")


# define what variables are to be read into the dataframe
momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2", #subleading neutral pi 4-momentum
              "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1", #leading neutrino, gen level
              "gen_vis_p_1", "gen_vis_p_2",
              "gen_vis_E_1", "gen_vis_E_2",
              "gen_vis_phi_1", "gen_vis_phi_2",
              "gen_vis_eta_1", "gen_vis_eta_2",
              
              
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
                   "aco_angle_1", "pv_angle"
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


def one_d(val):
    return tf.constant(val, shape = df4["pi_px_1"].shape, dtype = np.float32)

def Mom4_to_tf(Mom4_1D):
    return tf.convert_to_tensor(Mom4_1D, dtype = 'float32')

def remove_nan (variable):
    return tf.where(tf.math.is_nan(variable), 0*df4[ "pi_E_1"], variable)


    
        
##############################################################################################################################
#neutrinos refs, in E, px, py, pz form


def get_a1(df4):
    #Charged and neutral pion momenta
    pi_1_4Mom = Momentum4(df4["pi_E_1"],df4["pi_px_1"],df4["pi_py_1"],df4["pi_pz_1"])
    pi2_1_4Mom = Momentum4(df4["pi2_E_1"],df4["pi2_px_1"],df4["pi2_py_1"],df4["pi2_pz_1"])
    pi3_1_4Mom = Momentum4(df4["pi3_E_1"],df4["pi3_px_1"],df4["pi3_py_1"],df4["pi3_pz_1"])

    #Same for the pi0
    pi_2_4Mom = Momentum4(df4["pi_E_2"],df4["pi_px_2"],df4["pi_py_2"],df4["pi_pz_2"]) 
    pi2_2_4Mom = Momentum4(df4["pi2_E_2"],df4["pi2_px_2"],df4["pi2_py_2"],df4["pi2_pz_2"]) 
    pi3_2_4Mom = Momentum4(df4["pi3_E_2"],df4["pi3_px_2"],df4["pi3_py_2"],df4["pi3_pz_2"]) 
    pi0_1_4Mom = Momentum4(df4["pi0_E_1"],df4["pi0_px_1"],df4["pi0_py_1"],df4["pi0_pz_1"])
    pi0_2_4Mom = Momentum4(df4["pi0_E_2"],df4["pi0_px_2"],df4["pi0_py_2"],df4["pi0_pz_2"])

    tau_1_vis = pi_1_4Mom + pi0_1_4Mom#pi2_1_4Mom + pi3_1_4Mom 
    tau_2_vis = pi_2_4Mom + pi0_2_4Mom #pi2_2_4Mom + pi3_2_4Mom 
    
    return tau_1_vis, tau_2_vis
    
    
def get_nu(df4):
    nu_1 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_1"])), df4["gen_nu_eta_1"], df4["gen_nu_phi_1"], df4["gen_nu_p_1"])
    nu_2 = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_2"])), df4["gen_nu_eta_2"], df4["gen_nu_phi_2"], df4["gen_nu_p_2"])
    return nu_1, nu_2


########################################################################################################
#Set up the NN stuff


#need them there, redefine later

i_tau1_e = 0
i_tau1_px = 1
i_tau1_py = 2
i_tau1_pz = 3
i_tau2_e = 4
i_tau2_px = 5
i_tau2_py = 6
i_tau2_pz = 7

i_nu1_px = 8
i_nu1_py = 9
i_nu1_pz = 10

i_nu2_px = 11
i_nu2_py = 12
i_nu2_pz = 13

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


def get_x_y(df4):
    alpha_1, alpha_2 = am.alphas_clamped(df4, decay_mode1, decay_mode2)
    alphas = [alpha_1, alpha_2]
    tau_1_vis, tau_2_vis = get_a1(df4)
    nu_1, nu_2 = get_nu(df4)
    ref = [#smear_px,py                      #0
        #one_d(1.776),                      #1
        #df4["metx"],                   #2
        #df4["mety"],                   #3
        Mom4_to_tf(tau_1_vis.e),       #4
        Mom4_to_tf(tau_1_vis.p_x),     #5
        Mom4_to_tf(tau_1_vis.p_y),     #6
        Mom4_to_tf(tau_1_vis.p_z),     #7
        Mom4_to_tf(tau_2_vis.e),       #8
        Mom4_to_tf(tau_2_vis.p_x),     #9 
        Mom4_to_tf(tau_2_vis.p_y),     #10
        Mom4_to_tf(tau_2_vis.p_z),     #11
        #one_d(125),                    #12
        #Mom4_to_tf(nu_1.e),            #13       corresponding to          #0
        Mom4_to_tf(nu_1.p_x),          #14                                 #1
        Mom4_to_tf(nu_1.p_y),          #15                                 #2
        Mom4_to_tf(nu_1.p_z),          #16                                 #3
        #Mom4_to_tf(nu_2.e),            #17                                 #4
        Mom4_to_tf(nu_2.p_x),          #18                                 #5
        Mom4_to_tf(nu_2.p_y),          #19                                 #6
        Mom4_to_tf(nu_2.p_z),          #20                                 #7
    ]
    y = tf.transpose(ref)
    
    x20 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    #*nu1_guess, *nu2_guess #15
                    *alphas
                    
                    ])
    x = tf.transpose(x20)
    
    return x, y
    
    


####Very important, the batch size must be defined beforehand to get rid of two trainable params !

m_Higgs_squared = 125**2
m_tau_squared = 1.776**2
E_size = 25
B_size = 500#2**10

#For now only
ratio_all = 1
ratio_p = 1
ratio_phi = 1
ratio_tau = 1
ratio_H = 1


x_train, y_train = get_x_y(df_train)
x_val, y_val = get_x_y(df_eval)


input_1 = tf.keras.Input(shape = x_val.shape, name="lab_frame")
x2 = tf.keras.layers.Dense(300, activation = 'relu', name="learning")(input_1)
x3 = tf.keras.layers.Dense(300, activation = 'relu', name="learning2")(x2)
x4 = tf.keras.layers.Dropout(0.2, name="dropout2")(x3)
output = tf.keras.layers.Dense(14, name="output")(x4)


model = tf.keras.Model(
    inputs=[input_1],
    outputs=[output],
)

model.summary()
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae', loss_D_p, loss_phi, loss_p, loss_mass_tau, loss_mass_Higgs])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
    epochs=E_size,
    batch_size = B_size)


res = np.array(model({"lab_frame": x_val}), dtype=np.float64)

nu_1_reco = Momentum4(res[:, i_nu1_px]**2+res[:, i_nu1_py]**2+np.sqrt(res[:, i_nu1_pz]**2), res[:, i_nu1_px], res[:, i_nu1_py], res[:, i_nu1_pz])


nu_2_reco = Momentum4(res[:, i_nu2_px]**2+res[:, i_nu2_py]**2+np.sqrt(res[:, i_nu2_pz]**2), res[:, i_nu2_px], res[:, i_nu2_py], res[:, i_nu2_pz])

#%% Add/remove columns to dataframe -  Thanks Kinglsey
#del df_eval['sv_x_1'], df_eval['sv_y_1'], df_eval['sv_z_1']
#del df_eval['sv_x_2'], df_eval['sv_y_2'], df_eval['sv_z_2']

#df_eval['reco_nu_p_1'] = nu_1_reco.p
#df_eval['reco_nu_p_2'] = nu_2_reco.p
#df_eval['reco_nu_phi_1'] = nu_1_reco.phi
#df_eval['reco_nu_phi_2'] = nu_2_reco.phi
#df_eval['reco_nu_eta_1'] = nu_1_reco.eta
#df_eval['reco_nu_eta_2'] = nu_2_reco.eta

#%%  Write root file
#to save only the a1-a1 decays
df_eval = df4 

treeBranches = {column : str(df_eval[column].dtypes) for column in df_eval}
branchDict = {column : np.array(df_eval[column]) for column in df_eval}
tree = uproot.newtree(treeBranches, title="ntuple", compression=uproot.ZLIB(3))

with uproot.recreate("MVAFILE_AllHiggs_tt_10_10_w_gen.root") as f:
    f["ntuple"] = tree
    f["ntuple"].extend(branchDict)
