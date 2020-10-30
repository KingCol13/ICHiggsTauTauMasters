#!/usr/bin/env python
# coding: utf-8

# ## Task 2

# Setup a NN to regress aco_angle_1 - this will give us some ideas about how we need to setup the NN in order for it to make use of the low-level information and then we can use a similar architecture for our final NN setup. After reading around online a bit one possible reason that the NN is not working very well is because the CP observables depend on what rest frame you determine them in and possibly the NN is not well setup to handle Lorentz boosts into different frames. I found a paper which suggest how to setup the first layers of a NN in order to perform such Lorentz boosts (https://arxiv.org/pdf/1812.09722.pdf) - this might be a good place to start, but of course if you have other ideas you are free to follow themSet up a Neural Network to reconstruct the aco_angle_1 from basic variables. 

#This file is for experimenting with the LBN aim for today: Make this work and have an algorithm that can include both an LBN *trained* layer and also another layer trained on cross products.

import sys
sys.path.append("/eos/home-m/acraplet/.local/lib/python2.7/site-packages")
import uproot 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import xgboost as xgb

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from Cross_product import LBN, LBNLayer, FeatureFactoryBase, FeatureFactory
import tensorflow as tf
import keras

from pylorentz import Momentum4
from pylorentz import Vector4
from pylorentz import Position4

######################################################################################


# loading the tree
tree = uproot.open("/eos/user/d/dwinterb/SWAN_projects/Masters_CP/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
print("\n Tree loaded\n")


# define what variables are to be read into the dataframe
momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2"] #subleading neutral pi 4-momentum

other_features = [ "ip_x_1", "ip_y_1", "ip_z_1",        #leading impact parameter
                   "ip_x_2", "ip_y_2", "ip_z_2",        #subleading impact parameter
                   "y_1_1", "y_1_2"]    # ratios of energies

target = [    "aco_angle_1"]  #acoplanarity angle
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2","rand","wt_cp_ps","wt_cp_sm",
            ]

variables4=(momenta_features+other_features+target+selectors) #copying Kinglsey's way cause it is very clean

df4 = tree.pandas.df(variables4)

df4 = df4[
      (df4["tau_decay_mode_1"] == 1) 
    & (df4["tau_decay_mode_2"] == 1) 
    & (df4["mva_dm_1"] == 1) 
    & (df4["mva_dm_2"] == 1)
]

print(0.7*len(df4),'This is the length') #up to here we are fine

df_ps = df4[
      (df4["rand"]<df4["wt_cp_ps"]/2)     #a data frame only including the pseudoscalars
]

df_sm = df4[
      (df4["rand"]<df4["wt_cp_sm"]/2)     #data frame only including the scalars
]

print("panda Data frame created \n")

df4.head()






#########################Some geometrical functions#####################################


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



############### ALL Hand-calculated parameters #################

#The different *initial* 4 vectors, (E,px,py,pz)
pi_1=np.array([df4["pi_E_1"],df4["pi_px_1"],df4["pi_py_1"],df4["pi_pz_1"]])
pi_2=np.array([df4["pi_E_2"],df4["pi_px_2"],df4["pi_py_2"],df4["pi_pz_2"]])

pi0_1=np.array([df4["pi0_E_1"],df4["pi0_px_1"],df4["pi0_py_1"],df4["pi0_pz_1"]])
pi0_2=np.array([df4["pi0_E_2"],df4["pi0_px_2"],df4["pi0_py_2"],df4["pi0_pz_2"]])

#Charged and neutral pion momenta
pi_1_4Mom=Momentum4(df4["pi_E_1"],df4["pi_px_1"],df4["pi_py_1"],df4["pi_pz_1"])
pi_2_4Mom=Momentum4(df4["pi_E_2"],df4["pi_px_2"],df4["pi_py_2"],df4["pi_pz_2"])

#Same for the pi0
pi0_1_4Mom=Momentum4(df4["pi0_E_1"],df4["pi0_px_1"],df4["pi0_py_1"],df4["pi0_pz_1"])
pi0_2_4Mom=Momentum4(df4["pi0_E_2"],df4["pi0_px_2"],df4["pi0_py_2"],df4["pi0_pz_2"])

#This is the COM frame of the two charged pions w.r.t. which we'll boost
ref_COM_4Mom=Momentum4(pi_1_4Mom+pi_2_4Mom)

energies=[df4["pi_E_1"],df4["pi_E_2"],df4["pi0_E_1"],df4["pi0_E_2"]]

#Lorentz boost everything in the ZMF of the two charged pions
pi0_1_4Mom_star=pi0_1_4Mom.boost_particle(-ref_COM_4Mom)
pi0_2_4Mom_star=pi0_2_4Mom.boost_particle(-ref_COM_4Mom)

#Lorentz boost everything in the ZMF of the two neutral pions
pi_1_4Mom_star=pi_1_4Mom.boost_particle(-ref_COM_4Mom)
pi_2_4Mom_star=pi_2_4Mom.boost_particle(-ref_COM_4Mom)


#calculating the perpependicular component
pi0_1_3Mom_star_perp=cross_product(pi0_1_4Mom_star[1:],pi_1_4Mom_star[1:])
pi0_2_3Mom_star_perp=cross_product(pi0_2_4Mom_star[1:],pi_2_4Mom_star[1:])

#Now normalise:
pi0_1_3Mom_star_perp=pi0_1_3Mom_star_perp/norm(pi0_1_3Mom_star_perp)
pi0_2_3Mom_star_perp=pi0_2_3Mom_star_perp/norm(pi0_2_3Mom_star_perp)

pi0_1_4Mom_star_perp=[pi0_1_4Mom_star[0],*pi0_1_3Mom_star_perp]
pi0_2_4Mom_star_perp=[pi0_1_4Mom_star[0],*pi0_2_3Mom_star_perp]

#Calculating phi_star
phi_CP_unshifted=np.arccos(dot_product(pi0_1_3Mom_star_perp,pi0_2_3Mom_star_perp))

phi_CP=phi_CP_unshifted

#The energy ratios
y_T = np.array(df4['y_1_1']*df4['y_1_2'])

#The O variable
cross=np.array(np.cross(pi0_1_3Mom_star_perp.transpose(),pi0_2_3Mom_star_perp.transpose()).transpose())
bigO=dot_product(pi_2_4Mom_star[1:],cross)

#perform the shift w.r.t. O* sign
phi_CP=np.where(bigO>=0, 2*np.pi-phi_CP, phi_CP)#, phi_CP)

print('len phi', len(phi_CP))


#additionnal shift that needs to be done do see differences between odd and even scenarios, with y=Energy ratios
phi_CP=np.where(y_T>=0, np.where(phi_CP<np.pi, phi_CP+np.pi, phi_CP-np.pi), phi_CP)





################################# Here define the model ##############################
#The target
#target = y #df4[["aco_angle_1"]]
#target=[phi_CP_unshifted, bigO, y_T]
inputs = np.array([pi_1_4Mom_star, pi_2_4Mom_star, pi0_1_4Mom_star, pi0_2_4Mom_star])#, ref_COM_4Mom]


#for cross product to work the way it is currently coded we need to input the data like this
inputs = np.einsum("aij -> jai", inputs) 
x = np.array(inputs,dtype=np.float32)#.transpose()

#outputs = [pi0_1_4Mom_star]#, pi0_2_4Mom_star, pi_1_4Mom_star, pi_2_4Mom_star]
outputs = [pi0_1_3Mom_star_perp, pi0_2_3Mom_star_perp]

#need to do the same with y, this is the right geometry!
outputs = np.einsum("aij -> jai", outputs) 
y = np.array(outputs, dtype=np.float32)#.transpose()

model = tf.keras.models.Sequential()


#all the output we want  in some boosted frame
LBN_output_features = ["cross_product_z", "cross_product_x", "cross_product_y"] 


lbn_layer=LBNLayer((len(x[0]),4,), 4, features=LBN_output_features)

model = tf.keras.models.Sequential([
    #tf.keras.layers.Flatten(input_shape=x.shape),
    lbn_layer,
    #tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(6),
    tf.keras.layers.Reshape((2,3)) #here might be 2,3 instead, yes indeed
])

model.summary()


#Next run it
loss_fn = tf.keras.losses.MeanSquaredError() #try with this function but could do with loss="categorical_crossentropy" instead
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])

loss, acc = model.evaluate(x,  y, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


#set the weights to known values, thanks kinsgley
#weights = [np.eye(4), np.reshape(np.array([1, 1, 0, 0], dtype=np.float32), (4,1))]
#myLBNLayer.set_weights(weights)


# #train model
history = model.fit(x, y, validation_split = 0.3, epochs = 25)


# loss, acc = model.evaluate(x,  y, verbose=2)
# print("Trained model, accuracy: {:5.2f}%".format(100*acc))


hist1 = np.array(model(x)[:,0,0])
hist2 = np.array(y[:,0,0])

# hist3 = np.array(model(x)[3,0])
# hist4 = np.array(y[3,0])

# hist5 = np.array(model(x)[1,2])
# hist6 = np.array(y[1,2])

plt.figure()


#Checking the fraction of rights
difference=y[:,0,0]-model(x)[:,0,0]


# tf.keras.layers.Reshape((1))
# print(difference)

difference=np.reshape(difference, [-1])

k=np.where(difference<=10**(-3),1,0)
print('Fraction of well reconstructed vectors:',np.sum(k)/len(k))

plt.hist(hist1, alpha = 0.5, label = "cross_product component \n fraction of reco at $10^{-3}$:%.3f"%(np.sum(k)/len(k)))
plt.hist(hist2, alpha = 0.5,  label = "True value component")
plt.title("Histogram for cross product ")
plt.xlabel("Vector component")
plt.ylabel("Frequency")
plt.grid()
plt.legend()
plt.savefig("cross_product.png")

#plot traning
plt.figure()
plt.plot(history.history['loss'][10:], label="Training Loss")
plt.plot(history.history['val_loss'][10:], label="Validation Loss")
plt.title("Loss on Iteration")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_cross_product.png")






