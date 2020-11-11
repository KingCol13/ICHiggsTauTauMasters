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
mpl.use('Agg')
import matplotlib.pyplot as plt
from lbn_modified import LBN, LBNLayer
import tensorflow as tf
import keras


#for some reason pylorentz is installed somewhere differently ?
sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
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

pi0_1_4Mom_star_perp = [pi0_1_4Mom_star[0], pi0_1_3Mom_star_perp[0], pi0_1_3Mom_star_perp[1], pi0_1_3Mom_star_perp[2]]
pi0_2_4Mom_star_perp = [pi0_1_4Mom_star[0], pi0_2_3Mom_star_perp[0], pi0_2_3Mom_star_perp[1], pi0_2_3Mom_star_perp[2]]

#Calculating phi_star
phi_CP_unshifted=np.arccos(dot_product(pi0_1_3Mom_star_perp,pi0_2_3Mom_star_perp))

phi_CP=phi_CP_unshifted

#The energy ratios
y_T = np.array(df4['y_1_1']*df4['y_1_2'])

#The O variable
cross=np.array(np.cross(pi0_1_3Mom_star_perp.transpose(),pi0_2_3Mom_star_perp.transpose()).transpose())
bigO=dot_product(pi_2_4Mom_star[1:],cross)

#perform the shift w.r.t. O* sign
phi_CP_1 = np.where(bigO>=0, 2*np.pi-phi_CP, phi_CP)#, phi_CP)

phi_CP = phi_CP_1

print('len phi', len(phi_CP))

phi_CP_2 = np.where(y_T<=0, phi_CP+np.pi, phi_CP-np.pi)

#additionnal shift that needs to be done do see differences between odd and even scenarios, with y=Energy ratios
phi_CP=np.where(y_T>=0, np.where(phi_CP<np.pi, phi_CP+np.pi, phi_CP-np.pi), phi_CP)



################################# Here include aco_angle next ##############################


#inputs=[phi_CP_unshifted, bigO, y_T]
inputs=[pi0_1_4Mom, pi_1_4Mom, pi0_2_4Mom, pi_2_4Mom]
#inputs = [*pi0_2_4Mom_star_perp, *pi0_1_4Mom_star_perp, df4['y_1_1'], df4['y_1_2'], *pi_2_4Mom_star[1:]]

x = tf.convert_to_tensor(inputs, dtype=np.float32)
x = tf.transpose(x, [2, 0, 1])  #this is the correct transposition ?
# x = np.array(inputs,dtype=np.float32).transpose()

node_nb=30#64#48#32#64

#The target
#target = df4[["aco_angle_1"]]
#target = [pi_1_4Mom_star, pi_2_4Mom_star, pi0_1_4Mom_star, pi0_2_4Mom_star]
target = [phi_CP]#[]#, bigO, y_T]
y = tf.transpose(tf.convert_to_tensor(target, dtype=np.float32))
#tf.transpose(tf.convert_to_tensor(target, dtype=np.float32))
#y = tf.transpose(y, [2, 0, 1])  #this is the correct transposition ?
#y = np.array(target,dtype=np.float32).transpose() #this is the target


# plt.plot(np.cos(phi_CP_unshifted[:100]), 'rx')
# plt.plot(phi_CP_unshifted[:100], 'bx')
# plt.savefig('delete.png')
# raise end

print("\n the std of y", tf.math.reduce_std(y))
print("\n the std of x", tf.math.reduce_std(x))


#Now we will try and use lbn to get aco_angle_1 from the 'raw data'
# start a sequential model
model = tf.keras.models.Sequential()



fig = plt.figure(figsize=(10,10), frameon = False)
plt.title("Neural Network Performance for phi_CP \n Single input feature, [PRODUCT, 30r, 30r, MeanSquareError] (100 epochs)", fontsize = 'xx-large')
#plt.axis('off')

#all the output we want  in some boosted frame
LBN_output_features = ["only_phi_CP_1", "only_y_tau"]#, "y_tau", "big_O"]#, "pi0_1_star", "pi_1_star", "pi0_2_star", "pi0_1_star"], "lambda_1_perp", "lambda_2_perp", ""E", "px", "py", "pz"]


#define NN model and compile, now merging 2 3 and all the way to output
model = tf.keras.models.Sequential([
    #define the layer, thanks Kingsley
    LBNLayer((4, 4), 4, n_restframes = 1, boost_mode = LBN.PRODUCT, features = LBN_output_features),
    tf.keras.layers.Dense(node_nb, activation = 'relu'),
    tf.keras.layers.Dense(node_nb, activation = 'relu'),
    tf.keras.layers.Dense(1),
    #tf.keras.layers.Reshape((4, 4))
])

#Next run it
loss_fn = tf.keras.losses.MeanSquaredError() #common to the 4 iterations
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])


#train model
history = model.fit(x, y, validation_split = 0.3, epochs = 100)



d = -3
dd = -1

def frac(d = -2):
    difference = y[:, 0]-model(x)[:, 0]
    difference = np.reshape(difference, [-1])
    print(difference[:10])
    l = np.where(abs(difference)<=10**(d),1,0)
    return float(float(np.sum(l))/len(l))

hist1 = np.array(model(x)[:, 0])
hist2 = np.array(y[:, 0])

#ax = fig.add_subplot(2,2,1)
plt.hist(hist1, bins = 100, alpha = 0.5, label = "NN $\phi_{CP}$ component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(dd, frac(dd), d, frac(d)))
plt.hist(hist2, bins = 100, alpha = 0.5, label = 'True $\phi_{CP}$ - Features: %s'%LBN_output_features[0])
plt.ylabel("Frequency", fontsize = 'x-large')
plt.xlabel("phi_CP (epsilon = 10e-5)", fontsize = 'x-large')
plt.grid()
plt.legend()#prop = {'size', 10})

plt.savefig('Test_31')

raise End
























########################## more subplots ###############################

#all the output we want  in some boosted frame
LBN_output_features = ["E", "px", "py", "pz"]#, "y_tau", "big_O"]#, "pi0_1_star", "pi_1_star", "pi0_2_star", "pi0_1_star"], "lambda_1_perp", "lambda_2_perp", "


#define NN model and compile, now merging 2 3 and all the way to output
model = tf.keras.models.Sequential([
    #define the layer, thanks Kingsley
    LBNLayer((4, 4), 4, n_restframes = 1, boost_mode = LBN.PRODUCT, features = LBN_output_features),
    tf.keras.layers.Dense(node_nb, activation = 'relu'),
    tf.keras.layers.Dense(node_nb, activation = 'relu'),
    tf.keras.layers.Dense(1),
    #tf.keras.layers.Reshape((4, 4))
])


#Next run it
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])


#train model
history = model.fit(x, y, validation_split = 0.3, epochs = 25)
print('Model is trained.')

hist1 = np.array(model(x)[:, 0])
hist2 = np.array(y[:, 0])

ax = fig.add_subplot(2,2,2)
plt.hist(hist1, bins = 100, alpha = 0.5, label = "NN $\phi_{CP}^{un}$ component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(dd, frac(dd), d, frac(d)))
plt.hist(hist2, bins = 100, alpha = 0.5, label = 'True $\phi_{CP}^{un}$ component - Inputs: E, px, py, pz')
plt.grid()
plt.legend()

plt.savefig('Test_4')


#all the output we want  in some boosted frame
LBN_output_features = ["only_phi_CP_un", "only_big_O", "only_y_tau"]#, "y_tau", "big_O"]#, "pi0_1_star", "pi_1_star", "pi0_2_star", "pi0_1_star"], "lambda_1_perp", "lambda_2_perp", "


#define NN model and compile, now merging 2 3 and all the way to output
model = tf.keras.models.Sequential([
    #define the layer, thanks Kingsley
    LBNLayer((4, 4), 4, n_restframes = 1, boost_mode = LBN.PRODUCT, features = LBN_output_features),
    tf.keras.layers.Dense(node_nb, activation = 'relu'),
    tf.keras.layers.Dense(node_nb, activation = 'relu'),
    tf.keras.layers.Dense(1),
    #tf.keras.layers.Reshape((4, 4))
])


#Next run it
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])


#train model
history = model.fit(x, y, validation_split = 0.3, epochs = 25)
print('Model is trained.')

hist1 = np.array(model(x)[:, 0])
hist2 = np.array(y[:, 0])

ax = fig.add_subplot(2,2,3)
plt.hist(hist1, bins = 100, alpha = 0.5, label = "NN $\phi_{CP}^{un}$ component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(dd, frac(dd), d, frac(d)))
plt.hist(hist2, bins = 100, alpha = 0.5, label = 'True $\phi_{CP}^{un}$ component - Inputs: cos($\phi_{CP}^{un}$), big_O, $y_{\tau}$')
plt.ylabel("Frequency")
plt.xlabel("phi_cp_unshifted")
plt.grid()
plt.legend()

plt.savefig('Test_4')

#all the output we want  in some boosted frame
LBN_output_features = ["lambda_1_perp", "lambda_2_perp", "only_big_O", "only_y_tau"]#, "y_tau", "big_O"]#, "pi0_1_star", "pi_1_star", "pi0_2_star", "pi0_1_star"], "lambda_1_perp", "lambda_2_perp", "


#define NN model and compile, now merging 2 3 and all the way to output
model = tf.keras.models.Sequential([
    #define the layer, thanks Kingsley
    LBNLayer((4, 4), 4, n_restframes = 1, boost_mode = LBN.PRODUCT, features = LBN_output_features),
    tf.keras.layers.Dense(node_nb, activation = 'relu'),
    tf.keras.layers.Dense(node_nb, activation = 'relu'),
    tf.keras.layers.Dense(1),
    #tf.keras.layers.Reshape((4, 4))
])


#Next run it
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])


#train model
history = model.fit(x, y, validation_split = 0.3, epochs = 25)
print('Model is trained.')

hist1 = np.array(model(x)[:, 0])
hist2 = np.array(y[:, 0])

ax = fig.add_subplot(2,2,4)
plt.hist(hist1, bins = 100, alpha = 0.5, label = "NN $\phi_{CP}^{un}$ component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(dd, frac(dd), d, frac(d)))
plt.hist(hist2, bins = 100, alpha = 0.5, label = 'True $\phi_{CP}^{un}$ component - Inputs: $\lambda_{\perp}^{1,2}$, big_O, $y_{\tau}$')
plt.xlabel("phi_cp_unshifted")
plt.grid()
plt.legend()

plt.savefig('Test_4')





################### To plot the reconstruction quality of a 4D vector ##############

# def frac(i, d = -2):
#     difference = y[:, 0, i]-model(x)[:, 0, i]
#     difference = np.reshape(difference, [-1])
#     print(difference[:10])
#     l = np.where(abs(difference)<=10**(d),1,0)
#     print(l[:10])
    
#     print (float(float(np.sum(l))/len(l)))
#     return float(float(np.sum(l))/len(l))

# hist1 = np.array(model(x)[:, 0, 0])
# hist2 = np.array(y[:, 0, 0])


# hist3 = np.array(model(x)[:, 0, 1])
# hist4 = np.array(y[:, 0, 1])

# hist5 = np.array(model(x)[:, 0, 2])
# hist6 = np.array(y[:, 0, 2])

# hist7 = np.array(model(x)[:, 0, 3])
# hist8 = np.array(y[:, 0, 3])

# dd = 0
# dd2 = -2

# fig = plt.figure(figsize=(10,10), frameon = False)
# plt.title("Neural Network Performance for lambda perp \n basics + normalised perp + tier2 features (25 epochs)")
# plt.axis('off')

# ax = fig.add_subplot(2,2,1)#, constrained_layout=True)
# plt.hist(hist1, bins = 100, alpha = 0.5, label = "NN E component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(dd, frac(0, dd), dd2, frac(0, dd2)))
# plt.hist(hist2, bins = 100, alpha = 0.5, label = "True E component")
# plt.ylabel("Frequency")
# plt.grid()
# plt.legend()

# ax = fig.add_subplot(2,2,2)#, constrained_layout=True)
# plt.hist(hist3, bins = 100, alpha = 0.5, label = "NN px component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(dd,frac(1, dd), dd2,frac(1, dd2)))
# plt.hist(hist4, bins = 100, alpha = 0.5, label = "True px component")
# plt.grid()
# plt.legend()

# ax = fig.add_subplot(2,2,3)#, constrained_layout=True)
# plt.hist(hist5, bins = 100, alpha = 0.5, label = "NN py component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(dd,frac(2, dd), dd2,frac(2, dd2)))
# plt.hist(hist6, bins = 100, alpha = 0.5, label = "True py component")
# plt.grid()
# plt.ylabel("Frequency")
# plt.xlabel("4Vector component")
# plt.legend()

# ax = fig.add_subplot(2,2,4)#, constrained_layout=True)
# plt.hist(hist7, bins = 100, alpha = 0.5, label = "NN pz component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(dd,frac(3, dd), dd2,frac(3, dd2)))
# plt.hist(hist8, bins = 100, alpha = 0.5, label = "True pz component")
# plt.grid()
# plt.legend()
# plt.xlabel("4Vector component")


# plt.savefig("Lambda_basics_perpN_tier2.png")





##### After model is trained, try to add some layers and train again, does it do better then ? ##############

#The target
#target2 = df4[["aco_angle_1"]]
#y2 = np.array(target2,dtype=np.float32)#.transpose() #this is the second target


#Adding the new layer to it
#model.add(tf.keras.layers.Dense(32, activation='relu'))
#model.add(tf.keras.layers.Dense(32, activation='relu'))
#model.add(tf.keras.layers.Dense(1))
#print('New layers are added')

# Freeze all layers except the last three.
#for layer in model.layers[:-3]:
  #layer.trainable = False


#In doubt: compile it again, maybe not needed
#loss_fn = tf.keras.losses.MeanSquaredError() #try with this function but could do with loss="categorical_crossentropy" instead
#model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])

#print("\n the std of y2", tf.math.reduce_std(y2))

#and train model again
#history = model.fit(x, y2, validation_split = 0.3, epochs = 25)
#print('Model is trained for the second time (FREEZING)')



############################# Try to glue models together ###########################################

#encoder_input = keras.Input(shape=(28, 28, 1), name="original_img") 
#just copy paste from https://www.tensorflow.org/guide/keras/functional#all_models_are_callable_just_like_layers
#x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
#x = layers.Conv2D(32, 3, activation="relu")(x)
#x = layers.MaxPooling2D(3)(x)
#x = layers.Conv2D(32, 3, activation="relu")(x)
#x = layers.Conv2D(16, 3, activation="relu")(x)
#encoder_output = layers.GlobalMaxPooling2D()(x)

#encoder = keras.Model(encoder_input, encoder_output, name="encoder")
#encoder.summary()

#decoder_input = keras.Input(shape=(16,), name="encoded_img")
#x = layers.Reshape((4, 4, 1))(decoder_input)
#x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
#x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
#x = layers.UpSampling2D(3)(x)
#x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
#decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

#decoder = keras.Model(decoder_input, decoder_output, name="decoder")
#decoder.summary()

#autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
#encoded_img = encoder(autoencoder_input)
#decoded_img = decoder(encoded_img)
#autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
#autoencoder.summary()









#save it and re-use later
#model.save("From_2_to_output.model")


#plot result()



