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
from lbn import LBN, LBNLayer
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



######################### Investigate LBN for now #################################

#Question: does having the COM as input help at all the LBN ?

inputs = [pi_1_4Mom, pi_2_4Mom, pi0_1_4Mom, pi0_2_4Mom, ref_COM_4Mom]
x = np.array(inputs, dtype=np.float32).transpose()

outputs = [pi0_1_4Mom_star, pi0_2_4Mom_star, pi_1_4Mom_star, pi_2_4Mom_star]
y = np.array(outputs, dtype=np.float32).transpose()


LBN_output_features = ["E", "px","py","pz","pt","pair_dy"]

model = tf.keras.models.Sequential([
    #tf.keras.layers.Flatten( input_shape=(5,4)),
    LBNLayer((len(x[0]),4,), 120, boost_mode=LBN.PAIRS, features=LBN_output_features),
    #tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Reshape((4,4))
])

loss_fn = tf.keras.losses.MeanSquaredError() #try with this function but could do with loss="categorical_crossentropy" instead
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])


#train model
history = model.fit(x, y, validation_split = 0.3, epochs = 25)
print('Model is trained for the first time')


hist1 = np.array(model(x)[:,0][0])
hist2 = np.array(y[:,0][0])


plt.figure()
plt.hist(hist1, label = "LBN guess")
plt.hist(hist2, label = "True")
plt.title("Histogram of Neural Network Performance for LBN-L1")
plt.xlabel("Boosted 4 momentum")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("test_LBN.png")




raise END



################################# Here define the model ##############################

#Now try and include the LBN
#x = tf.convert_to_tensor(np.array(df4[momenta_features],dtype=np.float32), dtype=np.float32)


#try the different input 'geometry'
#inputs=[phi_CP_unshifted, bigO, y_T]
inputs=[*pi0_2_4Mom_star_perp, *pi0_1_4Mom_star_perp, *pi_2_4Mom_star,*pi_1_4Mom_star]
#inputs = [*pi0_2_4Mom_star_perp, *pi0_1_4Mom_star_perp, df4['y_1_1'], df4['y_1_2'], *pi_2_4Mom_star[1:]]
x = np.array(inputs,dtype=np.float32).transpose()

node_nb=64#48#32#64

#The target
target = df4[["aco_angle_1"]]
#target=[phi_CP_unshifted, bigO, y_T]
y = np.array(target,dtype=np.float32)#.transpose() #this is the target

print("\n the std of y", tf.math.reduce_std(y))
print("\n the std of x", tf.math.reduce_std(x))


#Now we will try and use lbn to get aco_angle_1 from the 'raw data'
# start a sequential model
model = tf.keras.models.Sequential()


#all the output we want  in some boosted frame
output = ["E", "px", "py", "pz"]  


#define NN model and compile, now merging 2 3 and all the way to output
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(node_nb, activation='relu', input_shape=(len(x[0]),)),
    tf.keras.layers.Dense(node_nb, activation='relu', input_shape=(len(x[0]),)),   
    tf.keras.layers.Dense(3), #this is the glue
    tf.keras.layers.Dense(32, activation='relu', input_shape=(len(x[0]),)),
    tf.keras.layers.Dense(32, activation='relu', input_shape=(len(x[0]),)),
    tf.keras.layers.Dense(1) #this is the output, maybe phi_CP
    #tf.keras.layers.Reshape((4, 4))
])

model.summary()


#Next run it
loss_fn = tf.keras.losses.MeanSquaredError() #try with this function but could do with loss="categorical_crossentropy" instead
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])


#train model
history = model.fit(x, y, validation_split = 0.3, epochs = 25)
print('Model is trained for the first time')



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




hist1 = np.array(model(x)[:,0])
hist2 = np.array(y[:,0])

plt.figure()
plt.hist(hist1, label = "phi_CP component")
plt.hist(hist2, label = "True phi_CP component")
plt.title("Histogram of Neural Network Performance for L2-Output")
plt.xlabel("Phi_CP")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("Good_CP.png")




#save it and re-use later
#model.save("From_2_to_output.model")


#plot result()



