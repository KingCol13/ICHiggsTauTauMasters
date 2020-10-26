#!/usr/bin/env python
# coding: utf-8

# ## Task 2

# Setup a NN to regress aco_angle_1 - this will give us some ideas about how we need to setup the NN in order for it to make use of the low-level information and then we can use a similar architecture for our final NN setup. After reading around online a bit one possible reason that the NN is not working very well is because the CP observables depend on what rest frame you determine them in and possibly the NN is not well setup to handle Lorentz boosts into different frames. I found a paper which suggest how to setup the first layers of a NN in order to perform such Lorentz boosts (https://arxiv.org/pdf/1812.09722.pdf) - this might be a good place to start, but of course if you have other ideas you are free to follow themSet up a Neural Network to reconstruct the aco_angle_1 from basic variables. 

#This file is for experimenting with the LBN aim for today: Make this work and have an algorithm that can include both an LBN *trained* layer and also another layer trained on cross products.

import sys
#sys.path.append("/eos/home-m/acraplet/.local/lib/python2.7/site-packages")
#import uproot 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import xgboost as xgb

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from lbn_modified import LBN, LBNLayer
import tensorflow as tf
import keras

from pylorentz import Momentum4
from pylorentz import Vector4
from pylorentz import Position4




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


######################### Invetigate LBN for now ################################

#Question: can we code the lbn so that it does the cross product better than the other layers ?


n_data=1000
x1 = np.random.rand(n_data, 4)
x2 = np.random.rand(n_data, 4)

#x=x.T
#x2=x2.T

y = [np.cross(x1[i][1:-1],x2[i][1:-1]) for i in range(len(x1))]




################################# Here define the model ##############################
inputs = [x1, x2]
x = np.array(inputs,dtype=np.float32)#.transpose()


#The target
target = y #df4[["aco_angle_1"]]
#target=[phi_CP_unshifted, bigO, y_T]
y = np.array(target,dtype=np.float32).transpose() #this is the target


#Now we will try and use lbn to get aco_angle_1 from the 'raw data'
# start a sequential model
model = tf.keras.models.Sequential()


#all the output we want  in some boosted frame
LBN_output_features = ["cross_product"]#,"px","py","pz"]  


#define NN model and compile, now merging 2 3 and all the way to output
model = tf.keras.models.Sequential([
    #tf.keras.layers.Flatten( input_shape=(5,4)),
    LBNLayer((len(x[0]),4,), 4, boost_mode=LBN.PAIRS, features=LBN_output_features),
    #tf.keras.layers.BatchNormalization(), 
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.summary()


#Next run it
loss_fn = tf.keras.losses.MeanSquaredError() #try with this function but could do with loss="categorical_crossentropy" instead
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])


#train model
history = model.fit(x, y, validation_split = 0.3, epochs = 25)
print('Model is trained for the first time')



