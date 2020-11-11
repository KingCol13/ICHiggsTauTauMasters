#!/usr/bin/env python
# coding: utf-8

# ## Task 2

# short test script to investigate the cross product layer / lbn

import sys
sys.path.append("/eos/home-m/acraplet/.local/lib/python2.7/site-packages")
import uproot 
import numpy as np
import pandas as pd
#import functools
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, roc_curve, roc_auc_score
#import xgboost as xgb

#import lbn_modified as lbn_m


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from lbn import LBN, LBNLayer
import tensorflow as tf
import keras

from pylorentz import Momentum4
from pylorentz import Vector4
from pylorentz import Position4


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

def cross_product_numpy(vector):  #to have cross product of vectors (, n, n)
    cross_z=[]
    for i in range(len(vector)):
        cross_z.append([])
        for j in range(len(vector[0])):
            cross_z[i].append([])
            for k in range(len(vector[0])):
                #print(i,j,k)
                #print(vector[i][j][1:])
                #print(cross_z[i][j])
                cross_z[i][j].append(cross_product(vector[i][j][1:], vector[i][k][1:])[2])               
    return (cross_z)

################################### Now the maths ##########################################

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
boost = Momentum4(ref_COM_4Mom[0], -ref_COM_4Mom[1], -ref_COM_4Mom[2], -ref_COM_4Mom[3])

energies=[df4["pi_E_1"],df4["pi_E_2"],df4["pi0_E_1"],df4["pi0_E_2"]]

#Lorentz boost everything in the ZMF of the two charged pions
pi0_1_4Mom_star=pi0_1_4Mom.boost_particle(boost)
pi0_2_4Mom_star=pi0_2_4Mom.boost_particle(boost)

#Lorentz boost everything in the ZMF of the two neutral pions
pi_1_4Mom_star=pi_1_4Mom.boost_particle(boost)
pi_2_4Mom_star=pi_2_4Mom.boost_particle(boost)


#calculating the perpependicular component
pi0_1_3Mom_star_perp=cross_product(pi0_1_4Mom_star[1:],pi_1_4Mom_star[1:])
pi0_2_3Mom_star_perp=cross_product(pi0_2_4Mom_star[1:],pi_2_4Mom_star[1:])


##################################  Now the testing #####################################


#a class for testing and sanity checks:
class My_vectors:
    """
    This is a class to imitate what is done by the LBN and check our cross product function works
    """
    def __init__(self, vector_list):
#         self.px = tf.transpose(vector_list[1, ...])
#         print(self.px)
#         self.py = tf.transpose(vector_list[2, ...])
#         self.pz = tf.transpose(vector_list[3, ...])
        self.px = vector_list[1, ...]
        print(self.px)
        self.py = vector_list[2, ...]
        self.pz = vector_list[3, ...]
        self.pvect = vector_list[1:, ...]
        
        print(self.pvect, 'this is pvect')
        self.n = len(self.px[0]) 
        
        print(self.n)
        
        #this all depends if we are transposed or not ! 
        #technically the second one should always be good since we have (, n, n) or (n, n, )
        #the upper triangle indices, directly taken from lbn 
        self.triu_indices = np.arange(self.n**2).reshape(self.n, self.n)[np.triu_indices(self.n, 1)]
       
    def cross_product_z(self, **opts):
        """
        Z component of cross product between momenta of each pair of particles
        """
        #we need to expand in 2d to have the right matrix arrangement
        cross_z = tf.expand_dims(self.px, axis=-1)*tf.expand_dims(self.py, axis=-2)
       
        #only transpose the two last sides, we want some pairwise operations
        cross_z_T=tf.einsum('aij -> aji', cross_z) 
        
        #print("This is cross_z-cross_z_T", cross_z-cross_z_T)

        #and then substract and keep the right half of the triangle to have cross product
        yy = tf.gather(tf.reshape(cross_z-cross_z_T, [-1, self.n**2]), self.triu_indices, axis=1)
        return yy
    
    def cross_product_x(self, **opts):
        """
        X component of cross product between momenta of each pair of particles
        """
        #we need to expand in 2d to have the right matrix arrangement
        cross_x = tf.expand_dims(self.py, axis=-1)*tf.expand_dims(self.pz, axis=-2)
        
#         cross_x = tf.einsum('aik, akj -> aij', tf.expand_dims(self.py, axis=-1),
#                                                 tf.expand_dims(self.pz, axis=-2))

        #only transpose the two last sides, we want some pairwise operations
        cross_x_T=tf.einsum('aij -> aji', cross_x) 

        #and then substract and keep the right half of the triangle to have cross product
        yy = tf.gather(tf.reshape(cross_x-cross_x_T, [-1, self.n**2]), self.triu_indices, axis=1)
        return yy
    
    def cross_product_y(self, **opts):
        """
        Y component of cross product between momenta of each pair of particles
        """
        #we need to expand in 2d to have the right matrix arrangement
        cross_y = tf.expand_dims(self.pz, axis=-1)*tf.expand_dims(self.px, axis=-2)

        #only transpose the two last sides, we want some pairwise operations
        cross_y_T = tf.einsum('aij -> aji', cross_y) 

        #and then substract and keep the right half of the triangle to have cross product
        yy = tf.gather(tf.reshape(cross_y-cross_y_T, [-1, self.n**2]), self.triu_indices, axis=1)
        return yy

    
#Next: check if we can retrive the true cross products with this operation
#This will hopefully clear out the fog about the transposition
#Check if we can get pi0_1_3Mom_star_perp and pi0_2_3Mom_star_perp from 
# [pi0_1_4Mom_star,pi_1_4Mom_star,pi0_2_4Mom_star,pi_2_4Mom_star]

boosted=np.array([pi0_1_4Mom_star,pi_1_4Mom_star,pi0_2_4Mom_star,pi_2_4Mom_star])

boosted_short=np.array([pi0_1_4Mom_star[:4],pi_1_4Mom_star[:4],pi0_2_4Mom_star[:4],pi_2_4Mom_star[:4]])

#in that case I want them transposed because otherwise they have (4,4,) and not (, 4, 4) as I want

print(boosted.T.shape)

print(pi0_1_4Mom_star[1:])
print(pi_1_4Mom_star[1:])

print(pi_1_4Mom_star[1:][..., 0])

#just performing the appropiate transpose
boosted_right = tf.einsum("aij -> ija", boosted)

my_vect = My_vectors(boosted_right)
cross_x = my_vect.cross_product_x()

print('\n done\n')

print(my_vect.cross_product_z()[0][1], 'is this correct?') #this should be the z component of the first lambda
# print(my_vect.cross_product_z()[1], 'or maybe this one ?')

# print(my_vect.cross_product_x()[0]) #this should be the z component of the first lambda
# print(my_vect.cross_product_x()[1])

# #The cross product gives the same thing
# #print(cross_product_numpy(boosted.T)[0])
# #print(cross_product_numpy(boosted.T)[1])

# print(pi0_1_3Mom_star_perp.T.shape)#maybe need a transpose again, probably actually but won't change anything with the [0][0] component
# print(pi0_1_3Mom_star_perp.T[0])
# print(pi0_1_3Mom_star_perp.T[1])


print('\n now some shapes')
print(pi0_1_3Mom_star_perp.T[...,0].shape)
print(cross_x[...,1].shape)


print('\n now some values')
#print(pi0_1_3Mom_star_perp.T[0,...][0], cross_x[...,0][0])

#it is the 0th component that we need

plt.figure()
plt.plot(pi0_1_3Mom_star_perp.T[...,0], my_vect.cross_product_x()[...,0], label='x component of $\lambda^{1*}_{\perp}$')
plt.plot(pi0_1_3Mom_star_perp.T[...,1], my_vect.cross_product_y()[...,0], label='y component of $\lambda^{1*}_{\perp}$')
plt.plot(pi0_1_3Mom_star_perp.T[...,2], my_vect.cross_product_z()[...,0], label='z component of $\lambda^{1*}_{\perp}$')

plt.plot(pi0_2_3Mom_star_perp.T[...,0], my_vect.cross_product_x()[...,5], label='x component of $\lambda^{2*}_{\perp}$')
plt.plot(pi0_2_3Mom_star_perp.T[...,1], my_vect.cross_product_y()[...,5], label='y component of $\lambda^{2*}_{\perp}$')
plt.plot(pi0_2_3Mom_star_perp.T[...,2], my_vect.cross_product_z()[...,5], label='z component of $\lambda^{2*}_{\perp}$')

plt.legend()
plt.grid()
plt.title('Verification of cross-product formula w/o training', fontsize='xx-large')
plt.xlabel('True component of pi0_{1,2}_3Mom_star_perp', fontsize='large')
plt.ylabel('Cross product estimation', fontsize='large')

#plt.savefig('Cross_prod_working.png')





####################### stuff required to work with a ########################


# a = np.array([[[1,2,3,4], [1,2,3,4], [1,2,3,4], [1,2,3,4]],
#     [[1,2,5,4], [1,2,3,4], [1,6,3,4], [1,2,3,6]],
#     [[1,2,3,4], [1,2,3,4], [1,3,3,4], [1,2,3,4]],
#     [[1,2,3,4], [1,2,3,4], [1,6,3,4], [1,2,3,4]],
#     [[1,2,3,4], [1,2,3,4], [1,6,3,4], [1,2,3,4]]
# ])
# #a should have dimensions (4,4,4), it does, this is good news
# print(a.T.shape)
#Now check if we can put a in the My_vectors class
# my_vect = My_vectors(a) 
# my_vect.cross_product_z()
# print(cross_product_numpy(a))

