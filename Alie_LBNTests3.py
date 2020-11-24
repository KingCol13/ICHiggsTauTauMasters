#!/usr/bin/env python
# coding: utf-8

# ## Task 2

#This is a script for testing with including more particles in the LBN to have the shifts wrt different variables 


import sys
sys.path.append("/eos/home-m/acraplet/.local/lib/python2.7/site-packages")
import uproot 
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from lbn_modified3 import LBN, LBNLayer
import tensorflow as tf


#for some reason pylorentz is installed somewhere differently ?
sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
from pylorentz import Momentum4
from pylorentz import Vector4
from pylorentz import Position4




######################################################################################


# loading the tree
tree = uproot.open("/eos/user/d/dwinterb/SWAN_projects/Masters_CP/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
print("\n Tree loaded\n")
#tree = uproot.open("/eos/user/d/dwinterb/SWAN_projects/Masters_CP/MVAFILE_AllHiggs_tt.root")["ntuple"]

# define what variables are to be read into the dataframe
momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2"] #subleading neutral pi 4-momentum

other_features = [ "ip_x_1", "ip_y_1", "ip_z_1",        #leading impact parameter
                   "ip_x_2", "ip_y_2", "ip_z_2",        #subleading impact parameter
                   "y_1_1", "y_1_2",
                   "gen_phitt"
                 ]    # ratios of energies

target = [ "aco_angle_1", "aco_angle_6", "aco_angle_5", "aco_angle_7"]  #acoplanarity angle
    
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

class Calculation:
    def __init__(self, df):
        #this is the function doing all the calculations manually just takes as input the dataframe
        #The different *initial* 4 vectors, (E,px,py,pz)
        self.pi_1 = np.array([df["pi_E_1"],df["pi_px_1"],df["pi_py_1"],df["pi_pz_1"]])
        self.pi_2 = np.array([df["pi_E_2"],df["pi_px_2"],df["pi_py_2"],df["pi_pz_2"]])

        self.pi0_1 = np.array([df["pi0_E_1"],df["pi0_px_1"],df["pi0_py_1"],df["pi0_pz_1"]])
        self.pi0_2 = np.array([df["pi0_E_2"],df["pi0_px_2"],df["pi0_py_2"],df["pi0_pz_2"]])

        #Charged and neutral pion momenta
        self.pi_1_4Mom = Momentum4(df["pi_E_1"],df["pi_px_1"],df["pi_py_1"],df["pi_pz_1"])
        self.pi_2_4Mom = Momentum4(df["pi_E_2"],df["pi_px_2"],df["pi_py_2"],df["pi_pz_2"])

        #Same for the pi0
        self.pi0_1_4Mom = Momentum4(df["pi0_E_1"],df["pi0_px_1"],df["pi0_py_1"],df["pi0_pz_1"])
        self.pi0_2_4Mom = Momentum4(df["pi0_E_2"],df["pi0_px_2"],df["pi0_py_2"],df["pi0_pz_2"])

        self.impact_param_1 = Momentum4(np.zeros(len(df["ip_x_1"])),df["ip_x_1"],df["ip_y_1"],df["ip_z_1"])
        self.impact_param_2 = Momentum4(np.zeros(len(df["ip_x_2"])),df["ip_x_2"],df["ip_y_2"],df["ip_z_2"])

        #comment or uncomment depending on which aco_angle you want 
        #self.pi0_1_4Mom = self.impact_param_1
        #self.pi0_2_4Mom = self.impact_param_2

        #This is the COM frame of the two charged pions w.r.t. which we'll boost
        self.ref_COM_4Mom = Momentum4(self.pi_1_4Mom+self.pi_2_4Mom)
        boost = Momentum4(self.ref_COM_4Mom[0], -self.ref_COM_4Mom[1], -self.ref_COM_4Mom[2], -self.ref_COM_4Mom[3])
        
        
        boost = -self.ref_COM_4Mom
        #energies=[df4["pi_E_1"],df4["pi_E_2"],df4["pi0_E_1"],df4["pi0_E_2"]]

        #Lorentz boost everything in the ZMF of the two charged pions
        self.pi0_1_4Mom_star = self.pi0_1_4Mom.boost_particle(boost)
        self.pi0_2_4Mom_star = self.pi0_2_4Mom.boost_particle(boost)

        #Lorentz boost everything in the ZMF of the two neutral pions
        self.pi_1_4Mom_star = self.pi_1_4Mom.boost_particle(boost)
        self.pi_2_4Mom_star = self.pi_2_4Mom.boost_particle(boost)


        #calculating the perpependicular component
        pi0_1_3Mom_star_perp=cross_product(self.pi0_1_4Mom_star[1:], self.pi_1_4Mom_star[1:])
        pi0_2_3Mom_star_perp=cross_product(self.pi0_2_4Mom_star[1:], self.pi_2_4Mom_star[1:])

        #Now normalise:
        pi0_1_3Mom_star_perp=pi0_1_3Mom_star_perp/norm(pi0_1_3Mom_star_perp)
        pi0_2_3Mom_star_perp=pi0_2_3Mom_star_perp/norm(pi0_2_3Mom_star_perp)

        self.pi0_1_4Mom_star_perp = [self.pi0_1_4Mom_star[0], pi0_1_3Mom_star_perp[0], 
                                     pi0_1_3Mom_star_perp[1], pi0_1_3Mom_star_perp[2]]

        self.pi0_2_4Mom_star_perp = [self.pi0_1_4Mom_star[0], pi0_2_3Mom_star_perp[0], 
                                     pi0_2_3Mom_star_perp[1], pi0_2_3Mom_star_perp[2]]

        #Calculating phi_star
        self.phi_CP_unshifted = np.arccos(dot_product(pi0_1_3Mom_star_perp,pi0_2_3Mom_star_perp))
        
        print(self.phi_CP_unshifted[:10],'This is phi_CP')

        self.phi_CP = self.phi_CP_unshifted
        
        print(pi0_1_3Mom_star_perp[:,23], 'this is pi0_1_3mom')

        #The energy ratios
        self.y_T = np.array(df['y_1_1']*df['y_1_2'])
        #y_1_1 = (self.pi_1_4Mom_star[0] - self.pi0_1_4Mom_star[0])/(self.pi_1_4Mom_star[0] + self.pi0_1_4Mom_star[0])
        #y_1_2 = (self.pi_2_4Mom_star[0] - self.pi0_2_4Mom_star[0])/(self.pi_2_4Mom_star[0] + self.pi0_2_4Mom_star[0])

        #self.y_T = y_1_1 * y_1_2
        
        #The O variable
        cross = np.array(np.cross(pi0_1_3Mom_star_perp.transpose(),pi0_2_3Mom_star_perp.transpose()).transpose())
        self.bigO = dot_product(self.pi_2_4Mom_star[1:],cross)
        
        print(self.bigO[:10], '\n this is big0')

        #perform the shift w.r.t. O* sign
        
        
        #phi_CP=np.where(self.bigO>=0, 2*np.pi-self.phi_CP_unshifted, self.phi_CP_unshifted)
        self.phi_CP_1 = np.where(self.bigO>=0, 2*np.pi-self.phi_CP_unshifted, self.phi_CP_unshifted)
        
        print(self.phi_CP_1[:10], '\n this is after first shft')

       # self.phi_CP_2 = np.where(self.y_T<=0, self.phi_CP+np.pi, self.phi_CP-np.pi)

        #additionnal shift that needs to be done do see differences between odd and even scenarios, with y=Energy ratios
        self.phi_CP = np.where(self.y_T>=0, np.where(self.phi_CP_1<np.pi, self.phi_CP_1+np.pi, self.phi_CP_1-np.pi), self.phi_CP_1)
        
        self.df = df
        
        print(self.phi_CP[:10], 'this is full')
        
        self.y = df["aco_angle_1"]
    
    def checks(self):

        target = [self.df["aco_angle_1"]]#self.df["aco_angle_7"]]
        y = tf.transpose(tf.convert_to_tensor(target, dtype=np.float32))

        inputs = [self.pi0_1_4Mom, self.pi_1_4Mom, self.pi0_2_4Mom, self.pi_2_4Mom]
        x = tf.convert_to_tensor(inputs, dtype=np.float32)
        x = tf.transpose(x, [2, 0, 1])
        
        k = tf.convert_to_tensor([
                          self.impact_param_1[0], self.impact_param_1[1], self.impact_param_1[2], self.impact_param_1[3],
                          #self.pi0_1_4Mom[0], self.pi0_1_4Mom[1], self.pi0_1_4Mom[2], self.pi0_1_4Mom[3],
                          self.pi_1_4Mom[0], self.pi_1_4Mom[1], self.pi_1_4Mom[2], self.pi_1_4Mom[3],
                          #self.pi0_2_4Mom[0], self.pi0_2_4Mom[1], self.pi0_2_4Mom[2], self.pi0_2_4Mom[3],
                          self.impact_param_2[0], self.impact_param_2[1], self.impact_param_2[2], self.impact_param_2[3],
                          self.pi_2_4Mom[0], self.pi_2_4Mom[1], self.pi_2_4Mom[2], self.pi_2_4Mom[3]],
                         dtype=np.float32)

# the extra info we are giving
        l = tf.convert_to_tensor([self.y_T], dtype=np.float32)

        return x,y,k,l

################################# Here include aco_angle next ##############################

target = Calculation(df_sm)
x,y,k,l = target.checks()
node_nb = 30 #64#48#32#64
need = 'aco_angle_6'
figure_nb = 66



plt.figure()
plt.hist(target.phi_CP, bins = 100, alpha = 0.5, label = 'My guess')
plt.hist(target.y, bins = 100, alpha = 0.5, label = 'aco_angle_1')
plt.grid()
plt.legend()
plt.savefig('aco_angle_6')

raise end
#the lab frame inputs



#all the output we want  in some boosted frame
LBN_output_features = ["only_phi_CP_1"]#"only_phi_CP_1"]#, "only_y_tau"]

#define our LBN layer:
myLBNLayer = LBNLayer((4,4), 4, n_restframes=1, boost_mode=LBN.PRODUCT, features=LBN_output_features)


input_1 = tf.keras.Input(shape = (len(k),), name="lab_frame")
input_2 = tf.keras.Input(shape=(1, ), name="y_tau")  # Variable-length sequence of ints

#output_LBN = tf.keras.layers.Dense(4, activation = 'relu', name='activ_1')(input_1)#myLBNLayer(input_1)
resh = tf.keras.layers.Reshape((4,4), name='reshape')(input_1)

#set the LBN weights to known values, thanks kingsley
#weights = [np.eye(4), np.reshape(np.array([0, 1, 0, 1], dtype=np.float32), (4,1))]
#myLBNLayer.set_weights(weights)


#myLBNLayer.trainable = False 

output_LBN = myLBNLayer(resh)

# Merge all available features into a single large vector via concatenation
x1 = tf.keras.layers.concatenate([output_LBN, input_2]) #out_1])

# Stick a logistic regression for priority prediction on top of the features
x2 = tf.keras.layers.Dense(30, activation = 'relu', name="learning")(x1)
x = tf.keras.layers.Dense(30, activation = 'relu', name="learning2")(x2)

output = tf.keras.layers.Dense(1, name="output")(x)

# Instantiate an end-to-end model predicting both priority and department
model = tf.keras.Model(
    inputs=[input_1, input_2],
    outputs=[output],
)

tf.keras.utils.plot_model(model, "functional_API_shape_1.png", show_shapes=True)
model.summary()

#here we could have different weights to the different outputs if we had them and/or different loss functions
# model.compile(
#     optimizer = tf.keras.optimizers.RMSprop(1e-3),
#     loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#     metrics=["acc"],
# )

loss_fn = tf.keras.losses.MeanSquaredError() #common to the 4 iterations
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])


history = model.fit(
    {"lab_frame": tf.transpose(k), "y_tau": tf.transpose(l)},#tf.transpose(k)
    {"output": y},
    validation_split = 0.3,
    epochs=25,
    batch_size = 32
)


hist1 = np.array(model({"lab_frame": tf.transpose(k), "y_tau": tf.transpose(l)})[:, 0])
hist2 = np.array(y[:, 0])


def frac(d = -2):
    global l, k 
    difference = y[:, 0]-model({"lab_frame": tf.transpose(k), "y_tau": tf.transpose(l)})[:, 0]
    difference = np.reshape(difference, [-1])
    l = np.where(abs(difference)<=10**(d),1,0)
    return float(float(np.sum(l))/len(l))
                               
d = -3
dd = -1
fig = plt.figure('4_fig', figsize=(10,10), frameon = False)
plt.title('Performance with functional API\n%s'%need, fontsize = 'x-large', weight = 'bold')

ax = fig.add_subplot(2,2,1)
plt.hist(hist1, bins = 100, alpha = 0.5, label = "NN %s component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(need, dd, frac(dd), d, frac(d)))
plt.hist(hist2, bins = 100, alpha = 0.5, label = 'True %s - Features: phi_CP_1 (fixed)'%(need))
plt.ylabel("Frequency", fontsize = 'x-large')
plt.xlabel("%s (epsilon = 10e-5)"%(need), fontsize = 'x-large')
plt.grid()
plt.legend()#prop = {'size', 10})

plt.savefig('Test_%i'%(figure_nb))



x,y,k,l = Calculation(df_ps).checks()
hist3 = np.array(model({"lab_frame": tf.transpose(k), "y_tau": tf.transpose(l)})[:, 0])
hist4 = np.array(y[:, 0])

ax = fig.add_subplot(2,2,2)
plt.hist(hist3, bins = 100, alpha = 0.5, label = "NN %s PS component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(need, dd, frac(dd), d, frac(d)))
plt.hist(hist4, bins = 100, alpha = 0.5, label = 'True %s PS - Features: %s'%(need, LBN_output_features[0]))
plt.ylabel("Frequency", fontsize = 'x-large')
plt.xlabel("%s PS (epsilon = 10e-5)"%(need), fontsize = 'x-large')
plt.grid()
plt.legend()#prop = {'size', 10})

plt.savefig('Test_%i'%(figure_nb))

x,y,k,l = Calculation(df_sm).checks()
hist5 = np.array(model({"lab_frame": tf.transpose(k), "y_tau": tf.transpose(l)})[:, 0])
hist6 = np.array(y[:, 0])
plt.figure('4_fig')
ax = fig.add_subplot(2,2,3)
plt.hist(hist5, bins = 100, alpha = 0.5, label = "NN %s SM component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(need, dd, frac(dd), d, frac(d)))
plt.hist(hist6, bins = 100, alpha = 0.5, label = 'True %s SM - Features: %s'%(need, LBN_output_features[0]))
plt.ylabel("Frequency", fontsize = 'x-large')
plt.xlabel("%s SM (epsilon = 10e-5)"%(need), fontsize = 'x-large')
plt.grid()
plt.legend()#prop = {'size', 10})

plt.savefig('Test_%i'%(figure_nb))

ax = fig.add_subplot(2,2,4)
plt.hist(hist5, bins = 100, alpha = 0.5, label = "NN %s SM component"%need)# : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(need, dd, frac(dd), d, frac(d)))
plt.hist(hist3, bins = 100, alpha = 0.5, label = "NN %s PS component"%need)# : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(need, dd, frac(dd), d, frac(d)))
plt.ylabel("Frequency", fontsize = 'x-large')
plt.xlabel("Comparision %s SM-PS (epsilon = 10e-5)"%(need), fontsize = 'x-large')
plt.grid()
plt.legend()#prop = {'size', 10})

plt.savefig('Test_%i'%(figure_nb))


raise END











model = tf.keras.models.Sequential()


fig = plt.figure('4_fig', figsize=(10,10), frameon = False)
plt.title("Neural Network Performance for phi_CP \n[PRODUCT pre_trained, 30r, 30r, MeanSquareError] (25 and 50 epochs)", fontsize = 'xx-large')
plt.axis('off')

#all the output we want  in some boosted frame
LBN_output_features = ["only_big_O"]#"only_phi_CP_1"]#, "only_y_tau"]

#define our LBN layer:
myLBNLayer = LBNLayer((4,4), 4, n_restframes=1, boost_mode=LBN.PRODUCT, features=LBN_output_features)

#set the LBN weights to known values, thanks kingsley
weights = [np.eye(4), np.reshape(np.array([0, 1, 0, 1], dtype=np.float32), (4,1))]
myLBNLayer.set_weights(weights)



#define NN model and compile, now merging 2 3 and all the way to output
model = tf.keras.models.Sequential([
    #define the layer, thanks Kingsley
    myLBNLayer,
    #LBNLayer((4, 4), 4, n_restframes = 1, boost_mode = LBN.PRODUCT, features = LBN_output_features),
    #tf.keras.layers.Dense(node_nb, activation = 'relu'),
    #tf.keras.layers.Dense(node_nb, activation = 'relu'),
    #tf.keras.layers.Dense(node_nb, activation = 'relu'),
    #tf.keras.layers.Dense(1),
    #tf.keras.layers.Reshape((4, 4))
])


loss_fn = tf.keras.losses.MeanSquaredError() #common to the 4 iterations
model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])

# #train model LBN
#history = model.fit(x, y_1, validation_split = 0.3, epochs = 5)
# 

print(model.layers[0].weights, 'the weight')

# #re-use the weights for before  
# model.load_weights("model_aco_1")


# model.add(tf.keras.layers.Dense(node_nb, activation = 'relu'))
# model.add(tf.keras.layers.Dense(node_nb, activation = 'relu'))
# model.add(tf.keras.layers.Dense(1))
# model.layers[0].trainable = False
# model.summary()


# # #Next run it
# loss_fn = tf.keras.losses.MeanSquaredError() #common to the 4 iterations
# model.compile(loss = loss_fn, optimizer = 'adam', metrics = ['mae'])


# # #train model
# # history = model.fit(x, y, validation_split = 0.3, epochs = 25)

# model.load_weights("model_aco_1_phase2")


d = -3
dd = -1

plot_diff=[]

def frac(d = -2):
    difference = y[:, 0]-model(x)[:, 0]
    difference = np.reshape(difference, [-1])
    plot_diff.append(difference[:100])
    print(difference[:10])
    l = np.where(abs(difference)<=10**(d),1,0)
    return float(float(np.sum(l))/len(l))

hist1 = np.array(model(x)[:, 0])
hist2 = np.array(y[:, 0])

plt.figure('4_fig')
ax = fig.add_subplot(2,2,1)
plt.hist(hist1, bins = 100, alpha = 0.5, label = "NN %s component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(need, dd, frac(dd), d, frac(d)))
plt.hist(hist2, bins = 100, alpha = 0.5, label = 'True %s - Features: %s'%(need, LBN_output_features[0]))
plt.ylabel("Frequency", fontsize = 'x-large')
plt.xlabel("%s (epsilon = 10e-5)"%(need), fontsize = 'x-large')
plt.grid()
plt.legend()#prop = {'size', 10})

plt.savefig('Test_%i'%(figure_nb))

x,y = Calculation(df_ps).checks()
hist3 = np.array(model(x)[:, 0])
hist4 = np.array(y[:, 0])


plt.figure('4_fig')
ax = fig.add_subplot(2,2,2)
plt.hist(hist3, bins = 100, alpha = 0.5, label = "NN %s PS component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(need, dd, frac(dd), d, frac(d)))
plt.hist(hist4, bins = 100, alpha = 0.5, label = 'True %s PS - Features: %s'%(need, LBN_output_features[0]))
plt.ylabel("Frequency", fontsize = 'x-large')
plt.xlabel("%s PS (epsilon = 10e-5)"%(need), fontsize = 'x-large')
plt.grid()
plt.legend()#prop = {'size', 10})

plt.savefig('Test_%i'%(figure_nb))

x,y = Calculation(df_sm).checks()
hist5 = np.array(model(x)[:, 0])
hist6 = np.array(y[:, 0])
plt.figure('4_fig')
ax = fig.add_subplot(2,2,3)
plt.hist(hist5, bins = 100, alpha = 0.5, label = "NN %s SM component : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(need, dd, frac(dd), d, frac(d)))
plt.hist(hist6, bins = 100, alpha = 0.5, label = 'True %s SM - Features: %s'%(need, LBN_output_features[0]))
plt.ylabel("Frequency", fontsize = 'x-large')
plt.xlabel("%s SM (epsilon = 10e-5)"%(need), fontsize = 'x-large')
plt.grid()
plt.legend()#prop = {'size', 10})

plt.savefig('Test_%i'%(figure_nb))

ax = fig.add_subplot(2,2,4)
plt.hist(hist5, bins = 100, alpha = 0.5, label = "NN %s SM component")# : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(need, dd, frac(dd), d, frac(d)))
plt.hist(hist3, bins = 100, alpha = 0.5, label = "NN %s PS component")# : fraction($\Delta$<$10^{%i}$)=%.3f \n fraction($\Delta$<$10^{%i}$)=%.3f"%(need, dd, frac(dd), d, frac(d)))
plt.ylabel("Frequency", fontsize = 'x-large')
plt.xlabel("Comparision %s SM-PS (epsilon = 10e-5)"%(need), fontsize = 'x-large')
plt.grid()
plt.legend()#prop = {'size', 10})

plt.savefig('Test_%i'%(figure_nb))

plt.close()
plt.figure('difference')
plt.title('Sanity check on the shifts for aco7\n y_T>0')
for i in range(100):
    if target.y_T[i]>=0:
        plt.plot(plot_diff[0][i], target.phi_CP_1[i]-np.pi, 'bx')
plt.xlabel('difference btw aco_7 and phi_CP_1')
plt.ylabel('phi_CP_1 - pi')
plt.savefig('Test_7')


raise End




