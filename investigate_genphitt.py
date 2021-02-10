#This is a script to run the bckg to check many different configurations of inputs when trying to regress neutrino momenta in the a1-rho channel

print('Hello World')
import sys
#sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
import uproot 
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from lbn_modified3 import LBN, LBNLayer
import tensorflow as tf


#for some reason pylorentz is installed somewhere differently ?
#sys.path.append("/eos/home-a/acraplet/.local/lib/python2.7/site-packages")
sys.path.append("/home/acraplet/Alie/Masters/ICHiggsTauTauMasters/")
from pylorentz import Momentum4
from pylorentz import Vector4
from pylorentz import Position4

# loading the tree
tree = uproot.open("/home/acraplet/Alie/Masters/MVAFILE_AllHiggs_tt.root")["ntuple"]
#tree = uproot.open("/eos/user/d/dwinterb/SWAN_projects/Masters_CP/MVAFILE_GluGluHToTauTauUncorrelatedDecay_Filtered_tt_2018.root")["ntuple"]
print("\n Tree loaded\n")


other_features = [ "gen_phitt",
                 ]    # ratios of energies

target = [ "aco_angle_1", 
          #"aco_angle_5"#" pv_angle"#, "aco_angle_5", "aco_angle_7"
         ]  
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2","wt_cp_ps","wt_cp_sm", "wt_cp_mm"
            ]

variables4=(other_features+target+selectors) 
print('Check 1')
df4 = tree.pandas.df(variables4)

df4 = df4[
      (df4["tau_decay_mode_1"] == 10) 
    & (df4["tau_decay_mode_2"] == 10) 
    & (df4["mva_dm_1"] == 10) 
    & (df4["mva_dm_2"] == 10)
    & (df4["aco_angle_1"] > -4000)
    #& (df4["wt_cp_mm"] != 1)
    #& (df4["wt_cp_sm"] != 1)
    #& (df4["wt_cp_ps"] != 1)
    #& (df4["gen_nu_p_2"] > -4000)
]

print(0.7*len(df4),'This is the length') #up to here we are fine

################# Calculation - clean #################

theta = -np.pi/2
reco_phitt2 = []
increment = 0.005

#len(df4['gen_phitt'])
for i in range(len(df4['gen_phitt'])):
    sm = np.array(df4['wt_cp_sm'])[i]
    mm = np.array(df4['wt_cp_mm'])[i]
    ps = np.array(df4['wt_cp_ps'])[i]
    weight_max = 0
    theta = -np.pi/2
    print(i/len(df4['gen_phitt']))
    while theta <= np.pi/2:
        event_weight = np.cos(theta)*np.cos(theta)*sm + np.sin(theta)*np.sin(theta)*ps + 2*np.cos(theta)*np.sin(theta)*(mm-sm/2-ps/2);
        
        
        if event_weight >= weight_max:
            #print('Old best weight: ', weight_max, 'new  best weight = ', event_weight, '\n')
            #print('Target: ', np.array(df4['gen_phitt'])[i], 'guess: ', (theta*180)/np.pi, 'i: ', i, '\n')
            weight_max = event_weight
            theta_save = theta
        theta = theta + increment
        
    reco_phitt2.append((180*theta_save)/np.pi)
    
print(reco_phitt2[:100], '\n')
print(df4['gen_phitt'][:100], '\n')
    



plt.plot(df4['gen_phitt'][:len(reco_phitt2)], np.array(reco_phitt2), 'x')
plt.ylabel('reco_phitt', fontsize = 'x-large')
plt.xlabel('gen phitt', fontsize = 'x-large')
plt.plot([-90, 90], [-90, 90], 'k--', label = 'x = y')
plt.grid()
plt.legend(prop = {'size': 16})
plt.show()


hist = np.array(np.cos(df4['gen_phitt'][:len(reco_phitt2)]) - np.cos(reco_phitt2))
plt.hist(hist, bins = 50, label = 'cos(gen_phi_tt) - cos(reco_phitt)\nMean difference: %.2f, diff std: %.3f' %(hist.mean(), hist.std()))
plt.legend(prop = {'size': 16})
plt.xlabel('Cosine difference', fontsize = 'x-large')
plt.grid()
plt.show()







hist = np.array(df4['gen_phitt'][:len(reco_phitt2)] - reco_phitt2)
plt.hist(hist, bins = 1000, label = 'gen_phi_tt - reco_phitt\nMean difference: %.2f, diff std: %.3f' %(hist.mean(), hist.std()))
plt.legend(prop = {'size': 16})
plt.xlabel('genphitt difference', fontsize = 'x-large')
plt.grid()
plt.show()








########## Training phase - best guess + aco_angle_1 ###############



y = tf.convert_to_tensor(df4['gen_phitt'], dtype = 'float32')
        
x = [tf.convert_to_tensor(df4['aco_angle_1'], dtype = 'float32'),
     tf.convert_to_tensor(reco_phitt2, dtype = 'float32'),
     #tf.convert_to_tensor(df4['wt_cp_sm'], dtype = 'float32'),
     #tf.convert_to_tensor(df4['wt_cp_ps'], dtype = 'float32'),
    ]

x = tf.transpose(x)

#2) set-up the model
trainFrac = 0.7
nodes1 = 300
nodes2 = 300
drop = 0.1
E_size = 25
B_size = 500

numTrain = int(trainFrac*x.shape[0])

print(numTrain)
x_train = x[:numTrain]
y_train = y[:numTrain]

x_val = x[numTrain:]
y_val = y[numTrain:]

input_1 = tf.keras.Input(shape = x.shape, name="lab_frame")
x2 = tf.keras.layers.Dense(nodes1, activation = 'relu', name="learning")(input_1)
x3 = tf.keras.layers.Dense(nodes2, activation = 'relu', name="learning2")(x2)
x4 = tf.keras.layers.Dropout(drop, name="dropout2")(x3)
output = tf.keras.layers.Dense(1, name="output")(x4)

model = tf.keras.Model(
    inputs=[input_1],
    outputs=[output],
)

model.summary()

model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = 'adam', metrics = ['mae'])#, loss_mass_Higgs, loss_mass_tau, loss_D_p])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
    epochs=E_size,
    batch_size = B_size)

x = 0
y = 0

diff = np.array(y_val-tf.transpose(model({"lab_frame": x_val}))[0])
print(diff.mean(), diff.std())



plt.hist(diff, bins = 1000, label='genphitt - regressed\nmean = %.2f, std  = %.2f'  %(diff.mean(), diff.std()))
plt.title('Regression quality - wt = 1 removed\nInputs = {aco_angle1, calculated phitt}', fontsize = 'xx-large')
plt.xlabel('genphitt - regressedphi_tt', fontsize = 'x-large')
plt.grid()
plt.legend(prop={'size':15})
plt.show()



#next step: smear the weights, to look like the ones we would have obtained with regressed neutrinos

#Scenario 1: regressed neutrinos, with weights std and mean as per tAUsPINNER

m_reg_nu_0 = 0.13
m_reg_nu_0p25 = 0.16
m_reg_nu_0p5 = 0.15

std_reg_nu_0 = 0.79
std_reg_nu_0p25 = 0.80
std_reg_nu_0p5 = 0.79

wt_cp_sm_smeared1 = []
wt_cp_mm_smeared1 = []
wt_cp_ps_smeared1 = []

for i in range(len(df4['wt_cp_sm'])):
    wt_cp_sm_smeared1.append(np.array(df4['wt_cp_sm'])[i]+np.random.normal(m_reg_nu_0, std_reg_nu_0))
    wt_cp_mm_smeared1.append(np.array(df4['wt_cp_mm'])[i]+np.random.normal(m_reg_nu_0p25, std_reg_nu_0p25))
    wt_cp_ps_smeared1.append(np.array(df4['wt_cp_ps'])[i]+np.random.normal(m_reg_nu_0p5, std_reg_nu_0p5))

theta = -np.pi/2
reco_phitt_smeared = []
increment = 0.005

#len(df4['gen_phitt'])
for i in range(len(df4['gen_phitt'])):
    sm = wt_cp_sm_smeared1[i] 
    mm = wt_cp_mm_smeared1[i]
    ps = wt_cp_ps_smeared1[i]
    weight_max = 0
    theta = -np.pi/2
    print(i/len(df4['gen_phitt']))
    while theta <= np.pi/2:
        event_weight = np.cos(theta)*np.cos(theta)*sm + np.sin(theta)*np.sin(theta)*ps + 2*np.cos(theta)*np.sin(theta)*(mm-sm/2-ps/2);
        
        
        if event_weight >= weight_max:
            #print('Old best weight: ', weight_max, 'new  best weight = ', event_weight, '\n')
            #print('Target: ', np.array(df4['gen_phitt'])[i], 'guess: ', (theta*180)/np.pi, 'i: ', i, '\n')
            weight_max = event_weight
            theta_save = theta
        theta = theta + increment
        
    reco_phitt_smeared.append((180*theta_save)/np.pi)
    
print(reco_phitt_smeared[:100], '\n')
print(df4['gen_phitt'][:100], '\n')



plt.hist(wt_cp_sm_smeared1 - np.array(df4['wt_cp_sm']), bins = 100)
plt.savefig('regressed_nu_fig-1')
plt.close()


plt.title('Phitt calculated from weights found\nwith smeared neutrinos - 0.005', fontsize = 'xx-large')
plt.plot(df4['gen_phitt'][:len(reco_phitt_smeared)], np.array(reco_phitt_smeared), 'x')
plt.ylabel('reco_phitt(weights from regressed nus)', fontsize = 'x-large')
plt.xlabel('gen phitt', fontsize = 'x-large')
plt.plot([-90, 90], [-90, 90], 'k--', label = 'x = y')
plt.grid()
plt.legend(prop = {'size': 16})
plt.savefig('regressed_nu_fig0')
plt.close()


plt.hist2d(df4['gen_phitt'][:len(reco_phitt_smeared)], np.array(reco_phitt_smeared), 50)
plt.plot([-90, 90], [-90, 90], 'k--', label = 'x=y')
plt.xlabel('Gen_phitt', fontsize = 'x-large')
plt.ylabel('phitt from wts found w/regressed nus', fontsize = 'x-large')
plt.title('Phitt calculated from weights found\nwith smeared neutrinos - 0.005', fontsize = 'xx-large')
plt.legend(prop = {'size': 16}, loc= 6)
plt.savefig('regressed_nu_fig1')
plt.close()


hist = np.array(np.cos(df4['gen_phitt'][:len(reco_phitt_smeared)]) - np.cos(reco_phitt_smeared))
plt.title('Phitt calculated with smeared weights\ncorresponding to regressed neutrinos - 0.005', fontsize = 'xx-large')
plt.hist(hist, bins = 50, label = 'cos(gen_phi_tt) - cos(reco_phitt)\nMean difference: %.2f, diff std: %.3f' %(hist.mean(), hist.std()))
plt.legend(prop = {'size': 16})
plt.xlabel('Cosine difference', fontsize = 'x-large')
plt.grid()
plt.savefig('regressed_nu_fig2')
plt.close()


plt.title('Phitt calculated with smeared weights\ncorresponding to regressed neutrinos - 0.005', fontsize = 'xx-large')
hist = np.array(df4['gen_phitt'][:len(reco_phitt_smeared)] - reco_phitt_smeared)
plt.hist(hist, bins = 1000, label = 'gen_phi_tt - reco_phitt\nMean difference: %.2f, diff std: %.3f' %(hist.mean(), hist.std()))
plt.legend(prop = {'size': 16})
plt.xlabel('genphitt difference', fontsize = 'x-large')
plt.grid()
plt.savefig('regressed_nu_fig3')
plt.close()


################# Nowwww - NN improvement ?     ###########
y = tf.convert_to_tensor(df4['gen_phitt'], dtype = 'float32')
        
x = [tf.convert_to_tensor(df4['aco_angle_1'], dtype = 'float32'),
     tf.convert_to_tensor(reco_phitt_smeared, dtype = 'float32'),
     #tf.convert_to_tensor(df4['wt_cp_sm'], dtype = 'float32'),
     #tf.convert_to_tensor(df4['wt_cp_ps'], dtype = 'float32'),
    ]

x = tf.transpose(x)

#2) set-up the model
trainFrac = 0.7
nodes1 = 300
nodes2 = 300
drop = 0.1
E_size = 25
B_size = 500

numTrain = int(trainFrac*x.shape[0])

print(numTrain)
x_train = x[:numTrain]
y_train = y[:numTrain]

x_val = x[numTrain:]
y_val = y[numTrain:]

input_1 = tf.keras.Input(shape = x.shape, name="lab_frame")
x2 = tf.keras.layers.Dense(nodes1, activation = 'relu', name="learning")(input_1)
x3 = tf.keras.layers.Dense(nodes2, activation = 'relu', name="learning2")(x2)
x4 = tf.keras.layers.Dropout(drop, name="dropout2")(x3)
output = tf.keras.layers.Dense(1, name="output")(x4)

model = tf.keras.Model(
    inputs=[input_1],
    outputs=[output],
)

model.summary()

def loss_cosines(y_true, y_pred):
    delta = 40*(tf.cos(y_true) - tf.cos(y_pred))**2 + (y_true - y_pred) ** 2
    return tf.convert_to_tensor(delta) #tone it up

#tf.keras.losses.CosineSimilarity()
#loss_cosines
# tf.keras.losses.MeanSquaredError() +
model.compile(loss = loss_cosines, optimizer = 'adam', metrics = ['mae'])#, loss_mass_Higgs, loss_mass_tau, loss_D_p])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
    epochs=E_size,
    batch_size = B_size)

x = 0
y = 0

diff = np.array(y_val-2*tf.transpose(model({"lab_frame": x_val}))[0])
print(diff.mean(), diff.std())



plt.hist(diff, bins = 1000, label='genphitt - 2regressed\nmean = %.2f, std  = %.2f'  %(diff.mean(), diff.std()))
plt.title('Regression quality - wt = 1 removed\nInputs = {aco_angle1, calculated phitt from regressed nu_wieghts}', fontsize = 'xx-large')
plt.xlabel('genphitt - 2regressedphi_tt', fontsize = 'x-large')
plt.grid()
plt.legend(prop={'size':15})
plt.savefig('regressed_nu_after_NN_fig1')
plt.close()

diff = np.array(y_val-tf.transpose(model({"lab_frame": x_val}))[0])
print(diff.mean(), diff.std())

plt.hist(diff, bins = 1000, label='genphitt - regressed\nmean = %.2f, std  = %.2f'  %(diff.mean(), diff.std()))
plt.title('Regression quality - wt = 1 removed - MSE and Cosineloss\nInputs = {aco_angle1, calculated phitt from regressed nu_wieghts}', fontsize = 'xx-large')
plt.xlabel('genphitt - regressedphi_tt', fontsize = 'x-large')
plt.grid()
plt.legend(prop={'size':15})
plt.savefig('regressed_nu_after_NN_fig2')
plt.close()

diff = np.array(np.cos(y_val)-np.cos(2*tf.transpose(model({"lab_frame": x_val}))[0]))
print(diff.mean(), diff.std())
plt.hist(diff, bins = 1000, label='genphitt - regressed\nmean = %.2f, std  = %.2f'  %(diff.mean(), diff.std()))
plt.title('Regression quality - wt = 1 removed\nInputs = {aco_angle1, calculated phitt from regressed nu_wieghts}', fontsize = 'xx-large')
plt.xlabel('genphitt - regressedphi_tt', fontsize = 'x-large')
plt.grid()
plt.legend(prop={'size':15})
plt.savefig('regressed_nu_after_NN_fig3')
plt.close()



plt.subplot(2, 1, 1)
plt.title('Regression quality - wt = 1 removed - MSE and Cosineloss\nInputs = {aco_angle1, calculated phitt from regressed nu_wieghts}', fontsize = 'xx-large')
plt.hist2d(y_val, tf.transpose(model({"lab_frame": x_val}))[0], 50)
plt.plot([-90, 90], [-90, 90], 'k--', label = 'y=x')

plt.xlabel('genphitt')
plt.ylabel('phitt, regressed')
plt.legend()

plt.subplot(2, 1, 2)
plt.hist2d(y_val, 2*tf.transpose(model({"lab_frame": x_val}))[0], 50)
plt.plot([-90, 90], [-90, 90], 'k--', label = 'y=x')

plt.xlabel('genphitt')
plt.ylabel('2 * phitt, regressed')
plt.legend()
plt.savefig('regressed_nu_after_NN_fig4')
plt.close()























################ And now generic neutrinos !!!! ###############################







#Scenario 1: generic neutrinos, with weights std and mean as per tAUsPINNER

m_reg_nu_0 = 0.17
m_reg_nu_0p25 = 0.20
m_reg_nu_0p5 = 0.19

std_reg_nu_0 = 0.78
std_reg_nu_0p25 = 0.79
std_reg_nu_0p5 = 0.78

wt_cp_sm_generic1 = []
wt_cp_mm_generic1 = []
wt_cp_ps_generic1 = []

for i in range(len(df4['wt_cp_sm'])):
    wt_cp_sm_generic1.append(np.array(df4['wt_cp_sm'])[i]+np.random.normal(m_reg_nu_0, std_reg_nu_0))
    wt_cp_mm_generic1.append(np.array(df4['wt_cp_mm'])[i]+np.random.normal(m_reg_nu_0p25, std_reg_nu_0p25))
    wt_cp_ps_generic1.append(np.array(df4['wt_cp_ps'])[i]+np.random.normal(m_reg_nu_0p5, std_reg_nu_0p5))

theta = -np.pi/2
reco_phitt_generic = []
increment = 0.005

#len(df4['gen_phitt'])
for i in range(len(df4['gen_phitt'])):
    sm = wt_cp_sm_generic1[i] 
    mm = wt_cp_mm_generic1[i]
    ps = wt_cp_ps_generic1[i]
    weight_max = 0
    theta = -np.pi/2
    print(i/len(df4['gen_phitt']))
    while theta <= np.pi/2:
        event_weight = np.cos(theta)*np.cos(theta)*sm + np.sin(theta)*np.sin(theta)*ps + 2*np.cos(theta)*np.sin(theta)*(mm-sm/2-ps/2);
        
        
        if event_weight >= weight_max:
            #print('Old best weight: ', weight_max, 'new  best weight = ', event_weight, '\n')
            #print('Target: ', np.array(df4['gen_phitt'])[i], 'guess: ', (theta*180)/np.pi, 'i: ', i, '\n')
            weight_max = event_weight
            theta_save = theta
        theta = theta + increment
        
    reco_phitt_generic.append((180*theta_save)/np.pi)
    
print(reco_phitt_generic[:100], '\n')
print(df4['gen_phitt'][:100], '\n')


plt.hist(wt_cp_sm_generic1 - np.array(df4['wt_cp_sm']), bins = 100)
plt.savefig('generic_nu_fig-1')
plt.close()


plt.title('Phitt calculated from weights found\nwith generic neutrinos - 0.005', fontsize = 'xx-large')
plt.plot(df4['gen_phitt'][:len(reco_phitt_generic)], np.array(reco_phitt_generic), 'x')
plt.ylabel('reco_phitt(weights from generic nus)', fontsize = 'x-large')
plt.xlabel('gen phitt', fontsize = 'x-large')
plt.plot([-90, 90], [-90, 90], 'k--', label = 'x = y')
plt.grid()
plt.legend(prop = {'size': 16})
plt.savefig('generic_nu_fig0')
plt.close()

plt.hist2d(df4['gen_phitt'][:len(reco_phitt_generic)], np.array(reco_phitt_generic), 50)
plt.plot([-90, 90], [-90, 90], 'k--', label = 'x=y')
plt.xlabel('Gen_phitt', fontsize = 'x-large')
plt.ylabel('phitt from wts found w/generic nus', fontsize = 'x-large')
plt.title('Phitt calculated from weights found\nwith generic neutrinos - 0.005', fontsize = 'xx-large')
plt.legend(prop = {'size': 16}, loc= 6)
plt.savefig('generic_nu_fig1')
plt.close()


hist = np.array(np.cos(df4['gen_phitt'][:len(reco_phitt_generic)]) - np.cos(reco_phitt_generic))
plt.title('Phitt calculated with generic weights\ncorresponding to generic neutrinos - 0.005', fontsize = 'xx-large')
plt.hist(hist, bins = 50, label = 'cos(gen_phi_tt) - cos(reco_phitt)\nMean difference: %.2f, diff std: %.3f' %(hist.mean(), hist.std()))
plt.legend(prop = {'size': 16})
plt.xlabel('Cosine difference', fontsize = 'x-large')
plt.grid()
plt.savefig('generic_nu_fig2')
plt.close()

plt.title('Phitt calculated with generic weights\ncorresponding to generic neutrinos - 0.005', fontsize = 'xx-large')
hist = np.array(df4['gen_phitt'][:len(reco_phitt_generic)] - reco_phitt_generic)
plt.hist(hist, bins = 1000, label = 'gen_phi_tt - reco_phitt\nMean difference: %.2f, diff std: %.3f' %(hist.mean(), hist.std()))
plt.legend(prop = {'size': 16})
plt.xlabel('genphitt difference', fontsize = 'x-large')
plt.grid()
plt.savefig('generic_nu_fig3')
plt.close()

################# Nowwww - NN improvement ?     ###########
y = tf.convert_to_tensor(df4['gen_phitt'], dtype = 'float32')
        
x = [tf.convert_to_tensor(df4['aco_angle_1'], dtype = 'float32'),
     tf.convert_to_tensor(reco_phitt_generic, dtype = 'float32'),
     #tf.convert_to_tensor(df4['wt_cp_sm'], dtype = 'float32'),
     #tf.convert_to_tensor(df4['wt_cp_ps'], dtype = 'float32'),
    ]

x = tf.transpose(x)

#2) set-up the model
trainFrac = 0.7
nodes1 = 300
nodes2 = 300
drop = 0.1
E_size = 25
B_size = 500

numTrain = int(trainFrac*x.shape[0])

print(numTrain)
x_train = x[:numTrain]
y_train = y[:numTrain]

x_val = x[numTrain:]
y_val = y[numTrain:]

input_1 = tf.keras.Input(shape = x.shape, name="lab_frame")
x2 = tf.keras.layers.Dense(nodes1, activation = 'relu', name="learning")(input_1)
x3 = tf.keras.layers.Dense(nodes2, activation = 'relu', name="learning2")(x2)
x4 = tf.keras.layers.Dropout(drop, name="dropout2")(x3)
output = tf.keras.layers.Dense(1, name="output")(x4)

model = tf.keras.Model(
    inputs=[input_1],
    outputs=[output],
)

model.summary()

def loss_cosines(y_true, y_pred):
    delta = 40*(tf.cos(y_true) - tf.cos(y_pred))**2 + (y_true - y_pred) ** 2
    return tf.convert_to_tensor(delta) #tone it up

#tf.keras.losses.CosineSimilarity()
#loss_cosines
# tf.keras.losses.MeanSquaredError() +
model.compile(loss = loss_cosines, optimizer = 'adam', metrics = ['mae'])#, loss_mass_Higgs, loss_mass_tau, loss_D_p])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
    epochs=E_size,
    batch_size = B_size)

x = 0
y = 0

diff = np.array(y_val-2*tf.transpose(model({"lab_frame": x_val}))[0])
print(diff.mean(), diff.std())



plt.hist(diff, bins = 1000, label='genphitt - 2generic\nmean = %.2f, std  = %.2f'  %(diff.mean(), diff.std()))
plt.title('Regression quality - wt = 1 removed\nInputs = {aco_angle1, calculated phitt from generic nu_wieghts}', fontsize = 'xx-large')
plt.xlabel('genphitt - 2genericphi_tt', fontsize = 'x-large')
plt.grid()
plt.legend(prop={'size':15})
plt.savefig('generic_nu_after_NN_fig1')
plt.close()

diff = np.array(y_val-tf.transpose(model({"lab_frame": x_val}))[0])
print(diff.mean(), diff.std())

plt.hist(diff, bins = 1000, label='genphitt - generic\nmean = %.2f, std  = %.2f'  %(diff.mean(), diff.std()))
plt.title('Regression quality - wt = 1 removed - MSE and Cosineloss\nInputs = {aco_angle1, calculated phitt from generic nu_wieghts}', fontsize = 'xx-large')
plt.xlabel('genphitt - genericphi_tt', fontsize = 'x-large')
plt.grid()
plt.legend(prop={'size':15})
plt.savefig('generic_nu_after_NN_fig2')
plt.close()

diff = np.array(np.cos(y_val)-np.cos(2*tf.transpose(model({"lab_frame": x_val}))[0]))
print(diff.mean(), diff.std())
plt.hist(diff, bins = 1000, label='genphitt - generic\nmean = %.2f, std  = %.2f'  %(diff.mean(), diff.std()))
plt.title('Regression quality - wt = 1 removed\nInputs = {aco_angle1, calculated phitt from generic nu_wieghts}', fontsize = 'xx-large')
plt.xlabel('genphitt - genericphi_tt', fontsize = 'x-large')
plt.grid()
plt.legend(prop={'size':15})
plt.savefig('generic_nu_after_NN_fig3')
plt.close()



plt.subplot(2, 1, 1)
plt.title('Regression quality - wt = 1 removed - MSE and Cosineloss\nInputs = {aco_angle1, calculated phitt from generic nu_wieghts}', fontsize = 'xx-large')
plt.hist2d(y_val, tf.transpose(model({"lab_frame": x_val}))[0], 50)
plt.plot([-90, 90], [-90, 90], 'k--', label = 'y=x')

plt.xlabel('genphitt')
plt.ylabel('phitt, generic')
plt.legend()

plt.subplot(2, 1, 2)
plt.hist2d(y_val, 2*tf.transpose(model({"lab_frame": x_val}))[0], 50)
plt.plot([-90, 90], [-90, 90], 'k--', label = 'y=x')

plt.xlabel('genphitt')
plt.ylabel('2 * phitt, generic')
plt.legend()
plt.savefig('generic_nu_after_NN_fig4')
plt.close()

