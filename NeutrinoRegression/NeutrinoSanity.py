#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:25:01 2020

@author: kingsley
"""

#%% imports

import uproot 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pylorentz import Momentum4
from pylorentz import Position4

#%% Data loading
tree = uproot.open("MVAFILE_AllHiggs_tt.root")["ntuple"]

momenta_features = [ "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", #leading charged pi 4-momentum
              "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", #subleading charged pi 4-momentum
              "pi0_E_1","pi0_px_1","pi0_py_1","pi0_pz_1", #leading neutral pi 4-momentum
              "pi0_E_2","pi0_px_2","pi0_py_2","pi0_pz_2"] #subleading neutral pi 4-momentum
    
selectors = [ "tau_decay_mode_1","tau_decay_mode_2",
             "mva_dm_1","mva_dm_2"
             ]

neutrino_features = [  "gen_nu_p_1", "gen_nu_p_2",
                       "gen_nu_phi_1", "gen_nu_phi_2",
                       "gen_nu_eta_1", "gen_nu_eta_2"]

df = tree.pandas.df(momenta_features+selectors+neutrino_features)

df = df[
      (df["tau_decay_mode_1"] == 1) 
    & (df["tau_decay_mode_2"] == 1) 
    & (df["mva_dm_1"] == 1) 
    & (df["mva_dm_2"] == 1)
]

df = df[df["gen_nu_p_1"] > -4000]

print("Data loaded.")

#%% Create 4-momenta

"""
okay, that was a problem on my side I think, I can now access the ntuple all right. 
I have performed the same checks and it definitely seems more right. I guess the difference 
between the rho and neutrino momenta component is 0 in average because the neutrinos are colinear 
to the visible decay products (?) but since they are much (infinitely) less massive they must carry 
away more energy than the rho in average (is that correct ?). Next, if we have your green 
light on this I guess we can check that the dot product between the visible decay products 
and neutrinos is (very close to) 0, just to quantify the limits of the colinear approximation

Daniel: ok thanks yes i agree that this looks correct then. The first plot I am not 100% sure about,
I thought that the neutrinos tended to carry less energy than the visible components - when i made a
plot of tau E - nu E myself this is what it looks like (maybe your firts plot is doing the
opposite - nu E - tau E?). The other check you could do is to make plots of invariant mass
of the visible + neutrino 4-vectors which should peak close to the tau mass.
For the dot product I think we expect it to be large rather than close to 0
(0 would mean they are perpendicular), so for collinear neutrinos we expect the dot
product divided by the magnitudes of the 2 vectors to be close to 1
"""

#The other check you could do is to make plots of invariant mass
#of the visible + neutrino 4-vectors which should peak close to the tau mass.

# Create our 4-vectors in the lab frame
# Charged pions:
pi_1_lab = Momentum4(df["pi_E_1"], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
pi_2_lab = Momentum4(df["pi_E_2"], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])

#Neutral pions
pi0_1_lab = Momentum4(df["pi0_E_1"], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
pi0_2_lab = Momentum4(df["pi0_E_2"], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])

#Neutrinos
nu_1_lab = Momentum4.e_m_eta_phi(df["gen_nu_p_1"], 0, df["gen_nu_eta_1"], df["gen_nu_phi_1"])
nu_2_lab = Momentum4.e_m_eta_phi(df["gen_nu_p_2"], 0, df["gen_nu_eta_2"], df["gen_nu_phi_2"])

tau_1_lab = pi_1_lab + pi0_1_lab + nu_1_lab
tau_2_lab = pi_2_lab + pi0_2_lab + nu_2_lab

tau_1_masses = tau_1_lab.m
tau_2_masses = np.real(tau_2_lab.m)

tau_1_masses = tau_1_masses[tau_1_masses>0.5]
tau_2_masses = tau_2_masses[tau_2_masses>0.5]

higgs_lab = tau_1_lab + tau_2_lab
higgs_masses = np.real(higgs_lab.m)

#%% Tau mass histogram

print("Mean leading tau mass: ", np.mean(tau_1_masses))
print("Mean subleading tau mass: ", np.mean(tau_2_masses))

plt.figure()
plt.title("Mass of Visible Decay Products + Tau Neutrino")
plt.xlabel("Mass / GeV")
plt.ylabel("Frequency")
#plt.xlim(-5, 5)
plt.hist(tau_1_masses, bins = 100, alpha = 0.5, label="Leading visible products + neutrino masses")
plt.hist(tau_2_masses, bins = 100, alpha = 0.5, label="Subleading visible products + neutrino masses")
plt.grid()
plt.legend(loc="upper right")
plt.show()

#%% Higgs mass histogram

print("Mean Higgs mass: ", np.mean(higgs_masses))

higgs_masses = higgs_masses[higgs_masses>50]

plt.figure()
plt.title("Mass of Higgs")
plt.xlabel("Mass / GeV")
plt.ylabel("Frequency")
#plt.xlim(-5, 5)
plt.hist(higgs_masses, bins = 100, alpha = 0.5)
plt.grid()
plt.show()

#%% Dot product

def norm(x):
    return np.sqrt(x[1]*x[1]+x[2]*x[2]+x[3]*x[3])

def norm_dot_prod(x, y):
    return (x[1]*y[1]+x[2]*y[2]+x[3]*y[3])/(norm(x)*norm(y))

dot_prods_1 = norm_dot_prod(pi_1_lab, nu_1_lab)

plt.figure()
plt.title("Pi_1 dot nu_1")
plt.xlabel("Value")
plt.ylabel("Frequency")
#plt.xlim(-5, 5)
plt.hist(dot_prods_1, bins = 100, alpha = 0.5)
plt.grid()
plt.show()