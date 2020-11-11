#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 16:40:49 2020

@author: kingsley
"""

import numpy as np
from ROOT import TLorentzVector
from pylorentz import Momentum4

def Lprint(p : TLorentzVector):
    print(p[0], p[1], p[2], p[3])

#%% TLorentz

p1 = TLorentzVector(0, 0, 0, 1)
p2 = TLorentzVector(1, 2, 2, 11)
p3 = TLorentzVector(-1, -3, 0, 11)

print("Before boost:")
Lprint(p2)
Lprint(p3)

boost = (p3+p2).BoostVector()
p2.Boost(-boost)
p3.Boost(-boost)

print("After boost:")
Lprint(p2)
Lprint(p3)

#%% LorentzVector

from ROOT import Math

p1 = Math.PxPyPzEVector(0, 0, 0, 1)

#%% PyLorentz

p1 = Momentum4(1, 0, 0, 0)
p2 = Momentum4(11, 1, 2, 2)
p3 = Momentum4(11, -1, -3, 0)

print("Before boost:")
print(p2)
print(p3)

boostP = p2+p3
boost = Momentum4(boostP[0], -boostP[1], -boostP[2], -boostP[3])
p2 = p2.boost_particle(boost)
p3 = p3.boost_particle(boost)

print("After boost:")
print(p2)
print(p3)

#%% PyLorentz position test

from pylorentz import Position4

pos1 = Position4(0, 1, 0, 0)
beta = 0.5

#by hand:
gamma = 1/np.sqrt(1-beta*beta)
print(gamma*(pos1[0] - beta*pos1[1]), gamma*(pos1[1] - beta*pos1[0]))

#using library:
pos2 = pos1.boost(1, 0, 0, beta=beta)
print(pos2)

#using boost_particle, flip px
boost = Momentum4(gamma, -0.5*gamma, 0, 0)
pos3 = pos1.boost_particle(boost)
print(pos3)

#%% Conclusion
"""
Using pyLorentz boost_particle with -FourMomenta results in an incorrect boost
the resulting energies are negative and the 3 momentum components are also
incorrect
"""