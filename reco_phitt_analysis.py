#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:25:01 2021

@author: kingsley
"""

#%% Imports

import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cm

#%% Read data

tree = uproot.open("ROOTfiles/MVAFILE_AllHiggs_tt_pseudo_phitt.root")["ntuple"]

selectors = ['mva_dm_1', 'mva_dm_2', 'tau_decay_mode_1', 'tau_decay_mode_2']

variables = ["aco_angle_1", "gen_phitt", "pseudo_phitt" ] #"pseudo_phitt"]

variables2 = ["pseudo_wt_cp_sm", "pseudo_wt_cp_mm", "pseudo_wt_cp_ps"]

gen_weights = ["wt_cp_sm", "wt_cp_mm", "wt_cp_ps"]

df = tree.pandas.df(gen_weights+variables+variables2+selectors)

#%% Selection

#pi-pi
#df = df[(df['mva_dm_1'] == 0) & (df['mva_dm_2'] == 0)]

#rho-rho:
#df = df[(df['mva_dm_1'] == 1) & (df['mva_dm_2'] == 1) & (df['tau_decay_mode_1'] == 1) & (df['tau_decay_mode_2'] == 1)]

#a1-a1 (a1->pi+2pi0)
#df = df[(df['mva_dm_1'] == 2) & (df['mva_dm_2'] == 2) & (df['tau_decay_mode_1'] == 1) & (df['tau_decay_mode_2'] == 1)]

#a1-a1 (a1->3pi))
df = df[(df['mva_dm_1'] == 10) & (df['mva_dm_2'] == 10)]

#a1-a1 combined
#df = df[((df['mva_dm_1'] == 2) | (df['mva_dm_1'] == 10)) & ((df['mva_dm_2'] == 2) | (df['mva_dm_2'] == 10))]

#%% Cleanup

df = df.dropna(subset=['gen_phitt', 'pseudo_phitt'])
#TODO: remove this when new non-logged maxed data comes in
df = df[(df['pseudo_phitt']<-57.32) | (df['pseudo_phitt']>-57.28)]
#df = df[df['aco_angle_1'] > -400]

#%% Fix shift in data
gen_phitt = np.array(df['gen_phitt'])
aco_angle_1 = np.array(df['aco_angle_1'])
new_phitt = np.array(df['pseudo_phitt'])
new_phitt = np.where(new_phitt>90, new_phitt-180, new_phitt)

wt_cp_sm = np.array(df['wt_cp_sm'])
wt_cp_mm = np.array(df['wt_cp_mm'])
wt_cp_ps = np.array(df['wt_cp_ps'])

#%% Histograms

plt.figure()
plt.xlabel("gen_phitt")
plt.ylabel("pseudo_phitt")
plt.hist2d(gen_phitt, new_phitt, 50)
plt.grid()
plt.colorbar() 
plt.show()

#%% not useful histograms
"""
plt.figure()
plt.xlabel("gen_phitt")
plt.ylabel("aco_angle_1")
plt.hist2d(gen_phitt, aco_angle_1, 50)
plt.grid()
plt.show()

plt.figure()
plt.xlabel("pseudo_phitt")
plt.ylabel("aco_angle_1")
plt.hist2d(new_phitt, aco_angle_1, 50)
plt.grid()
plt.show()
"""

#%%

res = gen_phitt - new_phitt
x_range = np.linspace(-180, 180, 10000)
m = len(res)/360
y_triangle = np.where(x_range<0, m*(x_range/90+2), m*(2-x_range/90))

def f(x, nvals, nbins):
    """
    draw no intelligence histogram
    equivalent of subtracting 2 uniform distributions
    and plotting the histogram
    """
    x_bin = np.floor(nbins*(x+180)/360)
    x0 = (x_bin*360)/nbins - 180
    x1 = ((x_bin+1)*360)/nbins - 180
    return (nvals/360)*np.where(x<0, (x1*x1-x0*x0)/180 + 2*(x1-x0), (x0*x0-x1*x1)/180 + 2*(x1-x0))

plt.figure()
plt.xlabel("gen_phitt - pseudo_phitt")
plt.ylabel("Frequency")
#plt.xlim(-100, 100)
plt.hist(res, bins = 100, alpha = 1, label="gen_phitt-pseudo_phitt mean={:.2f}, std={:.2f}".format(np.mean(res), np.std(res)))
plt.plot(x_range, f(x_range, len(res), 100), label="Triangular Distribution")
plt.grid()
plt.legend(loc="upper right", frameon=False)
plt.show()

#%% 3d bars reborn in colour

x = gen_phitt
y = new_phitt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Get bins using numpy
hist, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[-90, 90], [-90, 90]])

spacing = xedges[1]-xedges[0]

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + spacing/2, yedges[:-1] + spacing/2, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = np.ones_like(zpos)*spacing
dz = hist.ravel()

offset = dz + np.abs(dz.min())
fracs = offset.astype(float)/offset.max()
norm = colors.Normalize(fracs.min(), fracs.max())
colourmap = cm.jet(norm(fracs.tolist()))

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=colourmap)
plt.xlabel("gen_phitt")
plt.ylabel("pseudo_phitt")
ax.set_zlabel("frequency")
plt.grid()
plt.show()

#%% Surface plot

x = gen_phitt
y = new_phitt

hist, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[-90, 90], [-90, 90]])
spacing = xedges[1]-xedges[0]
zpos = hist.ravel()
xpos, ypos = np.meshgrid(xedges[:-1] + spacing/2, yedges[:-1] + spacing/2, indexing="ij")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xpos, ypos, hist,cmap='viridis', edgecolor='none')
ax.set_title('Surface plot')
ax.set_xlabel("gen_phitt")
ax.set_ylabel("pseudo_phitt")
ax.set_zlabel("frequency")
plt.show()
