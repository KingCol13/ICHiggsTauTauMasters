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

#tree = uproot.open("ROOTfiles/MVAFILE_AllHiggs_tt_pseudo_phitt.root")["ntuple"]
tree = uproot.open("ROOTfiles/MVAFILE_full_10_10_phitt.root")["ntuple"]

selectors = ['mva_dm_1', 'mva_dm_2', 'tau_decay_mode_1', 'tau_decay_mode_2']

variables = ["aco_angle_1", "gen_phitt" ] #"pseudo_phitt"]

TSlevels = ["pseudo", "pola", "pola2", "reco", "reco2"]
baseTSVars = ["_wt_cp_sm", "_wt_cp_mm", "_wt_cp_ps", "_phitt"]
TSVars = [x+y for x in TSlevels for y in baseTSVars]

gen_weights = ["wt_cp_sm", "wt_cp_mm", "wt_cp_ps"]

df = tree.pandas.df(gen_weights+variables+TSVars+selectors)

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

#%% Select level to use

level = "pola"

#%% Cleanup

df = df.dropna(subset=['gen_phitt']+TSVars)

#%% Fix shift in data
gen_phitt = np.array(df['gen_phitt'])
aco_angle_1 = np.array(df['aco_angle_1'])
new_phitt = np.array(df[level+'_phitt'])

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

diff_ps = np.array(df["wt_cp_ps"]-df[level+"gen_phitt"])
diff_sm = np.array(df["wt_cp_sm"]-df[level+"_wt_cp_sm"])
diff_mm = np.array(df["wt_cp_mm"]-df[level+"_wt_cp_mm"])

plt.figure()
plt.hist(diff_ps, bins = 50, alpha = 0.5, label = 'gen - '+level+' ps weights\nMean: %.2f, std:%.2f'%(diff_ps.mean(), diff_ps.std()))
plt.hist(diff_sm, bins = 50, alpha = 0.5, label = 'gen - '+level+' sm weights\nMean: %.2f, std:%.2f'%(diff_sm.mean(), diff_sm.std()))
plt.hist(diff_mm, bins = 50, alpha = 0.5, label = 'gen - '+level+' mm weights\nMean: %.2f, std:%.2f'%(diff_mm.mean(), diff_mm.std()))
plt.xlabel('gen-'+level+' weights')
plt.ylabel("Frequency")
plt.legend()
plt.grid()
plt.show()

#%% Discrimination histogram

for level in levels:
    print(level)
    plt.figure()
    plt.title("Binary Discrimination")
    plt.xlabel(level+" / degrees")
    plt.ylabel("Frequency")
    plt.hist(df[df['wt_cp_sm']>df['wt_cp_ps']][level+'_phitt'], 50, label="Even", alpha=0.5)
    plt.hist(df[df['wt_cp_sm']<df['wt_cp_ps']][level+'_phitt'], 50, label="Odd", alpha=0.5)
    plt.legend()
    plt.grid()
    plt.show()

#%% Confusion Matrix

def getNconfusion(df, true_class, pred_class):
    true_sm = df['wt_cp_sm'] > df['wt_cp_ps']
    dfc = df.copy()
    if true_class=='sm':
        dfc = dfc[true_sm]
    else:
        dfc = dfc[~true_sm]
    
    pred_sm = df[level+'_wt_cp_sm'] > df[level+'_wt_cp_ps']
    if pred_class == 'sm':
        dfc = dfc[pred_sm]
    else:
        dfc = dfc[~pred_sm]
    
    return len(dfc)/len(df)

for level in levels:
    #naming trueclass_predclass
    sm_sm = getNconfusion(df, 'sm', 'sm')
    sm_ps = getNconfusion(df, 'sm', 'ps')
    ps_sm = getNconfusion(df, 'ps', 'sm')
    ps_ps = getNconfusion(df, 'ps', 'ps')
    
    print(level)
    print("\t\tTrue SM\tTrue PS")
    print("Pred SM\t{:.4f}\t{:.4f}".format(sm_sm, ps_sm))
    print("Pred PS\t{:.4f}\t{:.4f}".format(sm_ps, ps_ps))

#%% Profile plot

res = gen_phitt - new_phitt
x_range = np.linspace(-180, 180, 10000)

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
plt.xlabel("gen_phitt - "+level+"_phitt")
plt.ylabel("Frequency")
#plt.xlim(-100, 100)
plt.hist(res, bins = 100, alpha = 1, label="gen_phitt-"+level+"_phitt mean={:.2f}, std={:.2f}".format(np.mean(res), np.std(res)))
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
plt.ylabel(level+"_phitt")
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
ax.set_xlabel("gen_phitt")
ax.set_ylabel(level+"_phitt")
ax.set_zlabel("frequency")
plt.show()
