# This is a small module to calculate the alpha values for tau-> a1 decays to later be used in the training 

import numpy as np
import pandas as pd
import tensorflow as tf
from pylorentz import Momentum4
import basic_functions as bf

def alphas(df, decay_mode1, decay_mode2):
    #A function to return the alpha parameters of the collinear approximation
    if decay_mode1 == 10 and decay_mode2 == 10:
        print('\na1(3pr) - a1(3pr) Decay mode\n')
        #1) define the visible decay products
        pi_1_4Mom = Momentum4(df["pi_E_1"],df["pi_px_1"],df["pi_py_1"],df["pi_pz_1"])
        pi2_1_4Mom = Momentum4(df["pi2_E_1"],df["pi2_px_1"],df["pi2_py_1"],df["pi2_pz_1"])
        pi3_1_4Mom = Momentum4(df["pi3_E_1"],df["pi3_px_1"],df["pi3_py_1"],df["pi3_pz_1"])
        pi_2_4Mom = Momentum4(df["pi_E_2"],df["pi_px_2"],df["pi_py_2"],df["pi_pz_2"]) 
        pi2_2_4Mom = Momentum4(df["pi2_E_2"],df["pi2_px_2"],df["pi2_py_2"],df["pi2_pz_2"]) 
        pi3_2_4Mom = Momentum4(df["pi3_E_2"],df["pi3_px_2"],df["pi3_py_2"],df["pi3_pz_2"]) 
        tau_1_vis = pi_1_4Mom + pi2_1_4Mom + pi3_1_4Mom 
        tau_2_vis = pi_2_4Mom + pi2_2_4Mom + pi3_2_4Mom 
    
    
    alpha_2 = (df['mety']*tau_1_vis.p_x-df['metx']*tau_1_vis.p_y)/(tau_2_vis.p_y*tau_1_vis.p_x-tau_2_vis.p_x*tau_1_vis.p_y)

    alpha_1 = (df['metx']-alpha_2*tau_2_vis.p_x)/tau_1_vis.p_x
    
    
    #bring back to 1 when we are higher than 1, neutrino cannot carry away more than the tau momentum
    #alpha_1 = tf.where(abs(alpha_1)>=1, alpha_1/abs(alpha_1), alpha_1)
    
    #alpha_2 = tf.where(abs(alpha_2)>=1, alpha_2/abs(alpha_2), alpha_2)
    
    
    
    return alpha_1, alpha_2












def alphas_clamped(df, decay_mode1, decay_mode2):
    #A function to return the alpha parameters of the collinear approximation
    if decay_mode1 == 10 and decay_mode2 == 10:
        print('\na1(3pr) - a1(3pr) Decay mode\n')
        #1) define the visible decay products
        pi_1_4Mom = Momentum4(df["pi_E_1"],df["pi_px_1"],df["pi_py_1"],df["pi_pz_1"])
        pi2_1_4Mom = Momentum4(df["pi2_E_1"],df["pi2_px_1"],df["pi2_py_1"],df["pi2_pz_1"])
        pi3_1_4Mom = Momentum4(df["pi3_E_1"],df["pi3_px_1"],df["pi3_py_1"],df["pi3_pz_1"])
        pi_2_4Mom = Momentum4(df["pi_E_2"],df["pi_px_2"],df["pi_py_2"],df["pi_pz_2"]) 
        pi2_2_4Mom = Momentum4(df["pi2_E_2"],df["pi2_px_2"],df["pi2_py_2"],df["pi2_pz_2"]) 
        pi3_2_4Mom = Momentum4(df["pi3_E_2"],df["pi3_px_2"],df["pi3_py_2"],df["pi3_pz_2"]) 
        tau_1_vis = pi_1_4Mom + pi2_1_4Mom + pi3_1_4Mom 
        tau_2_vis = pi_2_4Mom + pi2_2_4Mom + pi3_2_4Mom 
    
    
    alpha_2 = (df['mety']*tau_1_vis.p_x-df['metx']*tau_1_vis.p_y)/(tau_2_vis.p_y*tau_1_vis.p_x-tau_2_vis.p_x*tau_1_vis.p_y)

    alpha_1 = (df['metx']-alpha_2*tau_2_vis.p_x)/tau_1_vis.p_x
    
    
    #bring back to 1 when we are higher than 1, neutrino cannot carry away more than the tau momentum
    alpha_1 = tf.where(abs(alpha_1)>=1, alpha_1/abs(alpha_1), alpha_1)
    
    alpha_2 = tf.where(abs(alpha_2)>=1, alpha_2/abs(alpha_2), alpha_2)
    
    
    
    return alpha_1, alpha_2
