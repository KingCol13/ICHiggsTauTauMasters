# This is a small module to calculate the alpha values for tau-> a1 decays to later be used in the training 

import numpy as np
import pandas as pd
import tensorflow as tf
from pylorentz import Momentum4
import basic_functions as bf

def alphas(df, decay_mode1, decay_mode2):
    #A function to return the alpha parameters of the collinear approximation
    tau_1_vis, tau_2_vis = bf.get_vis(df, decay_mode1, decay_mode2) 
    
    
    alpha_2 = (df['mety']*tau_1_vis.p_x-df['metx']*tau_1_vis.p_y)/(tau_2_vis.p_y*tau_1_vis.p_x-tau_2_vis.p_x*tau_1_vis.p_y)

    alpha_1 = (df['metx']-alpha_2*tau_2_vis.p_x)/tau_1_vis.p_x
    
    return alpha_1, alpha_2



def gen_alphas(df, decay_mode1, decay_mode2):
    #A function to return the alpha parameters of the collinear approximation
    tau_1_vis = Momentum4.e_eta_phi_p(df["gen_vis_E_1"],df["gen_vis_eta_1"],df["gen_vis_phi_1"],df["gen_vis_p_1"]) 
    
    tau_2_vis = Momentum4.e_eta_phi_p(df["gen_vis_E_2"],df["gen_vis_eta_2"],df["gen_vis_phi_2"],df["gen_vis_p_2"])
    
    
    alpha_2 = (df['mety']*tau_1_vis.p_x-df['metx']*tau_1_vis.p_y)/(tau_2_vis.p_y*tau_1_vis.p_x-tau_2_vis.p_x*tau_1_vis.p_y)

    alpha_1 = (df['metx']-alpha_2*tau_2_vis.p_x)/tau_1_vis.p_x
    
    return alpha_1, alpha_2








def alphas_clamped(df, decay_mode1, decay_mode2):
    #A function to return the alpha parameters of the collinear approximation
    tau_1_vis, tau_2_vis = bf.get_vis(df, decay_mode1, decay_mode2)
    
    
    alpha_2 = (df['mety']*tau_1_vis.p_x-df['metx']*tau_1_vis.p_y)/(tau_2_vis.p_y*tau_1_vis.p_x-tau_2_vis.p_x*tau_1_vis.p_y)

    alpha_1 = (df['metx']-alpha_2*tau_2_vis.p_x)/tau_1_vis.p_x
    
    
    #bring back to 1 when we are higher than 1, neutrino cannot carry away more than the tau momentum
    alpha_1 = tf.where(abs(alpha_1)>=1, alpha_1/abs(alpha_1), alpha_1)
    
    alpha_2 = tf.where(abs(alpha_2)>=1, alpha_2/abs(alpha_2), alpha_2)
    
    
    
    return alpha_1, alpha_2
