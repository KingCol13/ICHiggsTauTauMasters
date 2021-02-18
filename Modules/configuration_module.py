#this is a python module hosting the different configurations that can be used in neutrino regression
import numpy as np
from pylorentz import Momentum4
from pylorentz import Vector4
from pylorentz import Position4
import polarimetric_module as polari
import alpha_module as am
import basic_functions as bf
import tensorflow as tf


class Configurations:

    def __init__(self, df4, decay_mode1, decay_mode2):
    
        tau_1_vis, tau_2_vis = bf.get_vis(df4, decay_mode1, decay_mode2)
        
        self.tau_1_vis = tau_1_vis
        self.tau_2_vis = tau_2_vis
        
        alpha_1, alpha_2 = am.alphas_clamped(df4, decay_mode1, decay_mode2)
        alphas = [alpha_1, alpha_2]
        
        nu1_guess, nu2_guess = polari.polarimetric(df4, decay_mode1, decay_mode2)

        nu1_guess = [nu1_guess.e, nu1_guess.p_x, nu1_guess.p_y, nu1_guess.p_z]
        nu2_guess = [nu2_guess.e, nu2_guess.p_x, nu2_guess.p_y, nu2_guess.p_z]
        
        
        sv_cov = [df4["svcov00_1"], df4["svcov01_1"],df4["svcov02_1"],df4["svcov10_1"], 
                df4["svcov11_1"], df4["svcov12_1"], df4["svcov20_1"], df4["svcov21_1"], 
                df4["svcov22_1"], df4["svcov00_2"], df4["svcov01_2"], df4["svcov02_2"],
                df4["svcov10_2"], df4["svcov11_2"], df4["svcov12_2"], df4["svcov20_2"],
                df4["svcov21_2"], df4["svcov22_2"],]

        ip_cov = [df4["ipcov00_1"], df4["ipcov01_1"],df4["ipcov02_1"],df4["ipcov10_1"], 
                df4["ipcov11_1"], df4["ipcov12_1"], df4["ipcov20_1"], df4["ipcov21_1"], 
                df4["ipcov22_1"],df4["ipcov00_2"],df4["ipcov01_2"], 
                df4["ipcov02_2"],df4["ipcov10_2"],df4["ipcov11_2"],df4["ipcov12_2"],
                df4["ipcov20_2"],df4["ipcov21_2"], df4["ipcov22_2"],]


        met_cov = [df4["metcov00"], df4["metcov01"],df4["metcov10"], df4["metcov11"] ]

            
        x1 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    #df4["met"],                #1
                    #df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,              
                    ])
        self.x1 = tf.transpose(x1)

        x2 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], #df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], #df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    #df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    
                    
                    ])
        self.x2 = tf.transpose(x2)

        x3 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,              
                    ])
        self.x3 = tf.transpose(x3)

        
        x4 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    #df4["met"],                #1
                    #df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    
                    
                    ])
        self.x4 = tf.transpose(x4)
        
        
        x5 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,              
                    ])
        self.x5 = tf.transpose(x5)


        x6 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,              
                    ])
        self.x6 = tf.transpose(x6)

        x7 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    #df4["met"],                #1
                    #df4["metx"],df4["mety"],   #2,3
                    df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    
                    
                    ])
        self.x7 = tf.transpose(x7)


        x8 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,              
                    ])
        self.x8 = tf.transpose(x8)
        
        
        x9 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    *sv_cov,
                    *ip_cov,
                    *met_cov,
                    
                    
                    ])
        self.x9 = tf.transpose(x9)


        x10 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    #df4["met"],                #1
                    #df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    
                    
                    ])
        self.x10 = tf.transpose(x10)


        x11 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,    
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    #*lowest_m_guess_1, *lowest_m_guess_2 #15
                    ])
        self.x11 = tf.transpose(x11)
        
                
        x12 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], #df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], #df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    #df4["met"],                #1
                    #df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    *ip_cov,
                    #*met_cov,
                    
                    
                    ])
        self.x12 = tf.transpose(x12)



        x13 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    #df4["met"],                #1
                    #df4["metx"],df4["mety"],   #2,3
                    df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    *sv_cov,
                    #*ip_cov,
                    #*met_cov,              
                    ])
        self.x13 = tf.transpose(x13)


        x14 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    *met_cov,
                    
                    
                    ])
        self.x14 = tf.transpose(x14)

        x15 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    #*nu1_guess, *nu2_guess #15
                    *alphas
                    
                    ])
        self.x15 = tf.transpose(x15)

        x16 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    #*nu1_guess, *nu2_guess #15
                    *alphas
                    
                    ])
        self.x16 = tf.transpose(x16)

        x17 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    *nu1_guess, *nu2_guess #15
                    
                    
                    ])
        self.x17 = tf.transpose(x17)
        
        x18 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    *nu1_guess, *nu2_guess, #15
                    *alphas
                    
                    ])
        self.x18 = tf.transpose(x18)

        x19 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    *nu1_guess, *nu2_guess #15
                    
                    
                    
                    ])
        self.x19 = tf.transpose(x19)
        
        x20 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    #*nu1_guess, *nu2_guess #15
                    *alphas
                    
                    ])
        self.x20 = tf.transpose(x20)
        
        
        x21 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    *ip_cov,
                    *met_cov,
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    #*nu1_guess, *nu2_guess #15
                    *alphas
                    
                    ])
        self.x21 = tf.transpose(x21)

        x22 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    *nu1_guess, *nu2_guess, #15
                    *alphas
                    
                    ])
        self.x22 = tf.transpose(x22)

        x23 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    *ip_cov,
                    *met_cov,
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    *nu1_guess, *nu2_guess, #15
                    *alphas
                    
                    ])
        self.x23 = tf.transpose(x23)
        
        
        
        x24 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov,
                    #*guess1_1, *guess2_1 #13
                    #*guess1_2, *guess2_2 #14
                    *nu1_guess, *nu2_guess #15
                    
                    
                    
                    ])

        self.x24 = np.transpose(x24)

                
        x25 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    #*met_cov, 
                    *alphas
                    ])
        self.x25 = tf.transpose(x3)
                
                
            

        x26 = np.array([
                    #smear_px,
                    #smear_py,
                    bf.Mom4_to_tf(tau_1_vis.e),    #4
                    bf.Mom4_to_tf(tau_1_vis.p_x),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_y),  #4
                    bf.Mom4_to_tf(tau_1_vis.p_z),  #4
                    bf.Mom4_to_tf(tau_2_vis.e),    #5
                    bf.Mom4_to_tf(tau_2_vis.p_x),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_y),  #5
                    bf.Mom4_to_tf(tau_2_vis.p_z),  #5
                    #df4["ip_x_1"], df4["ip_y_1"], df4["ip_z_1"], #6
                    #df4["ip_x_2"], df4["ip_y_2"], df4["ip_z_2"], #7
                    #df4["ip_sig_2"], df4["ip_sig_1"], #8,9
                    df4["met"],                #1
                    df4["metx"],df4["mety"],   #2,3
                    #df4["sv_x_1"], df4["sv_y_1"], df4["sv_z_1"], #10
                    #df4["sv_x_2"], df4["sv_y_2"], df4["sv_z_2"], #11
                    #*sv_cov,
                    #*ip_cov,
                    *met_cov,     
                    *alphas
                    ])
        self.x26 = tf.transpose(x26)
