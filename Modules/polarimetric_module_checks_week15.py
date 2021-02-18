# This module helps calculating the best polarimetric approximation for neutrinos for given decay mode

import numpy as np
import pandas as pd
import tensorflow as tf
from pylorentz import Momentum4
import basic_functions as bf



def polarimetric(df, decay_mode1, decay_mode2):
    nu_1 = Momentum4.m_eta_phi_p(np.zeros(len(df["gen_nu_phi_1"])), df["gen_nu_eta_1"], df["gen_nu_phi_1"], df["gen_nu_p_1"])
    nu_2 = Momentum4.m_eta_phi_p(np.zeros(len(df["gen_nu_phi_2"])), df["gen_nu_eta_2"], df["gen_nu_phi_2"], df["gen_nu_p_2"])
    
    m_tau = 1.77686
    
    tau_1_vis, tau_2_vis = bf.get_vis(df, decay_mode1, decay_mode2) 
    
    
    #NEUTRINO 1
    #2 caluclate their mass -  we will only check for a1s for now
    m_vis = tau_1_vis.m
    m_vis = np.where(tau_1_vis.m==0, 1.260, tau_1_vis.m)
    #3 caluclate directions
    norm_sv_1 = bf.norm([df['sv_x_1'], df['sv_y_1'], df['sv_z_1']])
    #remove the 0 lenght data otherwise we have nans
    norm_sv_1 = np.where(norm_sv_1 == 0, 9999, norm_sv_1)
    
    
    dir_x_tau1 = df['sv_x_1']/norm_sv_1
    dir_y_tau1 = df['sv_y_1']/norm_sv_1
    dir_z_tau1 = df['sv_z_1']/norm_sv_1
    tau1_dir = [dir_x_tau1, dir_y_tau1, dir_z_tau1]
    vis_1_dir = [tau_1_vis.p_x, tau_1_vis.p_y, tau_1_vis.p_z]
    
    
    norm_vis_1 = np.where(bf.norm(vis_1_dir) == 0, 9999, bf.norm(vis_1_dir))
    #4 calculate angle between tau and visible decay product direction
    #here changed to sin because it wasn't right - heck if it ends up better
    theta_GJ = np.arccos(np.clip(bf.dot_product(tau1_dir, vis_1_dir/norm_vis_1), -1, 1))

    theta_GJ_max = np.arcsin(np.clip((m_tau**2 - m_vis**2)/(2*m_tau*tau_1_vis.p), -1, 1))
    theta_GJ = tf.where(theta_GJ >= theta_GJ_max, theta_GJ_max, theta_GJ)
    
    #5) caluclate the tau momentum 
    minus_b = (m_vis**2 + m_tau**2) * tau_1_vis.p * np.cos(theta_GJ)
    two_a = 2*(m_vis**2 + tau_1_vis.p2 * (np.sin(theta_GJ))**2)
    b_squared_m_four_ac = (m_vis**2 + tau_1_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_1_vis.p2*(np.sin(theta_GJ))**2)
    b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)
    
    #6) have the two solutions
    sol1_1 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
    sol2_1 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a
    tau1_sol_1 = Momentum4(np.sqrt(sol1_1**2+m_tau**2), dir_x_tau1*sol1_1, dir_y_tau1*sol1_1, dir_z_tau1*sol1_1)
    tau1_sol_2 = Momentum4(np.sqrt(sol2_1**2+m_tau**2), dir_x_tau1*sol2_1, dir_y_tau1*sol2_1, dir_z_tau1*sol2_1) 
    
    nu_1_guess1 = tau1_sol_1 - tau_1_vis
    nu_1_guess2 = tau1_sol_2 - tau_1_vis
    
    #NEUTRINO2 - method is identical 
    m_vis = tau_2_vis.m
    m_vis = np.where(tau_2_vis.m==0, 1.260, tau_2_vis.m)
    norm_sv_2 = bf.norm([df['sv_x_2'], df['sv_y_2'], df['sv_z_2']])
    #remove the 0 lenght data otherwise we have nans
    norm_sv_2 = np.where(norm_sv_2 == 0, 9999, norm_sv_2)
    
    
    
    dir_x_tau2 = df['sv_x_2']/norm_sv_2
    dir_y_tau2 = df['sv_y_2']/norm_sv_2
    dir_z_tau2 = df['sv_z_2']/norm_sv_2
    tau2_dir = [dir_x_tau2, dir_y_tau2, dir_z_tau2]
    vis_2_dir = [tau_2_vis.p_x, tau_2_vis.p_y, tau_2_vis.p_z]
    
    norm_vis_2 = np.where(bf.norm(vis_2_dir) == 0, 9999, bf.norm(vis_2_dir))
    
    cos_theta_GJ =np.clip(bf.dot_product(tau2_dir, vis_2_dir/norm_vis_2), -1, 1)
    theta_GJ = np.arccos(cos_theta_GJ)
    
    
    theta_GJ_max = np.arcsin(np.clip((m_tau**2 - m_vis**2)/(2*m_tau*tau_2_vis.p), -1, 1))
    theta_GJ = tf.where(theta_GJ >= theta_GJ_max, theta_GJ_max, theta_GJ)
    
    minus_b = (m_vis**2 + m_tau**2) * tau_2_vis.p * np.cos(theta_GJ)
    two_a = 2*(m_vis**2 + tau_2_vis.p2 * (np.sin(theta_GJ))**2)
    b_squared_m_four_ac = (m_vis**2 + tau_2_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_2_vis.p2*(np.sin(theta_GJ))**2)
    b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)

    sol1_2 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
    sol2_2 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a     
    
    tau2_sol_1 = Momentum4(np.sqrt(sol1_2**2+m_tau**2), dir_x_tau2*sol1_2, dir_y_tau2*sol1_2, dir_z_tau2*sol1_2)
    tau2_sol_2 = Momentum4(np.sqrt(sol2_2**2+m_tau**2), dir_x_tau2*sol2_2, dir_y_tau2*sol2_2, dir_z_tau2*sol2_2) 
    nu_2_guess1 = tau2_sol_1 - tau_2_vis
    nu_2_guess2 = tau2_sol_2 - tau_2_vis
    
    #And finally selecting the combinaison giving closest higgs mass
    #1) There are 4 possible combinaisons of solutions
    Higgs_1_1 = tau1_sol_1 + tau2_sol_1
    Higgs_1_2 = tau1_sol_1 + tau2_sol_2
    Higgs_2_1 = tau1_sol_2 + tau2_sol_1
    Higgs_2_2 = tau1_sol_2 + tau2_sol_2
    
    #2) check the best Higgs mass two sols by two sols
    m_Higgs_target = tf.constant(125.10, shape = Higgs_1_1.m.shape, dtype = np.float32)
    higgs_mass_check_1 = np.where(abs(Higgs_1_1.m - m_Higgs_target)<=abs(Higgs_2_1.m-m_Higgs_target), Higgs_1_1, Higgs_2_1)
    #this check only depends on the first tau
    nu1_b1 = np.where(abs(Higgs_1_1.m - m_Higgs_target)<=abs(Higgs_2_1.m-m_Higgs_target), nu_1_guess1, nu_1_guess2)
    higgs_mass_check_1 = Momentum4(higgs_mass_check_1[0], higgs_mass_check_1[1], higgs_mass_check_1[2], higgs_mass_check_1[3])
    #2) check the best Higgs mass two sols by two sols
    higgs_mass_check_2 = np.where(abs(Higgs_2_2.m - m_Higgs_target)<=abs(Higgs_1_2.m-m_Higgs_target), Higgs_2_2, Higgs_1_2)
    higgs_mass_check_2 = Momentum4(higgs_mass_check_2[0], higgs_mass_check_2[1], higgs_mass_check_2[2], higgs_mass_check_2[3])
    nu2_b2 = np.where(abs(Higgs_2_2.m - m_Higgs_target)<=abs(Higgs_1_2.m-m_Higgs_target), nu_2_guess2, nu_2_guess1)
    #3) Then taking the best out of the best twos
    best_higgs = np.where(abs(higgs_mass_check_1.m - m_Higgs_target)<=abs(higgs_mass_check_2.m-m_Higgs_target), higgs_mass_check_1, higgs_mass_check_2)
    best_higgs = Momentum4(best_higgs[0], best_higgs[1], best_higgs[2], best_higgs[3])
    
    #And saving the associated neutrinos
    best_guess_nu1 = np.where(abs(higgs_mass_check_1.m - m_Higgs_target)<=abs(higgs_mass_check_2.m-m_Higgs_target), nu1_b1, nu_1_guess2)
    best_guess_nu2 = np.where(abs(higgs_mass_check_1.m - m_Higgs_target)<=abs(higgs_mass_check_2.m-m_Higgs_target), nu_2_guess1, nu2_b2)
    best_guess_nu1 = Momentum4(best_guess_nu1[0], best_guess_nu1[1], best_guess_nu1[2], best_guess_nu1[3])
    best_guess_nu2 = Momentum4(best_guess_nu2[0], best_guess_nu2[1], best_guess_nu2[2], best_guess_nu2[3])
    
    #we return the two neutrinos 4 vectors
    return nu_1_guess1, nu_1_guess2, nu_2_guess1, nu_2_guess2, best_guess_nu1, best_guess_nu2 





def gen_polarimetric(df, decay_mode1, decay_mode2):
    nu_1 = Momentum4.m_eta_phi_p(np.zeros(len(df["gen_nu_phi_1"])), df["gen_nu_eta_1"], df["gen_nu_phi_1"], df["gen_nu_p_1"])
    nu_2 = Momentum4.m_eta_phi_p(np.zeros(len(df["gen_nu_phi_2"])), df["gen_nu_eta_2"], df["gen_nu_phi_2"], df["gen_nu_p_2"])
    
    m_tau = 1.77686
    
    tau_1_vis = Momentum4.e_eta_phi_p(df["gen_vis_E_1"],df["gen_vis_eta_1"],df["gen_vis_phi_1"],df["gen_vis_p_1"]) 
    
    tau_2_vis = Momentum4.e_eta_phi_p(df["gen_vis_E_2"],df["gen_vis_eta_2"],df["gen_vis_phi_2"],df["gen_vis_p_2"])
    
    
    #NEUTRINO 1
    #2 caluclate their mass -  we will only check for a1s for now
    m_vis = tau_1_vis.m
    m_vis = np.where(tau_1_vis.m==0, 1.260, tau_1_vis.m)
    #3 caluclate directions
    norm_sv_1 = bf.norm([df['sv_x_1'], df['sv_y_1'], df['sv_z_1']])
    #remove the 0 lenght data otherwise we have nans
    norm_sv_1 = np.where(norm_sv_1 == 0, 9999, norm_sv_1)
    
    
    dir_x_tau1 = df['sv_x_1']/norm_sv_1
    dir_y_tau1 = df['sv_y_1']/norm_sv_1
    dir_z_tau1 = df['sv_z_1']/norm_sv_1
    tau1_dir = [dir_x_tau1, dir_y_tau1, dir_z_tau1]
    vis_1_dir = [tau_1_vis.p_x, tau_1_vis.p_y, tau_1_vis.p_z]
    
    
    norm_vis_1 = np.where(bf.norm(vis_1_dir) == 0, 9999, bf.norm(vis_1_dir))
    #4 calculate angle between tau and visible decay product direction
    theta_GJ = np.arccos(np.clip(bf.dot_product(tau1_dir, vis_1_dir/norm_vis_1), -1, 1))

    theta_GJ_max = np.arcsin(np.clip((m_tau**2 - m_vis**2)/(2*m_tau*tau_1_vis.p), -1, 1))
    theta_GJ = tf.where(theta_GJ >= theta_GJ_max, theta_GJ_max, theta_GJ)
    
    #5) caluclate the tau momentum 
    minus_b = (m_vis**2 + m_tau**2) * tau_1_vis.p * np.cos(theta_GJ)
    two_a = 2*(m_vis**2 + tau_1_vis.p2 * (np.sin(theta_GJ))**2)
    b_squared_m_four_ac = (m_vis**2 + tau_1_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_1_vis.p2*(np.sin(theta_GJ))**2)
    b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)
    
    #6) have the two solutions
    sol1_1 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
    sol2_1 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a
    tau1_sol_1 = Momentum4(np.sqrt(sol1_1**2+m_tau**2), dir_x_tau1*sol1_1, dir_y_tau1*sol1_1, dir_z_tau1*sol1_1)
    tau1_sol_2 = Momentum4(np.sqrt(sol2_1**2+m_tau**2), dir_x_tau1*sol2_1, dir_y_tau1*sol2_1, dir_z_tau1*sol2_1) 
    nu_1_guess1 = tau1_sol_1 - tau_1_vis
    nu_1_guess2 = tau1_sol_2 - tau_1_vis
    
    #NEUTRINO2 - method is identical 
    m_vis = tau_2_vis.m
    m_vis = np.where(tau_2_vis.m==0, 1.260, tau_2_vis.m)
    norm_sv_2 = bf.norm([df['sv_x_2'], df['sv_y_2'], df['sv_z_2']])
    #remove the 0 lenght data otherwise we have nans
    norm_sv_2 = np.where(norm_sv_2 == 0, 9999, norm_sv_2)
    
    
    
    dir_x_tau2 = df['sv_x_2']/norm_sv_2
    dir_y_tau2 = df['sv_y_2']/norm_sv_2
    dir_z_tau2 = df['sv_z_2']/norm_sv_2
    tau2_dir = [dir_x_tau2, dir_y_tau2, dir_z_tau2]
    vis_2_dir = [tau_2_vis.p_x, tau_2_vis.p_y, tau_2_vis.p_z]
    
    norm_vis_2 = np.where(bf.norm(vis_2_dir) == 0, 9999, bf.norm(vis_2_dir))
    
    cos_theta_GJ =np.clip(bf.dot_product(tau2_dir, vis_2_dir/norm_vis_2), -1, 1)
    theta_GJ = np.arccos(cos_theta_GJ)
    
    
    theta_GJ_max = np.arcsin(np.clip((m_tau**2 - m_vis**2)/(2*m_tau*tau_2_vis.p), -1, 1))
    theta_GJ = tf.where(theta_GJ >= theta_GJ_max, theta_GJ_max, theta_GJ)
    
    minus_b = (m_vis**2 + m_tau**2) * tau_2_vis.p * np.cos(theta_GJ)
    two_a = 2*(m_vis**2 + tau_2_vis.p2 * (np.sin(theta_GJ))**2)
    b_squared_m_four_ac = (m_vis**2 + tau_2_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_2_vis.p2*(np.sin(theta_GJ))**2)
    b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)

    sol1_2 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
    sol2_2 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a     
    
    tau2_sol_1 = Momentum4(np.sqrt(sol1_2**2+m_tau**2), dir_x_tau2*sol1_2, dir_y_tau2*sol1_2, dir_z_tau2*sol1_2)
    tau2_sol_2 = Momentum4(np.sqrt(sol2_2**2+m_tau**2), dir_x_tau2*sol2_2, dir_y_tau2*sol2_2, dir_z_tau2*sol2_2) 
    nu_2_guess1 = tau2_sol_1 - tau_2_vis
    nu_2_guess2 = tau2_sol_2 - tau_2_vis
    
    #And finally selecting the combinaison giving closest higgs mass
    #1) There are 4 possible combinaisons of solutions
    Higgs_1_1 = tau1_sol_1 + tau2_sol_1
    Higgs_1_2 = tau1_sol_1 + tau2_sol_2
    Higgs_2_1 = tau1_sol_2 + tau2_sol_1
    Higgs_2_2 = tau1_sol_2 + tau2_sol_2
    
    #2) check the best Higgs mass two sols by two sols
    m_Higgs_target = tf.constant(125.10, shape = Higgs_1_1.m.shape, dtype = np.float32)
    higgs_mass_check_1 = np.where(abs(Higgs_1_1.m - m_Higgs_target)<=abs(Higgs_2_1.m-m_Higgs_target), Higgs_1_1, Higgs_2_1)
    #this check only depends on the first tau
    nu1_b1 = np.where(abs(Higgs_1_1.m - m_Higgs_target)<=abs(Higgs_2_1.m-m_Higgs_target), nu_1_guess1, nu_1_guess2)
    higgs_mass_check_1 = Momentum4(higgs_mass_check_1[0], higgs_mass_check_1[1], higgs_mass_check_1[2], higgs_mass_check_1[3])
    #2) check the best Higgs mass two sols by two sols
    higgs_mass_check_2 = np.where(abs(Higgs_2_2.m - m_Higgs_target)<=abs(Higgs_1_2.m-m_Higgs_target), Higgs_2_2, Higgs_1_2)
    higgs_mass_check_2 = Momentum4(higgs_mass_check_2[0], higgs_mass_check_2[1], higgs_mass_check_2[2], higgs_mass_check_2[3])
    nu2_b2 = np.where(abs(Higgs_2_2.m - m_Higgs_target)<=abs(Higgs_1_2.m-m_Higgs_target), nu_2_guess2, nu_2_guess1)
    #3) Then taking the best out of the best twos
    best_higgs = np.where(abs(higgs_mass_check_1.m - m_Higgs_target)<=abs(higgs_mass_check_2.m-m_Higgs_target), higgs_mass_check_1, higgs_mass_check_2)
    best_higgs = Momentum4(best_higgs[0], best_higgs[1], best_higgs[2], best_higgs[3])
    
    #And saving the associated neutrinos
    best_guess_nu1 = np.where(abs(higgs_mass_check_1.m - m_Higgs_target)<=abs(higgs_mass_check_2.m-m_Higgs_target), nu1_b1, nu_1_guess2)
    best_guess_nu2 = np.where(abs(higgs_mass_check_1.m - m_Higgs_target)<=abs(higgs_mass_check_2.m-m_Higgs_target), nu_2_guess1, nu2_b2)
    best_guess_nu1 = Momentum4(best_guess_nu1[0], best_guess_nu1[1], best_guess_nu1[2], best_guess_nu1[3])
    best_guess_nu2 = Momentum4(best_guess_nu2[0], best_guess_nu2[1], best_guess_nu2[2], best_guess_nu2[3])
    
    #we return the two neutrinos 4 vectors
    return nu_1_guess1, nu_1_guess2, nu_2_guess1, nu_2_guess2, best_guess_nu1, best_guess_nu2 

        

