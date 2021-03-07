# This module helps calculating the best polarimetric approximation for neutrinos for given decay mode

import numpy as np
import pandas as pd
import tensorflow as tf
from pylorentz import Momentum4
import basic_functions as bf



def polarimetric(df, decay_mode1, decay_mode2):
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
    
    print(sol1_2 [90:100])
    print(sol2_2 [90:100])
    
    #we return the two neutrinos 4 vectors
    return nu_1_guess1, nu_1_guess2, nu_2_guess1, nu_2_guess2, best_guess_nu1, best_guess_nu2 





def gen_polarimetric(df, decay_mode1, decay_mode2):
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

        

def polarimetric_theta_max(df, decay_mode1, decay_mode2):
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
    
    #if we end up reaching theta_max -> we need to improve the tau_direction
    updated_dir_tau1 = vis_1_dir/norm_vis_1*np.cos(theta_GJ) + vis_1_dir/norm_vis_1*np.sin(theta_GJ)
    dir_x_tau1 = updated_dir_tau1[0]
    dir_y_tau1 = updated_dir_tau1[1]
    dir_z_tau1 = updated_dir_tau1[2]
    
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
    
    cos_theta_GJ = np.clip(bf.dot_product(tau2_dir, vis_2_dir/norm_vis_2), -1, 1)
    theta_GJ = np.arccos(cos_theta_GJ)
    
    
    theta_GJ_max = np.arcsin(np.clip((m_tau**2 - m_vis**2)/(2*m_tau*tau_2_vis.p), -1, 1))
    theta_GJ = tf.where(theta_GJ >= theta_GJ_max, theta_GJ_max, theta_GJ)
    
    minus_b = (m_vis**2 + m_tau**2) * tau_2_vis.p * np.cos(theta_GJ)
    two_a = 2*(m_vis**2 + tau_2_vis.p2 * (np.sin(theta_GJ))**2)
    b_squared_m_four_ac = (m_vis**2 + tau_2_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_2_vis.p2*(np.sin(theta_GJ))**2)
    b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)

    sol1_2 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
    sol2_2 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a     
    
    #if we end up reaching theta_max -> we need to improve the tau_direction
    updated_dir_tau2 = vis_2_dir/norm_vis_2*np.cos(theta_GJ) + vis_2_dir/norm_vis_2*np.sin(theta_GJ)
    dir_x_tau2 = updated_dir_tau2[0]
    dir_y_tau2 = updated_dir_tau2[1]
    dir_z_tau2 = updated_dir_tau2[2]
    
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




def polarimetric_no_clamping(df, decay_mode1, decay_mode2):
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
    
    ##5) caluclate the tau momentum 
    #minus_b = (m_vis**2 + m_tau**2) * tau_1_vis.p * np.cos(theta_GJ)
    #two_a = 2*(m_vis**2 + tau_1_vis.p2 * (np.sin(theta_GJ))**2)
    #b_squared_m_four_ac = (m_vis**2 + tau_1_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_1_vis.p2*(np.sin(theta_GJ))**2)
    #b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)
    
    ##6) have the two solutions
    #sol1_1 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
    #sol2_1 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a
    
    
    #doing it the same way as in the CPP code - better separation?
    a = 4.0 * (m_vis**2 + tau_1_vis.p2 * (np.sin(theta_GJ))**2)
    b = -4.0 * (m_vis**2 + m_tau**2) * tau_1_vis.p * np.cos(theta_GJ)
    c = 4.0 * m_tau**2 * (m_vis**2 + tau_1_vis.p2) - (m_vis**2 + m_tau**2)**2
    
    D = np.array(np.clip(b*b - 4*a*c, 0.0, 10.0**100))    
    #in case any non physical angle has slipped through, shouldn't have
    
    #print(np.sqrt(D))
    #np.sign(b)*
    q = -0.5 * (b + np.sign(b) * np.sqrt(D))
    sol1_1 = c/q
    sol2_1 = q/a
    
    
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
    
    #minus_b = (m_vis**2 + m_tau**2) * tau_2_vis.p * np.cos(theta_GJ)
    #two_a = 2*(m_vis**2 + tau_2_vis.p2 * (np.sin(theta_GJ))**2)
    
    #b_squared_m_four_ac = (m_vis**2 + tau_2_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_2_vis.p2*(np.sin(theta_GJ))**2)
    
    #w = 0
    #for i in range(len(b_squared_m_four_ac)):
        #if b_squared_m_four_ac[i] < 0  and w<10:
            #print ('Issue %i', i)
            #w = w+1
    #b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)

    #sol1_2 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
    #sol2_2 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a    
    
    
    #doing it the same way as in the CPP code - better separation ?
    a = 4.0 * (m_vis**2 + tau_1_vis.p2 * (np.sin(theta_GJ))**2)
    b = -4.0 * (m_vis**2 + m_tau**2) * tau_1_vis.p * np.cos(theta_GJ)
    c = 4.0 * m_tau**2 * (m_vis**2 + tau_1_vis.p2) - (m_vis**2 + m_tau**2)**2
    
    D = np.array(np.clip(b*b - 4*a*c, 0.0, 10.0**100))    
    
    #print(D[:10])
    
    #w=0
    #for i in range(len(b*b - 4*a*c)):
        #if b[i]*b[i] - 4*a[i]*c[i] < 0 and w<10:
            #print ('Issue %i', i)
            #w = w+1
    #in case any non physical angle has slipped through, shouldn't have
    #q = b + np.sign(b)*np.sqrt(D)
    q = -0.5 *(b + np.sign(b)*np.sqrt(D))
    #print(q)
    sol1_2 = c/q
    sol2_2 = q/a
    
    
    print(sol1_2 [90:100])
    print(sol2_2 [90:100])
    
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



def closest_rotation(a1_dir_hat, sv_dir_hat, theta_GJ, alpha, beta):
    a1_dir_hat = np.array(a1_dir_hat)
    sv_dir_hat = np.array(sv_dir_hat)
    
    a1_x = a1_dir_hat[0]
    a1_y = a1_dir_hat[1]
    
    #create a normalised vector prependicular to a1
    n_1_x = 1/np.sqrt(1+(a1_x/a1_y)**2)
    n_1_y = -a1_x * n_1_x / a1_y
    n_1 = np.array([n_1_x, n_1_y, 0])
    
    #construct a second vector perpendicular to a1 and to the first one to define the plane perpendicular to a1
    n_2 = np.array(np.cross(n_1, a1_dir_hat)/bf.norm(np.cross(n_1, a1_dir_hat)))
    
    #Now any 'y' vector describing the outward-ness of the tau vector can be described by the linear combinaison of those two vectors
    #we can choose the sign cause modulus squared need to be one
    #the linear combinaison factor
    
    y = alpha * n_1 + beta * n_2
    y_hat = y/bf.norm(y)
    
    tau_guess = np.cos(theta_GJ)*a1_dir_hat + np.sin(theta_GJ) * y_hat
    
    return np.array(tau_guess)



def closest_tau(a1_dir_hat, sv_dir_hat, theta_GJ):
    #if we are smaller than theta max, no need to do anything
    diff_angle = abs(np.arccos(np.clip(bf.dot_product(a1_dir_hat, sv_dir_hat), -1, 1))) - abs(theta_GJ)
    
    
    a1_dir_hat = np.array(a1_dir_hat)
    sv_dir_hat = np.array(sv_dir_hat)
    
    a1_x = a1_dir_hat[0]
    a1_y = a1_dir_hat[1]
    
    #create a normalised vector prependicular to a1
    n_1_x = 1/np.sqrt(1+(a1_x/a1_y)**2)
    n_1_y = -a1_x * n_1_x / a1_y
    n_1 = np.array([n_1_x, n_1_y, np.zeros(len(n_1_y))])
    
    #construct a second vector perpendicular to a1 and to the first one to define the plane perpendicular to a1
    #n_2 = np.array(np.cross(n_1, a1_dir_hat)/bf.norm(np.cross(n_1, a1_dir_hat)))
    n_2 = []
    
    for i in range(len(a1_x)):
        n_2.append(np.cross(np.transpose(n_1)[i], np.transpose(a1_dir_hat)[i])/bf.norm(np.cross(np.transpose(n_1)[i], np.transpose(a1_dir_hat)[i])))
        
    n_2 = np.transpose(np.array(n_2))
    #Now any 'y' vector describing the outward-ness of the tau vector can be described by the linear combinaison of those two vectors
    #we can choose the sign cause modulus squared need to be one
    #the linear combinaison factor
    factors = np.linspace(-1, 1, 100)
    old_dot = np.zeros(len(n_1_y))-9999

    #naive estimation -  bad?
    tau_best = np.cos(theta_GJ)*a1_dir_hat + np.sin(theta_GJ) * a1_dir_hat
    
    for alpha in factors:
        for beta in factors:
            y = alpha * n_1 + beta * n_2
            y_hat = y/bf.norm(y)
    
            tau_guess = np.cos(theta_GJ)*a1_dir_hat + np.sin(theta_GJ) * y_hat
            tau_guess = tau_guess/bf.norm(tau_guess)
            
            dot_new = bf.dot_product(tau_guess, sv_dir_hat)-1
            
            tau_best = np.where(dot_new>=old_dot, tau_guess, tau_best)
            old_dot = np.where(dot_new>=old_dot, dot_new, old_dot)
            
            print(old_dot)
    
    
    tau_best = np.where(diff_angle>=0, tau_best, sv_dir_hat)
    return tau_best




def polarimetric_change_dir(df, decay_mode1, decay_mode2):
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
    
    #if we end up reaching theta_max -> we need to improve the tau_direction
    updated_dir_tau1 = closest_tau(vis_1_dir/norm_vis_1, tau1_dir, theta_GJ)
    dir_x_tau1 = updated_dir_tau1[0]
    dir_y_tau1 = updated_dir_tau1[1]
    dir_z_tau1 = updated_dir_tau1[2]
    
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
    
    cos_theta_GJ = np.clip(bf.dot_product(tau2_dir, vis_2_dir/norm_vis_2), -1, 1)
    theta_GJ = np.arccos(cos_theta_GJ)
    
    
    theta_GJ_max = np.arcsin(np.clip((m_tau**2 - m_vis**2)/(2*m_tau*tau_2_vis.p), -1, 1))
    theta_GJ = tf.where(theta_GJ >= theta_GJ_max, theta_GJ_max, theta_GJ)
    
    minus_b = (m_vis**2 + m_tau**2) * tau_2_vis.p * np.cos(theta_GJ)
    two_a = 2*(m_vis**2 + tau_2_vis.p2 * (np.sin(theta_GJ))**2)
    b_squared_m_four_ac = (m_vis**2 + tau_2_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_2_vis.p2*(np.sin(theta_GJ))**2)
    b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)

    sol1_2 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
    sol2_2 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a     
    
    #if we end up reaching theta_max -> we need to improve the tau_direction
    updated_dir_tau2 = closest_tau(vis_2_dir/norm_vis_2, tau2_dir, theta_GJ)
    
    dir_x_tau2 = updated_dir_tau2[0]
    dir_y_tau2 = updated_dir_tau2[1]
    dir_z_tau2 = updated_dir_tau2[2]
    
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

