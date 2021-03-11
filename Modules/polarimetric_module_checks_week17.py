# This module helps calculating the best polarimetric approximation for neutrinos for given decay mode

import numpy as np
import pandas as pd
import tensorflow as tf
from pylorentz import Momentum4
import basic_functions as bf


def polarimetric_full(df, decay_mode1, decay_mode2, direction = 'geo', selector= 'HiggsM_met', theta = False):
    if decay_mode1 == 10 and decay_mode2 == 10:
        return a1_a1_polarimetric(df, decay_mode1, decay_mode2, direction, selector, theta)
    if decay_mode1 == 10 and decay_mode2 == 0:
        return a1_pi_polarimetric(df, decay_mode1, decay_mode2, direction, selector, theta)


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
    factors = np.linspace(-1, 1, 50)
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
            
            #print(old_dot)
    
    
    tau_best = np.where(diff_angle>=0, tau_best, sv_dir_hat)
    return tau_best



def met_choice(Higgs_option1, Higgs_option2, nu_1_option1, nu_2_option1, nu_1_option2, nu_2_option2, df4, alpha = 0, beta = 0):
    
    m_Higgs = 125.10
    
    reco_met_x_option1 = nu_1_option1.p_x + nu_2_option1.p_x
    reco_met_x_option2 = nu_1_option2.p_x + nu_2_option2.p_x
    
    reco_met_y_option1 = nu_1_option1.p_y + nu_2_option1.p_y
    reco_met_y_option2 = nu_1_option2.p_y + nu_2_option2.p_y
    
    diff_Higgs1 = (Higgs_option1.m-m_Higgs)**2
    diff_Higgs2 = (Higgs_option2.m-m_Higgs)**2
    
    diff_metx1 = (reco_met_x_option1 - df4['metx'])**2
    diff_mety1 = (reco_met_y_option1 - df4['mety'])**2
    
    diff_metx2 = (reco_met_x_option2 - df4['metx'])**2
    diff_mety2 = (reco_met_y_option2 - df4['mety'])**2
    
    total_diff1 = diff_Higgs1 + alpha * diff_metx1 + beta * diff_mety1
    total_diff2 = diff_Higgs2 + alpha * diff_metx2 + beta * diff_mety2
    
    better_higgs = tf.where(total_diff1 <= total_diff2, Higgs_option1, Higgs_option2)
    
    better_nu1 = tf.where(total_diff1 <= total_diff2, nu_1_option1, nu_1_option2)
    
    better_nu2 = tf.where(total_diff1 <= total_diff2, nu_2_option1, nu_2_option2)
    
    better_higgs = Momentum4(better_higgs[0], better_higgs[1], better_higgs[2], better_higgs[3])
    
    better_nu1 = Momentum4(better_nu1[0], better_nu1[1], better_nu1[2], better_nu1[3])
    
    better_nu2 = Momentum4(better_nu2[0], better_nu2[1], better_nu2[2], better_nu2[3])
    
    return better_higgs, better_nu1, better_nu2



def tau_mass(tau_option1, tau_option2, nu_option1, nu_option2):
    m_tau = 1.77686
    diff1 = (tau_option1.m - m_tau)**2
    diff2 = (tau_option2.m - m_tau)**2
    
    better_tau = tf.where(diff1 <= diff2, tau_option1, tau_option2)
    better_nu1 = tf.where(diff1 <= diff2, nu_option1, nu_option2)
    
    better_tau = Momentum4(better_tau[0], better_tau[1], better_tau[2], better_tau[3])
    
    better_nu1 = Momentum4(better_nu1[0], better_nu1[1], better_nu1[2], better_nu1[3])
    
    return better_tau, better_nu1


def best_p_choice(tau_option1, tau_option2, nu_option1, nu_option2, df4, index, decay_mode1, decay_mode2):
    
    tau_1_vis, tau_2_vis = bf.get_vis(df4, decay_mode1, decay_mode2) 
    
    if index == 1:
        nu = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_1"])), df4["gen_nu_eta_1"], df4["gen_nu_phi_1"], df4["gen_nu_p_1"])
        pseudo_tau =  tau_1_vis + nu
        
    if index == 2:
        nu = Momentum4.m_eta_phi_p(np.zeros(len(df4["gen_nu_phi_2"])), df4["gen_nu_eta_2"], df4["gen_nu_phi_2"], df4["gen_nu_p_2"])
        pseudo_tau =  tau_2_vis + nu
    
    total_diff1 = (tau_option1.p - pseudo_tau.p)**2
    total_diff2 = (tau_option2.p - pseudo_tau.p)**2
    
    better_tau = tf.where(total_diff1 <= total_diff2, tau_option1, tau_option2)
    better_nu = tf.where(total_diff1 <= total_diff2, nu_option1, nu_option2)
    
    better_tau = Momentum4(better_tau[0], better_tau[1], better_tau[2], better_tau[3])
    better_nu = Momentum4(better_nu[0], better_nu[1], better_nu[2], better_nu[3])
    
    return better_tau, better_nu


def a1_a1_polarimetric(df, decay_mode1, decay_mode2, direction, selector, theta = False):
    m_tau = 1.77686
    
    tau_1_vis, tau_2_vis = bf.get_vis(df, decay_mode1, decay_mode2) 
    m_vis = tau_1_vis.m
    m_vis = np.where(tau_1_vis.m==0, 1.260, tau_1_vis.m)
    norm_sv_1 = bf.norm([df['sv_x_1'], df['sv_y_1'], df['sv_z_1']])
    norm_sv_1 = np.where(norm_sv_1 == 0, 9999, norm_sv_1)

    dir_x_tau1 = df['sv_x_1']/norm_sv_1
    dir_y_tau1 = df['sv_y_1']/norm_sv_1
    dir_z_tau1 = df['sv_z_1']/norm_sv_1
    tau1_dir = [dir_x_tau1, dir_y_tau1, dir_z_tau1]
    vis_1_dir = [tau_1_vis.p_x, tau_1_vis.p_y, tau_1_vis.p_z]
    norm_vis_1 = np.where(bf.norm(vis_1_dir) == 0, 9999, bf.norm(vis_1_dir))
    
    
    theta_GJ = np.arccos(np.clip(bf.dot_product(tau1_dir, vis_1_dir/norm_vis_1), -1, 1))
    theta_GJ_max = np.arcsin(np.clip((m_tau**2 - m_vis**2)/(2*m_tau*tau_1_vis.p), -1, 1))
    theta_GJ = tf.where(theta_GJ >= theta_GJ_max, theta_GJ_max, theta_GJ)
    
    check1 =  tf.where(theta_GJ == theta_GJ_max, 1, 0)
    
    #print(sum(check)/len(check))
    
    #5) caluclate the tau momentum 
    minus_b = (m_vis**2 + m_tau**2) * tau_1_vis.p * np.cos(theta_GJ)
    two_a = 2*(m_vis**2 + tau_1_vis.p2 * (np.sin(theta_GJ))**2)
    b_squared_m_four_ac = (m_vis**2 + tau_1_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_1_vis.p2*(np.sin(theta_GJ))**2)
    b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)
    
    #6) have the two solutions
    sol1_1 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
    sol2_1 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a
    
    #Here: are we changing direction
    if direction == 'naive':
        updated_dir_tau1 = vis_1_dir/norm_vis_1*np.cos(theta_GJ) + vis_1_dir/norm_vis_1*np.sin(theta_GJ)
    if direction == 'geo':
        updated_dir_tau1 = closest_tau(vis_1_dir/norm_vis_1, tau1_dir, theta_GJ)
    if direction == 'sv':
        updated_dir_tau1 = tau1_dir
    
    if direction != 'sv' and direction != 'naive' and direction != 'geo':
        print('\n Wrong direction, please pick either of:\n- "sv"\n-"naive\n-"geo\n""')
        raise WrongDir
    
    
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
    
    
    check2 =  tf.where(theta_GJ == theta_GJ_max, 1, 0)
    #print(sum(check)/len(check))
    
    minus_b = (m_vis**2 + m_tau**2) * tau_2_vis.p * np.cos(theta_GJ)
    two_a = 2*(m_vis**2 + tau_2_vis.p2 * (np.sin(theta_GJ))**2)
    b_squared_m_four_ac = (m_vis**2 + tau_2_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_2_vis.p2*(np.sin(theta_GJ))**2)
    b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)

    sol1_2 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
    sol2_2 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a     
    
    #Here: are we changing direction
    if direction == 'naive':
        updated_dir_tau2 = vis_2_dir/norm_vis_2*np.cos(theta_GJ) + vis_2_dir/norm_vis_2*np.sin(theta_GJ)
    if direction == 'geo':
        updated_dir_tau2 = closest_tau(vis_2_dir/norm_vis_2, tau2_dir, theta_GJ)
    if direction == 'sv':
        updated_dir_tau2 = tau2_dir
    if direction != 'sv' and direction != 'naive' and direction != 'geo':
        print('\n Wrong direction, please pick either of:\n- "sv"\n-"naive"\n-"geo"\n')
        raise WrongDir
    
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
    
    
    if selector == 'HiggsM':
        alpha = 0
        beta = 0
        higgs_best1, nu_1_best1, nu_2_best1 = met_choice(Higgs_1_1, Higgs_2_2, nu_1_guess1, nu_2_guess1, nu_1_guess2, nu_2_guess2, df, alpha, beta)
        higgs_best2, nu_1_best2, nu_2_best2 = met_choice(Higgs_1_2, Higgs_2_1, nu_1_guess1, nu_2_guess2, nu_1_guess2, nu_2_guess1, df,  alpha, beta)
        higgs_best, best_guess_nu1, best_guess_nu2 = met_choice(higgs_best1, higgs_best2, nu_1_best1, nu_2_best1, nu_1_best2, nu_2_best2, df, alpha, beta)
    
    if selector == 'HiggsM_met':
        if direction == 'naive':
            alpha = 0.426
            beta = 0.384
        if direction == 'sv' or direction == 'geo':
            alpha = 0.405
            beta = 0.405
        higgs_best1, nu_1_best1, nu_2_best1 = met_choice(Higgs_1_1, Higgs_2_2, nu_1_guess1, nu_2_guess1, nu_1_guess2, nu_2_guess2, df, alpha, beta)
        higgs_best2, nu_1_best2, nu_2_best2 = met_choice(Higgs_1_2, Higgs_2_1, nu_1_guess1, nu_2_guess2, nu_1_guess2, nu_2_guess1, df,  alpha, beta)
        higgs_best, best_guess_nu1, best_guess_nu2 = met_choice(higgs_best1, higgs_best2, nu_1_best1, nu_2_best1, nu_1_best2, nu_2_best2, df, alpha, beta)
        
    if selector == 'tau_p':
        tau1_best, best_guess_nu1 = best_p_choice(tau1_sol_1, tau1_sol_2, nu_1_guess1, nu_1_guess2, df, 1, decay_mode1, decay_mode2)
        tau2_best, best_guess_nu2 = best_p_choice(tau2_sol_1, tau2_sol_2, nu_2_guess1, nu_2_guess2, df, 2, decay_mode1, decay_mode2)
        
        
    if selector != 'tau_p' and selector != 'HiggsM' and selector != 'HiggsM_met':
        print('\n Wrong selector, please pick either of:\n-"tau_p"\n-"HiggsM"\n-"HiggsM_met"\n')
        raise WrongSelector
    
    
    if theta == False:
    #we return the two neutrinos 4 vectors
        return nu_1_guess1, nu_1_guess2, nu_2_guess1, nu_2_guess2, best_guess_nu1, best_guess_nu2 
    
    if theta == True:
        return nu_1_guess1, nu_1_guess2, nu_2_guess1, nu_2_guess2, best_guess_nu1, best_guess_nu2, check1 + check2



def a1_pi_polarimetric(df, decay_mode1, decay_mode2, direction, selector, theta = False):
    m_tau = 1.77686
    tau_1_vis, tau_2_vis = bf.get_vis(df, decay_mode1, decay_mode2) 
    if decay_mode1 == 10:
        m_vis = tau_1_vis.m
        m_vis = np.where(tau_1_vis.m==0, 1.260, tau_1_vis.m)
        norm_sv_1 = bf.norm([df['sv_x_1'], df['sv_y_1'], df['sv_z_1']])
        norm_sv_1 = np.where(norm_sv_1 == 0, 9999, norm_sv_1)

        dir_x_tau1 = df['sv_x_1']/norm_sv_1
        dir_y_tau1 = df['sv_y_1']/norm_sv_1
        dir_z_tau1 = df['sv_z_1']/norm_sv_1
        tau1_dir = [dir_x_tau1, dir_y_tau1, dir_z_tau1]
        vis_1_dir = [tau_1_vis.p_x, tau_1_vis.p_y, tau_1_vis.p_z]
        norm_vis_1 = np.where(bf.norm(vis_1_dir) == 0, 9999, bf.norm(vis_1_dir))
        
        
        theta_GJ = np.arccos(np.clip(bf.dot_product(tau1_dir, vis_1_dir/norm_vis_1), -1, 1))
        theta_GJ_max = np.arcsin(np.clip((m_tau**2 - m_vis**2)/(2*m_tau*tau_1_vis.p), -1, 1))
        theta_GJ = tf.where(theta_GJ >= theta_GJ_max, theta_GJ_max, theta_GJ)
        check1 =  tf.where(theta_GJ == theta_GJ_max, 1, 0)
        
        #5) caluclate the tau momentum 
        minus_b = (m_vis**2 + m_tau**2) * tau_1_vis.p * np.cos(theta_GJ)
        two_a = 2*(m_vis**2 + tau_1_vis.p2 * (np.sin(theta_GJ))**2)
        b_squared_m_four_ac = (m_vis**2 + tau_1_vis.p2)*((m_vis**2 - m_tau**2)**2 - 4*m_tau**2*tau_1_vis.p2*(np.sin(theta_GJ))**2)
        b_squared_m_four_ac = tf.where (b_squared_m_four_ac<=0, b_squared_m_four_ac*0, b_squared_m_four_ac)
        
        #6) have the two solutions
        sol1_1 = (minus_b + np.sqrt(b_squared_m_four_ac))/two_a
        sol2_1 = (minus_b - np.sqrt(b_squared_m_four_ac))/two_a
        
        #Here: are we changing direction
        if direction == 'naive':
            updated_dir_tau1 = vis_1_dir/norm_vis_1*np.cos(theta_GJ) + vis_1_dir/norm_vis_1*np.sin(theta_GJ)
        if direction == 'geo':
            updated_dir_tau1 = closest_tau(vis_1_dir/norm_vis_1, tau1_dir, theta_GJ)
        if direction == 'sv':
            updated_dir_tau1 = tau1_dir
        
        if direction != 'sv' and direction != 'naive' and direction != 'geo':
            print('\n Wrong direction, please pick either of:\n- "sv"\n-"naive\n-"geo\n""')
            raise WrongDir
        
        
        dir_x_tau1 = updated_dir_tau1[0]
        dir_y_tau1 = updated_dir_tau1[1]
        dir_z_tau1 = updated_dir_tau1[2]
            
        
        tau1_sol_1 = Momentum4(np.sqrt(sol1_1**2+m_tau**2), dir_x_tau1*sol1_1, dir_y_tau1*sol1_1, dir_z_tau1*sol1_1)
        tau1_sol_2 = Momentum4(np.sqrt(sol2_1**2+m_tau**2), dir_x_tau1*sol2_1, dir_y_tau1*sol2_1, dir_z_tau1*sol2_1) 
        
        nu_1_guess1 = tau1_sol_1 - tau_1_vis
        nu_1_guess2 = tau1_sol_2 - tau_1_vis
        
        #take the average of the two solutions for the tau which in 63% of the times will be just the maxTheta 
        nu_1 = (nu_1_guess1 + nu_1_guess2)/2
        
        #check with gen info to make sure
        
        nu_1_gen = Momentum4.m_eta_phi_p(np.zeros(len(df["gen_nu_phi_1"])), df["gen_nu_eta_1"], df["gen_nu_phi_1"], df["gen_nu_p_1"])
        
        nu_2 = Momentum4.m_eta_phi_p(np.zeros(len(df["gen_nu_phi_2"])), df["gen_nu_eta_2"], df["gen_nu_phi_2"], df["gen_nu_p_2"])
        
        gen_metx = nu_1_gen.p_x + nu_2.p_x
        gen_mety = nu_1_gen.p_y + nu_2.p_y
        
        nu_2_px = gen_metx - nu_1.p_x  #nu_2.p_x #df['metx']
        nu_2_py = gen_mety - nu_1.p_y  #nu_2.p_y #df['mety']
        
        m_vis_2 = tau_2_vis.m
        #Now two solutions for nu_z
        a = m_tau**2 - m_vis_2**2 + 2 * nu_2_px * tau_2_vis.p_x + 2 * nu_2_py * tau_2_vis.p_y
        b = 2 * tau_2_vis.e
        c = 2 * tau_2_vis.p_z
        d = nu_2_px**2 + nu_2_py**2
        
        
        E_nu2_sol1 = (a*b + np.sqrt(np.clip(a**2 * c**2 - b**2 * c**2 *d + c**4 * d, 0, 10**10))) / (b**2 - c**2)
        E_nu2_sol2 = (a*b - np.sqrt(np.clip(a**2 * c**2 - b**2 * c**2 *d + c**4 * d, 0, 10**10))) / (b**2 - c**2)
        
        #print(E_nu2_sol1)
        #print(E_nu2_sol2)
        #print(sum(np.where(tau_2_vis.p_z==0, 1, 0)))
        
        nu_2_pz_sol1 = np.sqrt(np.clip(E_nu2_sol1**2 - nu_2_px**2 - nu_2_py**2, 0, 10**10))
        nu_2_pz_sol2 = np.sqrt(np.clip(E_nu2_sol2**2 - nu_2_px**2 - nu_2_py**2, 0, 10**10))
        
        
        nu_2_guess1 = Momentum4(0*nu_2_px, nu_2_px, nu_2_py, nu_2_pz_sol1)
        nu_2_guess2 = Momentum4(0*nu_2_px, nu_2_px, nu_2_py, -nu_2_pz_sol1)
        nu_2_guess3 = Momentum4(0*nu_2_px, nu_2_px, nu_2_py, nu_2_pz_sol2)
        nu_2_guess4 = Momentum4(0*nu_2_px, nu_2_px, nu_2_py, -nu_2_pz_sol2)
        
        tau2_sol_1 = nu_2_guess1 + tau_2_vis
        tau2_sol_2 = nu_2_guess2 + tau_2_vis
        tau2_sol_3 = nu_2_guess3 + tau_2_vis
        tau2_sol_4 = nu_2_guess4 + tau_2_vis
        
        #out of the - and + solution,  take sol with gives closest tau mass for the two sols of energy
        tau2_sol_1, nu_2_guess1 = tau_mass(tau2_sol_1, tau2_sol_2, nu_2_guess1, nu_2_guess2)
        tau2_sol_2, nu_2_guess2 = tau_mass(tau2_sol_3, tau2_sol_4, nu_2_guess3, nu_2_guess4)
        
        Higgs_1_1 = tau1_sol_1 + tau2_sol_1
        Higgs_1_2 = tau1_sol_1 + tau2_sol_2
        Higgs_2_1 = tau1_sol_2 + tau2_sol_1
        Higgs_2_2 = tau1_sol_2 + tau2_sol_2
        

        if selector == 'HiggsM':
            alpha = 0
            beta = 0
            higgs_best1, nu_1_best1, nu_2_best1 = met_choice(Higgs_1_1, Higgs_2_2, nu_1_guess1, nu_2_guess1, nu_1_guess2, nu_2_guess2, df, alpha, beta)
            higgs_best2, nu_1_best2, nu_2_best2 = met_choice(Higgs_1_2, Higgs_2_1, nu_1_guess1, nu_2_guess2, nu_1_guess2, nu_2_guess1, df,  alpha, beta)
            
            higgs_best, best_guess_nu1, best_guess_nu2 = met_choice(higgs_best1, higgs_best2, nu_1_best1, nu_2_best1, nu_1_best2, nu_2_best2, df, alpha, beta)
            
        if selector == 'tau_p':
            tau1_best, best_guess_nu1 = best_p_choice(tau1_sol_1, tau1_sol_2, nu_1_guess1, nu_1_guess2, df, 1, decay_mode1, decay_mode2)
            tau2_best, best_guess_nu2 = best_p_choice(tau2_sol_1, tau2_sol_2, nu_2_guess1, nu_2_guess2, df, 2, decay_mode1, decay_mode2)
            
        if selector != 'tau_p' and selector != 'HiggsM':
            print('\n Wrong selector for a1-pi channel, please pick either of:\n-"tau_p"\n-"HiggsM"\n')
            raise WrongSelector
        
        if theta == False:
                #we return the two neutrinos 4 vectors
            return nu_1_guess1, nu_1_guess2, nu_2_guess1, nu_2_guess2, best_guess_nu1, best_guess_nu2 
                
        if theta == True:
            return nu_1_guess1, nu_1_guess2, nu_2_guess1, nu_2_guess2, best_guess_nu1, best_guess_nu2, check1
        
    

