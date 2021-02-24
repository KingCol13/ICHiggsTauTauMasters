import numpy as np
from pylorentz import Momentum4
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors

def norm(vector):
    if len(vector)!=3:
        print('This is only for a 3d vector')
    return np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)


def dot_product(vector1,vector2):
    if len(vector1)!=len(vector2):
        raise Arrays_of_different_size
    prod=0
    for i in range(len(vector1)):
        prod=prod+vector1[i]*vector2[i]
    return prod


def get_vis(df4, decay1, decay2):
    pi_1_4Mom = Momentum4(df4["pi_E_1"],df4["pi_px_1"],df4["pi_py_1"],df4["pi_pz_1"])
    if decay1 == 10:
        pi2_1_4Mom = Momentum4(df4["pi2_E_1"],df4["pi2_px_1"],df4["pi2_py_1"],df4["pi2_pz_1"])
        pi3_1_4Mom = Momentum4(df4["pi3_E_1"],df4["pi3_px_1"],df4["pi3_py_1"],df4["pi3_pz_1"])
        tau_1_vis = pi_1_4Mom + pi2_1_4Mom + pi3_1_4Mom 
    if decay1 == 1 or decay1 ==2 :
        pi0_1_4Mom = Momentum4(df4["pi0_E_1"],df4["pi0_px_1"],df4["pi0_py_1"],df4["pi0_pz_1"])
        tau_1_vis = pi_1_4Mom + pi0_1_4Mom
    pi_2_4Mom = Momentum4(df4["pi_E_2"],df4["pi_px_2"],df4["pi_py_2"],df4["pi_pz_2"])  
    if decay2 == 10:
        pi2_2_4Mom = Momentum4(df4["pi2_E_2"],df4["pi2_px_2"],df4["pi2_py_2"],df4["pi2_pz_2"]) 
        pi3_2_4Mom = Momentum4(df4["pi3_E_2"],df4["pi3_px_2"],df4["pi3_py_2"],df4["pi3_pz_2"]) 
        tau_2_vis = pi_2_4Mom + pi2_2_4Mom + pi3_2_4Mom 
    if decay2 == 1 or decay2 == 2:
        pi0_2_4Mom = Momentum4(df4["pi0_E_2"],df4["pi0_px_2"],df4["pi0_py_2"],df4["pi0_pz_2"])
        tau_2_vis = pi_2_4Mom + pi0_2_4Mom
    
    return tau_1_vis, tau_2_vis

def get_gen_vis(df, decay1, decay2):
    tau_1_vis = Momentum4.e_eta_phi_p(df["gen_vis_E_1"],df["gen_vis_eta_1"],df["gen_vis_phi_1"],df["gen_vis_p_1"]) 
    
    tau_2_vis = Momentum4.e_eta_phi_p(df["gen_vis_E_2"],df["gen_vis_eta_2"],df["gen_vis_phi_2"],df["gen_vis_p_2"])
    return tau_1_vis, tau_2_vis


def Mom4_to_tf(Mom4_1D):
    return tf.convert_to_tensor(Mom4_1D, dtype = 'float32')


def cross_product(vector3_1,vector3_2):
    if len(vector3_1)!=3 or len(vector3_1)!=3:
        print('These are not 3D arrays !')
    x_perp_vector=vector3_1[1]*vector3_2[2]-vector3_1[2]*vector3_2[1]
    y_perp_vector=vector3_1[2]*vector3_2[0]-vector3_1[0]*vector3_2[2]
    z_perp_vector=vector3_1[0]*vector3_2[1]-vector3_1[1]*vector3_2[0]
    return np.array([x_perp_vector,y_perp_vector,z_perp_vector])


def plot_2d(qqty1, qqty2, label1, label2, xlim = (0, 100), ylim = (0,100), nb_bins = (100,100)):
    plt.hist2d(qqty1, qqty2, bins = nb_bins, norm=colors.LogNorm())
    plt.xlabel('%s'%label1, fontsize = 'x-large')
    plt.ylabel('%s'%label2, fontsize = 'x-large')
    #plt.plot([xlim[0], xlim[1]],[xlim[0], xlim[1]], 'k--')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.colorbar()
    plt.grid()
    plt.show()
    return 0


def plot_2d_normed(qqty1, qqty2, label1, label2, xlim = (0, 100), ylim = (0,100), nb_bins = (100,100)):
    plt.hist2d(qqty1, qqty2, bins = nb_bins, norm=colors.LogNorm(), density = True)
    plt.xlabel('%s'%label1, fontsize = 'x-large')
    plt.ylabel('%s'%label2, fontsize = 'x-large')
    #plt.plot([xlim[0], xlim[1]],[xlim[0], xlim[1]], 'k--')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.colorbar()
    plt.grid()
    plt.show()
    return 0



def plot_1d(qqty1, qqty2, label1, label2, xlim = (-100, 100), nb_bins = 1000):
    plt.hist((qqty1 - qqty2), bins = nb_bins, label = 'mean:%.2f std: %.2f'%(np.array(qqty1-qqty2).mean(), np.array(qqty1-qqty2).std()))
    plt.xlim(xlim[0], xlim[1])
    
    plt.xlabel('%s - %s'%(label1, label2), fontsize = 'x-large')
    plt.legend(prop = {'size':14})
    plt.grid()
    plt.show()
    return 0
