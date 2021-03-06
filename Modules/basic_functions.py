import numpy as np
from pylorentz import Momentum4
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors

def norm(vector):
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
    if decay1 == 0:
        tau_1_vis = pi_1_4Mom 
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
    if decay2 == 0:
        tau_2_vis = pi_2_4Mom 
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
    
    plt.xlabel('%s - %s'%(label1, label2), fontsize = 'large')
    plt.legend(prop = {'size':10})
    #plt.grid()
    #plt.show()
    return 0
    
    
def plot_sm_ps(angle, marker, df, nbins, name = 0):
    df_ps, df_sm = make_ps_sm(df)
    bins_array = np.linspace(0, 2*np.pi , nbins+1)
    zeros = np.linspace(0, 0, nbins+1)+1

 #   a = df[angle]
 #   list_bins = [0]
 #   for i in range (nbins):
 #       list_bins.append(np.quantile(a, (i+1)*1/nbins))
 #   bins_array = np.array(list_bins)

    hist_sm = np.histogram(df_sm[angle], bins = bins_array)
    hist_ps = np.histogram(df_ps[angle], bins = bins_array)
    if angle == 'aco_angle_1' or angle == 'aco_angle_5':
        #name = '1/' + angle
        plt.plot(bins_array[:-1], hist_ps[0]/hist_sm[0], marker, alpha = 0.7, label = name)
    else:
        if name == 0:
            name = angle
        plt.plot(bins_array[:-1], hist_sm[0]/hist_ps[0], marker, alpha = 0.7, label = name)
    plt.plot(bins_array[:-1], zeros[:-1], 'k-')
    
def make_ps_sm(df4):
    df_ps = df4[
        (df4["rand"]<df4["wt_cp_ps"]/2)     #a data frame only including the pseudoscalars
    ]

    df_sm = df4[
        (df4["rand"]<df4["wt_cp_sm"]/2)     #data frame only including the scalars
    ]
    return df_ps, df_sm
    

   
def improvement(df, variable_new, variable_old, nbins = 4):
    df_ps, df_sm = make_ps_sm(df)
    bins_array = np.linspace(0, 2*np.pi, nbins+1)
    #print(bins_array)
    #a = df[variable_old]
    #list_bins = []
#    for i in range (nbins):
#        list_bins.append(np.quantile(a, (i+1)*1/nbins))
#        print((i+1)*1/nbins)
#    bins_array = np.array(list_bins)
    
    #print(bins_array)
    
    hist_sm_new= np.histogram(df_sm[variable_new], bins = bins_array)
    hist_ps_new = np.histogram(df_ps[variable_new], bins = bins_array)
    
    hist_sm_old= np.histogram(df_sm[variable_old], bins = bins_array)
    hist_ps_old = np.histogram(df_ps[variable_old], bins = bins_array)
    
    sep_new = np.array(hist_sm_new[0]/hist_ps_new[0])
    sep_old = np.array(hist_sm_old[0]/hist_ps_old[0])
    
    if variable_old == 'aco_angle_1':
        sep_old = np.array(hist_ps_old[0]/hist_sm_old[0])
    print(min(sep_old), min(sep_new), sep_old.mean())
    
     #
    return -(min(sep_old)-min(sep_new))/(min(sep_old)-1)#(1-min(sep_new))/(1-min(sep_old))
        

