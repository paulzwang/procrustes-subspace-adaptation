import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(r'matplotlib_stylesheet\journal_nolatex.mplstyle')
import utils
from utils import NeuralNetwork
from utils import train_model
from utils import add_percent_noise
import psa
import TESTING_psa

import torch 
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import scipy.linalg as linalg

def read_data(data_directory):
    df = pd.read_csv(data_directory).drop_duplicates(subset=['time'], keep='first')
    t = np.array(df['time'])
    X = np.concatenate((np.array(df['rho']).reshape(-1,1),np.array(df['T']).reshape(-1,1),np.array(df['S']).reshape(-1,1)),axis=1)
    Y = np.array(df['heat_rate']).reshape(-1,1)
    return df, t, X, Y

def TS_to_H(time_series,window_length):
    """
    Transform 1D input time series into a trajectory matrix
    """
    L = window_length # The window length.
    N = len(time_series)
    K = N - L + 1 # The number of columns in the trajectory matrix.
    X = np.column_stack([time_series[i:i+L] for i in range(0,K)])
    return X

def find_trajectory_matrix(X,L=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(0,np.size(X,1)):
        Hxi = torch.tensor(TS_to_H(X[:,i],L),dtype=torch.float32).to(device)
        if i==0:
            Hx = Hxi
        else:
            Hx = torch.cat((Hx,Hxi),dim=0)
    return Hx

def H_to_TS(X_i):
    """
    Reconstructs input time series from trajectory matrix
    """
    # Reverse the column ordering of X_i
    X_rev = X_i[::-1]
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])


if __name__ == '__main__':
    mission1_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=100.0\Results_ctrl=0_ra=12000_rp=100.0_hl=0.150_90.0deg.csv'
    mission2_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=99.0\Results_ctrl=0_ra=12000_rp=99.0_hl=0.150_90.0deg.csv'
    mission3_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=98.0\Results_ctrl=0_ra=12000_rp=98.0_hl=0.150_90.0deg.csv'
    mission4_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=97.0\Results_ctrl=0_ra=12000_rp=97.0_hl=0.150_90.0deg.csv'
    mission5_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=96.0\Results_ctrl=0_ra=12000_rp=96.0_hl=0.150_90.0deg.csv'
    mission6_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=95.0\Results_ctrl=0_ra=12000_rp=95.0_hl=0.150_90.0deg.csv'
    mission7_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=94.0\Results_ctrl=0_ra=12000_rp=94.0_hl=0.150_90.0deg.csv'
    mission8_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=93.0\Results_ctrl=0_ra=12000_rp=93.0_hl=0.150_90.0deg.csv'
    mission9_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=92.0\Results_ctrl=0_ra=12000_rp=92.0_hl=0.150_90.0deg.csv'
    mission10_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=91.0\Results_ctrl=0_ra=12000_rp=91.0_hl=0.150_90.0deg.csv'
    mission11_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=90.0\Results_ctrl=0_ra=12000_rp=90.0_hl=0.150_90.0deg.csv'
    shift_list = [r'$r_p=90$ km', 
                  r'$r_p=91$ km', 
                  r'$r_p=92$ km', 
                  r'$r_p=93$ km', 
                  r'$r_p=94$ km', 
                  r'$r_p=95$ km', 
                  r'$r_p=96$ km', 
                  r'$r_p=97$ km',
                  r'$r_p=98$ km',
                  r'$r_p=99$ km'] # Model training performed on descending shifts

    # Read data from CSV
    df1, t1, X1, Y1 = read_data(mission1_data_directory)
    df2, t2, X2, Y2 = read_data(mission2_data_directory)
    df3, t3, X3, Y3 = read_data(mission3_data_directory)
    df4, t4, X4, Y4 = read_data(mission4_data_directory)
    df5, t5, X5, Y5 = read_data(mission5_data_directory)
    df6, t6, X6, Y6 = read_data(mission6_data_directory)
    df7, t7, X7, Y7 = read_data(mission7_data_directory)
    df8, t8, X8, Y8 = read_data(mission8_data_directory)
    df9, t9, X9, Y9 = read_data(mission9_data_directory)
    df10, t10, X10, Y10 = read_data(mission10_data_directory)
    df11, t11, X11, Y11 = read_data(mission11_data_directory)

    inputdomain_list = [X11, X10, X9, X8, X7, X6, X5, X4, X3, X2]

    window_length = 5
    fig0, ax0 = plt.subplots(2,2,layout='constrained')
    list_window_length = np.arange(1,51,1)
    linecolor_list0 = np.linspace(0.9,0.1,len(list_window_length))

    fig1, ax1 = plt.subplots(1,2,figsize=(3.25,2),layout='constrained')
    linecolor_list1 = np.linspace(0.8,0.1,len(inputdomain_list)) # Create an array of values to customize line color in RGB

    #=======================================================================================#
    # Plotting
    #=======================================================================================#
    """ Singular values for different window lengths """
    for idx, window_length in enumerate(list_window_length):
        Hx1 = find_trajectory_matrix(X1,L=window_length)
        U1,S1,V1 = np.linalg.svd(Hx1)
        """ Plot singular values """
        index = np.arange(0,len(S1),1)
        cumsum_normalized = np.cumsum(S1)/np.sum(S1)
        ax0[0,0].plot(index,S1,marker='o',markerfacecolor='none',markersize=2,zorder=len(list_window_length)-window_length,linestyle='-',rasterized=True,color=(0+float(linecolor_list0[idx]), 0, 1-float(linecolor_list0[idx])),label=f'$L={window_length}$')
        ax0[0,1].plot(index,cumsum_normalized,marker='o',markerfacecolor='none',markersize=2,zorder=len(list_window_length)-window_length,linestyle='-',rasterized=True,color=(0+float(linecolor_list0[idx]), 0, 1-float(linecolor_list0[idx-1])),label=f'$L={window_length}$')
        """ Zoom into elbow """
        if window_length == 5:
            ax0[1,0].plot(index,S1,marker='o',markerfacecolor='none',markersize=2,zorder=len(list_window_length)+1,linestyle='-',rasterized=True,color='0',label=f'$L={window_length}$')
            ax0[1,1].plot(index,cumsum_normalized,marker='o',markerfacecolor='none',markersize=2,zorder=len(list_window_length)+1,linestyle='-',rasterized=True,color='0',label=f'$L={window_length}$')    
        else:
            ax0[1,0].plot(index,S1,marker='o',markerfacecolor='none',markersize=2,zorder=len(list_window_length)-window_length,linestyle='-',rasterized=True,color=(0+float(linecolor_list0[idx]), 0, 1-float(linecolor_list0[idx])),label=f'$L={window_length}$')
            ax0[1,1].plot(index,cumsum_normalized,marker='o',markerfacecolor='none',markersize=2,zorder=len(list_window_length)-window_length,linestyle='-',rasterized=True,color=(0+float(linecolor_list0[idx]), 0, 1-float(linecolor_list0[idx])),label=f'$L={window_length}$')
   
    # ax0[0,0].set_xlabel(r'$i$', fontsize=9)
    ax0[0,0].set_ylabel(r'Singular value $\sigma_i$', fontsize=9)
    # ax0[0,1].set_xlabel(r'$r$', fontsize=9)
    ax0[0,1].set_ylabel(r'Cumulative sum $\sum_{i=1}^r \sigma_i$', fontsize=9)

    ax0[1,0].set_xlabel(r'$i$', fontsize=9)
    ax0[1,0].set_ylabel(r'Singular value $\sigma_i$', fontsize=9)
    ax0[1,0].set_xlim(-1,10.5)
    ax0[1,0].legend(framealpha=0.5,ncols=3,fontsize=4)
    ax0[1,1].set_xlabel(r'$r$', fontsize=9)
    ax0[1,1].set_ylabel(r'Cumulative sum $\sum_{i=1}^r \sigma_i$', fontsize=9)
    ax0[1,1].set_xlim(-1,10.5)

    """ Singular values for different missions """
    window_length=5
    for i in range(0,len(inputdomain_list)):
        X = inputdomain_list[i]
        Hx = find_trajectory_matrix(X,L=window_length)
        U,S,V = np.linalg.svd(Hx)

        """ Plot singular values """
        index = np.arange(0,len(S),1)
        cumsum_normalized = np.cumsum(S)/np.sum(S)
        ax1[0].plot(index,S,marker='o',markerfacecolor='none',markersize=2,linestyle='-',rasterized=True,color=(0, 0+float(linecolor_list1[i]), 1-float(linecolor_list1[i])),label=shift_list[i])
        ax1[1].plot(index,cumsum_normalized,marker='o',markerfacecolor='none',markersize=2,linestyle='-',rasterized=True,color=(0, 0+float(linecolor_list1[i]), 1-float(linecolor_list1[i])),label=shift_list[i])
        if i == len(inputdomain_list)-1:
            ax1[0].plot(index,S,marker='o',markerfacecolor='none',markersize=2,linestyle='-',rasterized=True,color='0',label=r"DRM, $r_p=100$ km")
            ax1[1].plot(index,cumsum_normalized,marker='o',markerfacecolor='none',markersize=2,linestyle='-',rasterized=True,color='0',label=r"DRM, $r_p=100$ km")
        
    ax1[0].legend(framealpha=0.5,fontsize=4)
    ax1[0].set_xlabel(r'$i$', fontsize=9)
    ax1[0].set_ylabel(r'Singular value $\sigma_i$', fontsize=9)
    ax1[1].set_xlabel(r'$r$', fontsize=9)
    ax1[1].set_ylabel(r'Cumulative sum $\sum_{i=1}^r \sigma_i$', fontsize=9)
    
    # plt.show()
    fig0.savefig('journal_plots/window_length_vs_singular_values.pdf', format='pdf')
    fig1.savefig('journal_plots/missions_vs_singular_values.pdf', format='pdf')