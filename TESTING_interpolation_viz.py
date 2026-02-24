import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(r'matplotlib_stylesheet\journal_nolatex.mplstyle')
import TESTING_psa as psa

from scipy.interpolate import make_interp_spline

# interptype = 'time'
interptype = 'removal'

def read_data(data_directory):
    df = pd.read_csv(data_directory).drop_duplicates(subset=['time'], keep='first')
    t = np.array(df['time'])
    X = np.concatenate((np.array(df['rho']).reshape(-1,1),np.array(df['T']).reshape(-1,1),np.array(df['S']).reshape(-1,1)),axis=1)
    Y = np.array(df['heat_rate']).reshape(-1,1)
    return df, t, X, Y

def time_interpolate(X,Z,ts,tt):
    """ Common Time Interpolation Scheme """
    big_time = np.unique(np.concatenate((ts,tt),axis=0))
    bs = make_interp_spline(ts,X)
    bt = make_interp_spline(tt,Z)
    X_interp = bs(big_time)
    Z_interp = bt(big_time)

    return X_interp, Z_interp, big_time

def removal_interpolate(X,Z,ts,tt):
    """ Removal of Zeros Interpolation Scheme """
    numpoints_remove = abs(len(ts)-len(tt))
    if len(ts) > len(tt):
        # Remove zeros from X
        idx = np.arange(X[:,0].shape[0])
        idx_zeros = idx[X[:,0] <= 0]
        indices_remove = np.linspace(idx_zeros[0],idx_zeros[-1],numpoints_remove,dtype=np.int16)
        X_interp = np.delete(X,indices_remove,axis=0)
        Z_interp = Z
        ts_interp = np.delete(ts,indices_remove)
        tt_interp = tt
    elif len(ts) < len(tt):
        # Remove zeros from Z
        idx = np.arange(Z[:,0].shape[0])
        idx_zeros = idx[Z[:,0] <= 0]
        indices_remove = np.linspace(idx_zeros[0],idx_zeros[-1],numpoints_remove,dtype=np.int16)
        X_interp = X
        Z_interp = np.delete(Z,indices_remove,axis=0)
        ts_interp = ts
        tt_interp = np.delete(tt,indices_remove,axis=0)
    else:
        X_interp = X
        Z_interp = Z
    
    return X_interp, Z_interp, ts_interp, tt_interp

def plot_interpolation(X_t,Z_t,big_time,X_r,Z_r,ts_interp,tt_interp):
    markersize = 2
    linewidth = 0.01
    fig, ax = plt.subplots(2,3,figsize=(6,6),layout='constrained')
    ax[0,0].plot(big_time, X_t[:,0],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Source $\rho$',rasterized=True)
    ax[0,0].plot(big_time, Z_t[:,0],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Target $\rho$',rasterized=True)
    ax[0,1].plot(big_time, X_t[:,1],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Source $T$',rasterized=True)
    ax[0,1].plot(big_time, Z_t[:,1],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Target $T$',rasterized=True)
    ax[0,2].plot(big_time, X_t[:,2],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Source $S$',rasterized=True)
    ax[0,2].plot(big_time, Z_t[:,2],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Target $S$',rasterized=True)

    ax[1,0].plot(ts_interp, X_r[:,0],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Source $\rho$',rasterized=True)
    ax[1,0].plot(tt_interp, Z_r[:,0],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Target $\rho$',rasterized=True)
    ax[1,1].plot(ts_interp, X_r[:,1],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Source $T$',rasterized=True)
    ax[1,1].plot(tt_interp, Z_r[:,1],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Target $T$',rasterized=True)
    ax[1,2].plot(ts_interp, X_r[:,2],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Source $S$',rasterized=True)
    ax[1,2].plot(tt_interp, Z_r[:,2],marker='o',markerfacecolor='none',linestyle='-',linewidth=linewidth,markersize=markersize,label=r'Target $S$',rasterized=True)

    for i in range(2):
        for j in range(3):
            ax[i,j].set_xlim(left=9500, right=11000)
            ax[i,j].legend(framealpha=0.5)

    return fig

if __name__ == '__main__':
    # Semimajor axis "a" is fixed. Eccentricity "e" is different between missions
    drm = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=12000_rp=100.0\Results_ctrl=0_ra=12000_rp=100.0_hl=0.150_90.0deg.csv'

    shift_rp_mission = r'data\periapsis_shift\4orbit_ra=12000_rp=97.0\Results_ctrl=0_ra=12000_rp=97.0_hl=0.150_90.0deg.csv'
    fix_a_mission = r'data\fixed_semimajor_eccentricity_shift\4orbit_ra=12003_rp=96.8\Results_ctrl=0_ra=12003_rp=96.8_hl=0.150_90.0deg.csv'
    fix_p_mission = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=11667_rp=100.0\Results_ctrl=0_ra=11667_rp=100.0_hl=0.150_90.0deg.csv'

    #=======================================================================================#
    # Data Preprocessing
    #=======================================================================================#
    df1, t1, X1, Y1 = read_data(drm)
    df2, t2, X2, Y2 = read_data(shift_rp_mission)
    df3, t3, X3, Y3 = read_data(fix_a_mission)
    df4, t4, X4, Y4 = read_data(fix_p_mission)
    X_list = [X2, X3, X4]
    t_list = [t2, t3, t4]
    fig_list = []

    for t, X in zip(t_list,X_list):
        X_t, Z_t, big_time = time_interpolate(X1,X,t1,t)
        X_r, Z_r, ts_interp, tt_interp = removal_interpolate(X1,X,t1,t)
        fig = plot_interpolation(X_t,Z_t,big_time,X_r,Z_r,ts_interp,tt_interp)
        fig_list.append(fig)
        # fig.savefig('TESTING_rp_viz.pdf', format='pdf')
    plt.show()