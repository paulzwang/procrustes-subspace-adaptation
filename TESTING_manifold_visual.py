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

def read_noisy_data(data_directory,noiselevel):
    df = pd.read_csv(data_directory).drop_duplicates(subset=['time'], keep='first')
    t = np.array(df['time'])

    noisy_rho = add_percent_noise(np.array(df['rho']).reshape(-1,1),percent_noise=noiselevel)
    noisy_T = add_percent_noise(np.array(df['T']).reshape(-1,1),percent_noise=noiselevel)
    noisy_S = add_percent_noise(np.array(df['S']).reshape(-1,1),percent_noise=noiselevel)

    X = np.concatenate((noisy_rho,noisy_T,noisy_S),axis=1)
    Y = np.array(df['heat_rate']).reshape(-1,1)
    return df, t, X, Y


if __name__ == '__main__':
    mission1_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=100.0\Results_ctrl=0_ra=12000_rp=100.0_hl=0.150_90.0deg.csv'
    mission2_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=99.0\Results_ctrl=0_ra=12000_rp=99.0_hl=0.150_90.0deg.csv'
    # mission2_data_directory = r'data\fixed_semimajor_eccentricity_shift\4orbit_ra=12009_rp=90.8\Results_ctrl=0_ra=12009_rp=90.8_hl=0.150_90.0deg.csv'
    # mission2_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=11498_rp=100.0\Results_ctrl=0_ra=11498_rp=100.0_hl=0.150_90.0deg.csv'
    # mission2_data_directory = r'data\periapsis_shift\8orbit_ra=12000_rp=100.0\Results_ctrl=0_ra=12000_rp=100.0_hl=0.150_90.0deg.csv'

    # Read data and add noise from CSV
    noiselevel = 0
    df1, t1, X1, Y1 = read_noisy_data(mission1_data_directory,noiselevel=0)
    df2, t2, X2, Y2 = read_noisy_data(mission2_data_directory,noiselevel)

    #=======================================================================================#
    # Subspace Alignment
    #=======================================================================================#
    window_length = 20 # 50
    k = 5 # subspace rank
    Xa, Za, Ys, Yt, Hx_proj, Hz_proj, Hx_proj_aligned, Hz_sub = psa.streaming_procrustes_subspace_adaptation(X1,X2,Y1,Y2,t1,t2,
                                                                                                                window_length,k,interptype='removal')

    #=======================================================================================#
    # Data Preprocessing
    #=======================================================================================#
    # Set training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.fit_transform(X2)
    X1_torch = torch.tensor(X1,dtype=torch.float32).to(device)
    X2_torch = torch.tensor(X2,dtype=torch.float32).to(device)
    Y1_torch = torch.tensor(Y1,dtype=torch.float32).to(device)
    Y2_torch = torch.tensor(Y2,dtype=torch.float32).to(device)

    Xa_torch = torch.tensor(Xa,dtype=torch.float32).to(device)
    Za_torch = torch.tensor(Za,dtype=torch.float32).to(device)
    Ys_torch = torch.tensor(Ys,dtype=torch.float32).to(device)
    Yt_torch = torch.tensor(Yt,dtype=torch.float32).to(device)

    #=======================================================================================#
    # Model Training
    #=======================================================================================#
    # Define neural network hyperparameters
    num_layers = int(4) # num_layers = int(4)
    num_neurons = int(200)#int(145)
    hidden_sizes = [num_neurons] * num_layers
    learning_rate = 0.006#0.00180
    num_epochs = 200 #200

    model_og = NeuralNetwork(input_size=X1_torch.size(1), hidden_sizes=hidden_sizes, output_size=Y1_torch.size(1)).to(device)
    model_da = NeuralNetwork(input_size=Xa_torch.size(1), hidden_sizes=hidden_sizes, output_size=Ys_torch.size(1)).to(device)

    loss_og = train_model(model_og,X1_torch,Y1_torch,num_epochs,learning_rate)
    loss_da = train_model(model_da,Xa_torch,Ys_torch,num_epochs,learning_rate,weight_decay=0)

    model_og.eval()
    model_da.eval()
    with torch.no_grad():
        Yspred_torch_og = model_og(X1_torch)
        Yspred_torch_da = model_da(Xa_torch)
        Ytpred_torch_og = model_og(X2_torch)
        Ytpred_torch_da = model_da(Za_torch)

    Yspred_og = Yspred_torch_og.cpu().detach().numpy() # Save to local memory (.cpu), convert to numpy array (.detach.numpy), and convert to scalar value (.item)
    Yspred_da = Yspred_torch_da.cpu().detach().numpy()
    Ytpred_og = Ytpred_torch_og.cpu().detach().numpy()
    Ytpred_da = Ytpred_torch_da.cpu().detach().numpy()

    Yspred_da = psa.H_to_TS(Yspred_da.T)
    Ytpred_da = psa.H_to_TS(Ytpred_da.T)

    rmse_og = root_mean_squared_error(Y2, Ytpred_og)
    rmse_da = root_mean_squared_error(Y2, Ytpred_da)
    r2_og = r2_score(Y2, Ytpred_og)
    r2_da = r2_score(Y2, Ytpred_da)


    #=======================================================================================#
    # Plotting
    #=======================================================================================#
    legend_fontsize = 6.25
    
    mosaic = [["A", "B", "C"]]
    fig, ax1 = plt.subplot_mosaic(mosaic, per_subplot_kw={"B": {"projection": "3d"},"C": {"projection": "3d"}},figsize=(6,3),width_ratios=[0.8,1,1],layout='constrained')
    # Subplot 'B' and 'C' are 3D, while 'A' is 2D.

    # Target Domain
    ax1["A"].scatter(Ytpred_og,Y2,s=2,label=f"RMSE: {round(rmse_og,4)}",color='red', rasterized=True)
    ax1["A"].scatter(Ytpred_da,Y2,s=2,label=f"RMSE: {round(rmse_da,4)}",color='darkorchid', rasterized=True)
    # Plot line y=x, the ideal predicted vs. actual curve
    lims = [
        np.min([ax1["A"].get_xlim(), ax1["A"].get_ylim()]),  # min of both axes
        np.max([ax1["A"].get_xlim(), ax1["A"].get_ylim()]),  # max1 of both axes
    ]
    ax1["A"].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax1["A"].set_aspect('equal')
    ax1["A"].set_xlim(lims)
    ax1["A"].set_ylim(lims)
    ax1["A"].set_xlabel('Predicted Output')
    ax1["A"].set_ylabel('Actual Output') 
    ax1["A"].legend(fontsize=6.25,framealpha=0.5)

    """ Non-Interpolated Manifolds """
    scolors = Hx_proj[2,:] #np.linspace(0,Hx_proj.shape[1],num=Hx_proj.shape[1])
    tcolors = Za[:,2]
    ax1["B"].scatter(Hx_proj[0,:],Hx_proj[1,:],Hx_proj[2,:],s=4,marker='.',c=scolors,cmap='viridis',label='$H_{X,\mathrm{proj}}$',rasterized=True,depthshade=False)
    ax1["B"].scatter(Za[:,0],Za[:,1],Za[:,2],s=4,marker='.',c=tcolors,cmap='plasma',label='$H_{Z,\mathrm{proj}}$',rasterized=True,depthshade=False)
    ax1["B"].legend(loc='upper left',framealpha=0.5,fontsize=legend_fontsize)
    ax1["B"].tick_params(pad=-5)

    ax1["C"].scatter(Hx_proj_aligned[0,:],Hx_proj_aligned[1,:],Hx_proj_aligned[2,:],s=4,marker='.',c=scolors,cmap='viridis',label='Aligned $H_{X,\mathrm{proj}}$ ($X_a$)',rasterized=True,depthshade=False)
    ax1["C"].scatter(Za[:,0],Za[:,1],Za[:,2],s=4,marker='.',c=tcolors,cmap='plasma',label='$H_{Z,\mathrm{proj}}$ ($Z_a$)',rasterized=True,depthshade=False)
    ax1["C"].legend(loc='upper left',framealpha=0.5,fontsize=legend_fontsize)
    ax1["C"].tick_params(pad=-5)
    plt.show()