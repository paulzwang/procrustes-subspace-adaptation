import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use(r'matplotlib_stylesheet\journal_nolatex.mplstyle')
import utils
from utils import NeuralNetwork
from utils import train_model
import psa
import torch 
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import scipy.linalg as linalg

# Set training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
interptype = 'time'

def read_data(data_directory):
    df = pd.read_csv(data_directory).drop_duplicates(subset=['time'], keep='first')
    t = np.array(df['time'])
    X = np.concatenate((np.array(df['rho']).reshape(-1,1),np.array(df['T']).reshape(-1,1),np.array(df['S']).reshape(-1,1)),axis=1)
    Y = np.array(df['heat_rate']).reshape(-1,1)
    return df, t, X, Y

def partition_data(percent_seen,t1,X1,Y1,t2,X2,Y2):
    index_seen2 = round(percent_seen*(t2.shape[0]-1))
    index_seen1 = np.argmin(np.abs(t1 - t2[index_seen2])) # Get time of percent seen
    # Split the data into seen/unseen
    t2_seen = t2[0:index_seen2]
    t2_unseen = t2[index_seen2+1:]
    X2_seen = X2[0:index_seen2,:]
    X2_unseen = X2[index_seen2+1:]
    Y2_seen = Y2[0:index_seen2,:]
    Y2_unseen = Y2[index_seen2+1:,:]
    
    t1_seen = t1[0:index_seen1]
    t1_unseen = t1[index_seen1+1:]
    X1_seen = X1[0:index_seen1,:]
    X1_unseen = X1[index_seen1+1:]
    Y1_seen = Y1[0:index_seen1,:]
    Y1_unseen = Y1[index_seen1+1:,:]
    data_dict = {'t1_seen':t1_seen,'t1_unseen':t1_unseen,'X1_seen':X1_seen,'X1_unseen':X1_unseen,'Y1_seen':Y1_seen,'Y1_unseen':Y1_unseen,
                 't2_seen':t2_seen,'t2_unseen':t2_unseen,'X2_seen':X2_seen,'X2_unseen':X2_unseen,'Y2_seen':Y2_seen,'Y2_unseen':Y2_unseen}
    return data_dict

def time_delay_embedding(X,L):
    # Inputs are in the column space of X
    for i in range(0,np.size(X,1)): # Iterate across columns of X
        Hxi = psa.find_trajectory_matrix(X[:,i],L)
        # Stack Hankel matrices
        if i==0:
            Hx = Hxi
        else:
            Hx = np.concatenate((Hx,Hxi),axis=0)
    return Hx

if __name__ == '__main__':
    # rp is different between missions
    mission1_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=100.0\Results_ctrl=0_ra=12000_rp=100.0_hl=0.150_90.0deg.csv'
    mission2_data_directory = r'data\periapsis_shift\4orbit_ra=12000_rp=96.0\Results_ctrl=0_ra=12000_rp=96.0_hl=0.150_90.0deg.csv'
    # mission2_data_directory = r'data\fixed_semimajor_eccentricity_shift\4orbit_ra=12004_rp=95.3\Results_ctrl=0_ra=12004_rp=95.3_hl=0.150_90.0deg.csv'
    # mission2_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=11498_rp=100.0\Results_ctrl=0_ra=11498_rp=100.0_hl=0.150_90.0deg.csv'
    # mission2_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=10578_rp=100.1\Results_ctrl=0_ra=10578_rp=100.1_hl=0.150_90.0deg.csv'

    df1, t1, X1, Y1 = read_data(mission1_data_directory)
    df2, t2, X2, Y2 = read_data(mission2_data_directory)

    # Specify up to which data point is seen by the model
    percent_seen = 0.41
    list_rmse_og = []
    list_rmse_da = []
    list_r2_og = []
    list_r2_da = []
    list_unaligned_d = []
    list_aligned_d = []
    list_numpoints= []


    #=======================================================================================#
    # Partition data into streamed and unseen data
    #=======================================================================================#
    data_dict = partition_data(percent_seen=percent_seen,t1=t1,X1=X1,Y1=Y1,t2=t2,X2=X2,Y2=Y2)
    t1_seen = data_dict['t1_seen']
    t1_unseen = data_dict['t1_unseen']
    X1_seen = data_dict['X1_seen']
    X1_unseen = data_dict['X1_unseen']
    Y1_seen = data_dict['Y1_seen']
    Y1_unseen = data_dict['Y1_unseen']
    t2_seen = data_dict['t2_seen']
    t2_unseen = data_dict['t2_unseen']
    X2_seen = data_dict['X2_seen']
    X2_unseen = data_dict['X2_unseen']
    Y2_seen = data_dict['Y2_seen']
    Y2_unseen = data_dict['Y2_unseen']

    #=======================================================================================#
    # Subspace Alignment
    #=======================================================================================#
    # Xa, Za are derived from partially observed X1 and X2. Ys and Yt are Hankelised, partially observed Y1 and Y2
    L = 5
    Xa, Za, Ys, Yt, Hx_proj, Hz_proj, Hx_proj_aligned, Hz_sub = psa.streaming_procrustes_subspace_adaptation(X1_seen,X2_seen,Y1_seen,Y2_seen,t1_seen,t2_seen,
                                                                        window_length=L,k=5,interptype=interptype,manifold_visual=False)

    #=======================================================================================#
    # Data Preprocessing
    #=======================================================================================#
    # Scale all inputs
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.fit_transform(X2)
    X1_seen = scaler.fit_transform(X1_seen)

    # Find trajectory matrices of X1 and X2
    Hx1 = time_delay_embedding(X1,L=L)
    Hx2 = time_delay_embedding(X2,L=L)
    Hx1_torch = torch.tensor(Hx1,dtype=torch.float32).to(device)
    Hx2_torch = torch.tensor(Hx2,dtype=torch.float32).to(device)

    # Project Hx2 onto operational subspace
    Hx2_proj = (Hz_sub.T @ Hx2).T
    Hx2_proj_torch = torch.tensor(Hx2_proj,dtype=torch.float32).to(device)

    # Create copies of inputs as torch tensors
    X1_torch = torch.tensor(X1,dtype=torch.float32).to(device)
    X2_torch = torch.tensor(X2,dtype=torch.float32).to(device)
    Y1_torch = torch.tensor(Y1,dtype=torch.float32).to(device)
    Y2_torch = torch.tensor(Y2,dtype=torch.float32).to(device)

    X1_seen_torch = torch.tensor(X1_seen,dtype=torch.float32).to(device)
    Y1_seen_torch = torch.tensor(Y1_seen,dtype=torch.float32).to(device)

    Xa_torch = torch.tensor(Xa,dtype=torch.float32).to(device)
    Za_torch = torch.tensor(Za,dtype=torch.float32).to(device)
    Ys_torch = torch.tensor(Ys,dtype=torch.float32).to(device)
    Yt_torch = torch.tensor(Yt,dtype=torch.float32).to(device)

    #=======================================================================================#
    # Model Training
    #=======================================================================================#
    
    # Define neural network hyperparameters
    num_layers = int(4)#int(4)
    num_neurons = int(200)#int(200)
    hidden_sizes = [num_neurons] * num_layers
    learning_rate = 0.006
    num_epochs = 200 

    model_og = NeuralNetwork(input_size=X1_torch.size(1), hidden_sizes=hidden_sizes, output_size=Y1_torch.size(1)).to(device)
    model_da = NeuralNetwork(input_size=Xa_torch.size(1), hidden_sizes=hidden_sizes, output_size=Ys_torch.size(1)).to(device)

    loss_og = train_model(model_og,X1_torch,Y1_torch,num_epochs,learning_rate)
    loss_da = train_model(model_da,Xa_torch,Ys_torch,num_epochs,learning_rate,weight_decay=0) # Note the change in L2 regularizer


    #=======================================================================================#
    # Model Evaluation
    #=======================================================================================#
    model_og.eval()
    model_da.eval()
    with torch.no_grad():
        Ytpred_og_torch = model_og(X2_torch)
        Ytpred_da_torch = model_da(Hx2_proj_torch)

    Ytpred_og = Ytpred_og_torch.cpu().detach().numpy()
    Ytpred_da = Ytpred_da_torch.cpu().detach().numpy()

    Ytpred_og = psa.H_to_TS(Ytpred_og.T)
    Ytpred_da = psa.H_to_TS(Ytpred_da.T)


    #=======================================================================================#
    # Metrics
    #=======================================================================================#
    rmse_og = root_mean_squared_error(Y2, Ytpred_og)
    rmse_da = root_mean_squared_error(Y2, Ytpred_da)
    r2_og = r2_score(Y2, Ytpred_og)
    r2_da = r2_score(Y2, Ytpred_da)
    print(f'Baseline RMSE: {rmse_og}')
    print(f'Domain adapted RMSE: {rmse_da}')

    """ Collect RMSE, R2, and subspace distances """
    list_rmse_og.append(rmse_og)
    list_rmse_da.append(rmse_da)
    list_r2_og.append(r2_og)
    list_r2_da.append(r2_da)
    list_unaligned_d.append(linalg.norm(Hx_proj - Hz_proj))
    list_aligned_d.append(linalg.norm(Hx_proj_aligned - Hz_proj))
    list_numpoints.append(X1_seen.shape[0])



    #=======================================================================================#
    # Plotting
    #=======================================================================================#
    fig0, ax0 = plt.subplots(1,2,subplot_kw=dict(projection='3d'),figsize=(6,3),layout='constrained') # manifolds
    scolors = Hx_proj[2,:] #np.linspace(0,Hx_proj.shape[1],num=Hx_proj.shape[1])
    tcolors = Hz_proj[2,:] #np.linspace(0,Hz_proj.shape[1],num=Hz_proj.shape[1])
    ax0[0].scatter(Hx_proj[0,:],Hx_proj[1,:],Hx_proj[2,:],s=4,marker='.',c=scolors,cmap='viridis',label='$H_{X,\mathrm{proj}}$',rasterized=True,depthshade=False)
    ax0[0].scatter(Hz_proj[0,:],Hz_proj[1,:],Hz_proj[2,:],s=4,marker='.',c=tcolors,cmap='plasma',label='$H_{Z,\mathrm{proj}}$',rasterized=True,depthshade=False)
    ax0[0].legend(loc='upper left',framealpha=0.5)
    ax0[0].tick_params(pad=-5)

    scolors = Xa[:,2] #np.linspace(0,Xa.shape[0],num=Xa.shape[0])
    tcolors = Za[:,2] #np.linspace(0,Za.shape[0],num=Za.shape[0])
    ax0[1].scatter(Xa[:,0],Xa[:,1],Xa[:,2],s=4,marker='.',c=scolors,cmap='viridis',label='$X_a$',rasterized=True,depthshade=False)
    ax0[1].scatter(Za[:,0],Za[:,1],Za[:,2],s=4,marker='.',c=tcolors,cmap='plasma',label='$Z_a$',rasterized=True,depthshade=False)
    ax0[1].legend(loc='upper left',framealpha=0.5)
    ax0[1].tick_params(pad=-5)
    plt.show()

    fig1, ax1 = plt.subplots(1,2,width_ratios=[0.4,1],layout='constrained')
    # Target Domain
    ax1[0].scatter(Ytpred_og,Y2,s=2,label=f"RMSE: {round(rmse_og,4)}",color='red', rasterized=True)
    ax1[0].scatter(Ytpred_da,Y2,s=2,label=f"RMSE: {round(rmse_da,4)}",color='darkorchid', rasterized=True)
    # Plot line y=x, the ideal predicted vs. actual curve
    lims = [
        np.min([ax1[0].get_xlim(), ax1[0].get_ylim()]),  # min of both axes
        np.max([ax1[0].get_xlim(), ax1[0].get_ylim()]),  # max1 of both axes
    ]
    ax1[0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax1[0].set_aspect('equal')
    ax1[0].set_xlim(lims)
    ax1[0].set_ylim(lims)
    ax1[0].set_xlabel('Predicted Output')
    ax1[0].set_ylabel('Actual Output') 
    ax1[0].legend(fontsize=6.25,framealpha=0.5)

    ax1[1].plot(Y2,label="Operational actual",color='gray')
    ax1[1].plot(Ytpred_og,label="No adaptation",color='red')
    ax1[1].plot(Ytpred_da,label="Domain adapted",color='darkorchid')
    ax1[1].set_xlabel('Time Step')
    ax1[1].set_ylabel('Heat Rate (W/cm$^2$)')
    ax1[1].legend(fontsize=6.25,framealpha=0.5)