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
from sklearn.preprocessing import StandardScaler

# Set training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_data(data_directory):
    df = pd.read_csv(data_directory).drop_duplicates(subset=['time'], keep='first')
    t = np.array(df['time'])
    X = np.concatenate((np.array(df['rho']).reshape(-1,1),np.array(df['T']).reshape(-1,1),np.array(df['S']).reshape(-1,1)),axis=1)
    Y = np.array(df['heat_rate']).reshape(-1,1)
    return df, t, X, Y

def partition_data(percent_seen,t,X,Y):
    index_seen = round(percent_seen*t.shape[0])
    # Split the data into seen/unseen
    t_seen = t[0:index_seen]
    t_unseen = t[index_seen+1:]
    X_seen = X[0:index_seen,:]
    X_unseen = X[index_seen+1:]
    Y_seen = Y[0:index_seen,:]
    Y_unseen = Y[index_seen+1:,:]
    return t_seen, t_unseen, X_seen, X_unseen, Y_seen, Y_unseen

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

    df1, t1, X1, Y1 = read_data(mission1_data_directory)
    df2, t2, X2, Y2 = read_data(mission2_data_directory)

    # Specify up to which data point is seen by the model
    percent_seen = 1

    #=======================================================================================#
    # Partition data into streamed and unseen data
    #=======================================================================================#
    t1_seen, t1_unseen, X1_seen, X1_unseen, Y1_seen, Y1_unseen = partition_data(percent_seen=percent_seen,t=t1,X=X1,Y=Y1)
    t2_seen, t2_unseen, X2_seen, X2_unseen, Y2_seen, Y2_unseen = partition_data(percent_seen=percent_seen,t=t2,X=X2,Y=Y2)

    #=======================================================================================#
    # Subspace Alignment
    #=======================================================================================#
    # Xa, Za are derived from partially observed X1 and X2. Ys and Yt are Hankelised, partially observed Y1 and Y2
    L = 5
    Xa, Za, Ys, Yt, _, _, _, Hz_sub = psa.streaming_procrustes_subspace_adaptation(X1_seen,X2_seen,Y1_seen,Y2_seen,t1_seen,t2_seen,
                                                                           window_length=L,k=5,interptype='time',manifold_visual=False)

    #=======================================================================================#
    # Data Preprocessing
    #=======================================================================================#
    # Scale all inputs
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X1)
    X2 = scaler.fit_transform(X2)

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

    model_og.eval()
    model_da.eval()
    with torch.no_grad():
        Ytpred_og_torch = model_og(X2_torch)
        Ytpred_da_torch = model_da(Hx2_proj_torch)

    Ytpred_og = Ytpred_og_torch.cpu().detach().numpy()
    Ytpred_da = Ytpred_da_torch.cpu().detach().numpy()

    Ytpred_og = psa.H_to_TS(Ytpred_og.T)
    Ytpred_da = psa.H_to_TS(Ytpred_da.T)

    rmse_og = root_mean_squared_error(Y2, Ytpred_og)
    rmse_da = root_mean_squared_error(Y2, Ytpred_da)

    print(f'Baseline RMSE: {rmse_og}')
    print(f'Domain adapted RMSE: {rmse_da}')



    #=======================================================================================#
    # Plotting
    #=======================================================================================#
    fig1, ax1 = plt.subplots(1,2,width_ratios=[0.4,1])
    plt.subplots_adjust(wspace=0.25)
    
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
    plt.show()