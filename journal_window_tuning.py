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

num_replications = 100

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

    # Read data and add noise from CSV
    noiselevel = 0
    df1, t1, X1, Y1 = read_noisy_data(mission1_data_directory,noiselevel=0)
    df2, t2, X2, Y2 = read_noisy_data(mission2_data_directory,noiselevel)

    #=======================================================================================#
    # Subspace Alignment
    #=======================================================================================#

    # Create empty lists for rmse, r2, and subspace distances 
    # Create empty lists for rmse, r2, and subspace distances 
    list_mean_rmse_og = []
    list_std_rmse_og = []

    list_mean_rmse_da = []
    list_std_rmse_da = []

    list_mean_r2_og = []
    list_std_r2_og = []

    list_mean_r2_da = []
    list_std_r2_da = []

    list_unaligned_d = []
    list_aligned_d = []
    list_numpoints = []

    list_window_length = np.arange(1,51,1)
    for window_length in list_window_length:
        k = window_length # subspace rank
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


        #=======================================================================================#
        # Training Replications
        #=======================================================================================#
        replications_rmse_og = []
        replications_rmse_da = []
        replications_r2_og = []
        replications_r2_da = []

        for replication_index in range(0,num_replications):

            print(f"Replication {replication_index+1}")
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

            Yspred_og = Yspred_torch_og.cpu().detach().numpy()
            Yspred_da = Yspred_torch_da.cpu().detach().numpy()
            Ytpred_og = Ytpred_torch_og.cpu().detach().numpy()
            Ytpred_da = Ytpred_torch_da.cpu().detach().numpy()

            Yspred_da = psa.H_to_TS(Yspred_da.T)
            Ytpred_da = psa.H_to_TS(Ytpred_da.T)

            if replication_index == 0:
                Ys = psa.H_to_TS(Ys.T)
                Yt = psa.H_to_TS(Yt.T)

            rmse_og = root_mean_squared_error(Y2, Ytpred_og)
            rmse_da = root_mean_squared_error(Yt, Ytpred_da)
            r2_og = r2_score(Y2, Ytpred_og)
            r2_da = r2_score(Yt, Ytpred_da)

            replications_rmse_og.append(rmse_og)
            replications_rmse_da.append(rmse_da)
            replications_r2_og.append(r2_og)
            replications_r2_da.append(r2_da)

        """ Compute mean and confidence interval of replications """
        mean_rmse_og = np.mean(replications_rmse_og)
        std_rmse_og = np.std(replications_rmse_og)

        mean_rmse_da = np.mean(replications_rmse_da)
        std_rmse_da = np.std(replications_rmse_da)

        mean_r2_og = np.mean(replications_r2_og)
        std_r2_og = np.std(replications_r2_og)

        mean_r2_da = np.mean(replications_r2_da)
        std_r2_da = np.std(replications_r2_da)

        """ Collect RMSE, R2, and subspace distances """
        list_mean_rmse_og.append(mean_rmse_og)
        list_std_rmse_og.append(3*std_rmse_og)

        list_mean_rmse_da.append(mean_rmse_da)
        list_std_rmse_da.append(3*std_rmse_da)

        list_mean_r2_og.append(mean_r2_og)
        list_std_r2_og.append(3*std_r2_og)

        list_mean_r2_da.append(mean_r2_da)
        list_std_r2_da.append(3*std_r2_da)

        list_unaligned_d.append(linalg.norm(Hx_proj - Hz_proj))
        list_aligned_d.append(linalg.norm(Hx_proj_aligned - Hz_proj))


    #=======================================================================================#
    # Plotting
    #=======================================================================================#
    """ Metrics """
    fig6, ax6 = plt.subplots(1,1,figsize=(3.25, 2.75),layout='constrained') # RMSE vs mission number
    ax6.errorbar(list_window_length,list_mean_rmse_da,yerr=list_std_rmse_da,marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='darkorchid',label='Domain adapted')
    ax6.set_xlabel('Window Length $L$')
    ax6.set_ylabel('RMSE')
    # ax6.legend(framealpha=0.5,fontsize=6.25)

    print(f"Mean RMSE values: {mean_rmse_og}")
    print(f"3sigma on RMSE values: {std_rmse_da}")
    print(f"Mean R2 values: {mean_r2_da}")
    print(f"3sigma on R2 values: {std_r2_da}")
    print(f"Unaligned d: {list_unaligned_d}")
    print(f"Aligned d: {list_aligned_d}")

    fig6.savefig('journal_plots/rmse_vs_window_length.pdf', format='pdf')