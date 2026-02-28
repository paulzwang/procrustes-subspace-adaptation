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
interptype = 'removal'

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
    mission1_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=12000_rp=100.0\Results_ctrl=0_ra=12000_rp=100.0_hl=0.150_90.0deg.csv'
    mission2_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=11667_rp=100.0\Results_ctrl=0_ra=11667_rp=100.0_hl=0.150_90.0deg.csv'
    mission3_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=11498_rp=100.0\Results_ctrl=0_ra=11498_rp=100.0_hl=0.150_90.0deg.csv'
    mission4_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=11334_rp=100.0\Results_ctrl=0_ra=11334_rp=100.0_hl=0.150_90.0deg.csv'
    mission5_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=11174_rp=100.1\Results_ctrl=0_ra=11174_rp=100.1_hl=0.150_90.0deg.csv'
    mission6_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=11019_rp=100.1\Results_ctrl=0_ra=11019_rp=100.1_hl=0.150_90.0deg.csv'
    mission7_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=10868_rp=100.1\Results_ctrl=0_ra=10868_rp=100.1_hl=0.150_90.0deg.csv'
    mission8_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=10721_rp=100.1\Results_ctrl=0_ra=10721_rp=100.1_hl=0.150_90.0deg.csv'
    mission9_data_directory = r'data\fixed_semilatus_eccentricity_shift\4orbit_ra=10578_rp=100.1\Results_ctrl=0_ra=10578_rp=100.1_hl=0.150_90.0deg.csv'

    shift_list = [r'$e=0.98125$ km', 
                  r'$e=0.98150$ km', 
                  r'$e=0.98175$ km', 
                  r'$e=0.98200$ km', 
                  r'$e=0.98225$ km', 
                  r'$e=0.98250$ km', 
                  r'$e=0.98275$ km', 
                  r'$e=0.98300$ km'] # Model training performed on descending shifts

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

    inputdomain_list = [X9, X8, X7, X6, X5, X4, X3, X2]
    outputdomain_list = [Y9, Y8, Y7, Y6, Y5, Y4, Y3, Y2]
    t_list = [t9, t8, t7, t6, t5, t4, t3, t2]

    #=======================================================================================#
    # Instantiate Plotting
    #=======================================================================================#
    label_fontsize = 9
    fig0, ax0 = plt.subplots(len(inputdomain_list),3,figsize=(6,8),layout='constrained')

    #=======================================================================================#
    # Training Loop
    #=======================================================================================#
    # Iterate through datasets
    for i in range(0,len(inputdomain_list)):
        X_fromlist = inputdomain_list[i]
        Y_fromlist = outputdomain_list[i]
        t_fromlist = t_list[i]

        # Specify up to which data point is seen by the model
        list_percent_seen = np.arange(0.02,1.02,0.02)
        list_rmse_og = []
        list_rmse_da = []
        list_r2_og = []
        list_r2_da = []
        list_unaligned_d = []
        list_aligned_d = []
        list_numpoints= []

        
        for idx,percent_seen in enumerate(list_percent_seen):
            #=======================================================================================#
            # Partition data into streamed and unseen data
            #=======================================================================================#
            t1_seen, t1_unseen, X1_seen, X1_unseen, Y1_seen, Y1_unseen = partition_data(percent_seen=percent_seen,t=t1,X=X1,Y=Y1)
            t2_seen, t2_unseen, X2_seen, X2_unseen, Y2_seen, Y2_unseen = partition_data(percent_seen=percent_seen,t=t_fromlist,X=X_fromlist,Y=Y_fromlist)

            #=======================================================================================#
            # Subspace Alignment
            #=======================================================================================#
            # Xa, Za are derived from partially observed X1 and X_fromlist. Ys and Yt are Hankelised, partially observed Y1 and Y_fromlist
            L = 5
            Xa, Za, Ys, Yt, Hx_proj, Hz_proj, Hx_proj_aligned, Hz_sub = psa.streaming_procrustes_subspace_adaptation(X1_seen,X2_seen,Y1_seen,Y2_seen,t1_seen,t2_seen,
                                                                                window_length=L,k=5,interptype=interptype,manifold_visual=False)

            #=======================================================================================#
            # Data Preprocessing
            #=======================================================================================#
            # Scale all inputs
            scaler = StandardScaler()
            X1 = scaler.fit_transform(X1)
            X_fromlist = scaler.fit_transform(X_fromlist)
            X1_seen = scaler.fit_transform(X1_seen)

            # Find trajectory matrices of X1 and X_fromlist
            Hx1 = time_delay_embedding(X1,L=L)
            Hx2 = time_delay_embedding(X_fromlist,L=L)
            Hx1_torch = torch.tensor(Hx1,dtype=torch.float32).to(device)
            Hx2_torch = torch.tensor(Hx2,dtype=torch.float32).to(device)

            # Project Hx2 onto operational subspace
            Hx2_proj = (Hz_sub.T @ Hx2).T
            Hx2_proj_torch = torch.tensor(Hx2_proj,dtype=torch.float32).to(device)

            # Create copies of inputs as torch tensors
            X1_torch = torch.tensor(X1,dtype=torch.float32).to(device)
            X2_torch = torch.tensor(X_fromlist,dtype=torch.float32).to(device)
            Y1_torch = torch.tensor(Y1,dtype=torch.float32).to(device)
            Y2_torch = torch.tensor(Y_fromlist,dtype=torch.float32).to(device)

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

            if idx == 0: # Train the baseline model only once. It cannot train on the operational data because it does not have access to Y2
                model_og = NeuralNetwork(input_size=X1_torch.size(1), hidden_sizes=hidden_sizes, output_size=Y1_torch.size(1)).to(device)            
                loss_og = train_model(model_og,X1_torch,Y1_torch,num_epochs,learning_rate)
            model_da = NeuralNetwork(input_size=Xa_torch.size(1), hidden_sizes=hidden_sizes, output_size=Ys_torch.size(1)).to(device)
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
            rmse_og = root_mean_squared_error(Y_fromlist, Ytpred_og)
            rmse_da = root_mean_squared_error(Y_fromlist, Ytpred_da)
            r2_og = r2_score(Y_fromlist, Ytpred_og)
            r2_da = r2_score(Y_fromlist, Ytpred_da)
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
        """ Subspace Distances, RMSE, and R2 """
        list_percent_seen = 100*list_percent_seen
        ax0[i,0].plot(list_percent_seen,np.array(list_unaligned_d)/np.array(list_numpoints),marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='maroon',label=r'Pre-adaptation $d/n$')
        ax0[i,0].plot(list_percent_seen,np.array(list_aligned_d)/np.array(list_numpoints),marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='navy',label=r'Post-adaptation $d/n$')
        ax0[i,1].plot(list_percent_seen,list_rmse_og,marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='red',label='No adaptation')
        ax0[i,1].plot(list_percent_seen,list_rmse_da,marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='darkorchid',label='Domain adapted')
        ax0[i,2].plot(list_percent_seen,list_r2_og,marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='red',label='No adaptation')
        ax0[i,2].plot(list_percent_seen,list_r2_da,marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='darkorchid',label='Domain adapted')
        if i == len(inputdomain_list)-1:
            ax0[i,0].legend(framealpha=0.5,fontsize=6.25)
            ax0[i,0].set_xlabel('Percent Data Observed',fontsize=label_fontsize)
            ax0[i,0].set_ylabel('Normalized\nSubspace Distance $d/n$',fontsize=label_fontsize)
            ax0[i,1].legend(framealpha=0.5,fontsize=6.25)
            ax0[i,1].set_xlabel('Percent Data Observed',fontsize=label_fontsize)
            ax0[i,1].set_ylabel('RMSE',fontsize=label_fontsize)
            ax0[i,2].set_xlabel('Percent Data Observed',fontsize=label_fontsize)
            ax0[i,2].set_ylabel(r'R$^2$',fontsize=label_fontsize)
            ax0[i,2].legend(framealpha=0.5,fontsize=6.25)


    fig0.savefig('experimental_plots/online_learning/fixed_p_decreasing_e_shifts_online_metrics.pdf', format='pdf')