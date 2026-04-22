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

num_replications = 100

def read_data(data_directory):
    df = pd.read_csv(data_directory).drop_duplicates(subset=['time'], keep='first')
    t = np.array(df['time'])
    X = np.concatenate((np.array(df['rho']).reshape(-1,1),np.array(df['T']).reshape(-1,1),np.array(df['S']).reshape(-1,1)),axis=1)
    Y = np.array(df['heat_rate']).reshape(-1,1)
    return df, t, X, Y

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
    outputdomain_list = [Y11, Y10, Y9, Y8, Y7, Y6, Y5, Y4, Y3, Y2]
    t_list = [t11, t10, t9, t8, t7, t6, t5, t4, t3, t2]
    
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

    list_numpoints= []

    #=======================================================================================#
    # Instantiate Plotting
    #=======================================================================================#
    fig6, ax6 = plt.subplots(1,3,figsize=(7,2),layout='constrained') # subspace distances vs mission number

    #=======================================================================================#
    # Training Loop
    #=======================================================================================#
    # Iterate through datasets
    for i in range(0,len(inputdomain_list)):
        X_fromlist = inputdomain_list[i]
        Y_fromlist = outputdomain_list[i]
        t_fromlist = t_list[i]

        #=======================================================================================#
        # Subspace Alignment
        #=======================================================================================#
        window_length = 5 # 50
        k = 5 # subspace rank
        if i >= 8:
            """ For r_p == 99 km and r_p == 98 km, use removal correpondence method """
            Xa, Za, Ys, Yt, Hx_proj, Hz_proj, Hx_proj_aligned, Hz_sub = psa.streaming_procrustes_subspace_adaptation(X1,X_fromlist,Y1,Y_fromlist,t1,t_fromlist,window_length,k,interptype='removal')
        else:
            """ Use time interpolation removal correpondence method """
            Xa, Za, Ys, Yt, Hx_proj, Hz_proj, Hx_proj_aligned, Hz_sub = psa.streaming_procrustes_subspace_adaptation(X1,X_fromlist,Y1,Y_fromlist,t1,t_fromlist,window_length,k,interptype='time')

        #=======================================================================================#
        # Data Preprocessing
        #=======================================================================================#
        # Set training device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        scaler = StandardScaler()
        X1 = scaler.fit_transform(X1)
        X_fromlist = scaler.fit_transform(X_fromlist)
        X1_torch = torch.tensor(X1,dtype=torch.float32).to(device)
        X_fromlist_torch = torch.tensor(X_fromlist,dtype=torch.float32).to(device)
        Y1_torch = torch.tensor(Y1,dtype=torch.float32).to(device)
        Y_fromlist_torch = torch.tensor(Y_fromlist,dtype=torch.float32).to(device)

        Xa_torch = torch.tensor(Xa,dtype=torch.float32).to(device)
        Za_torch = torch.tensor(Za,dtype=torch.float32).to(device)
        Ys_torch = torch.tensor(Ys,dtype=torch.float32).to(device)
        Yt_torch = torch.tensor(Yt,dtype=torch.float32).to(device)

        #=======================================================================================#
        # Model Training
        #=======================================================================================#
        # Define neural network hyperparameters
        num_layers = int(4)
        num_neurons = int(145)
        hidden_sizes = [num_neurons] * num_layers
        learning_rate = 0.006#0.00180
        num_epochs = 200 #200

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
                Ytpred_torch_og = model_og(X_fromlist_torch)
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

            rmse_og = root_mean_squared_error(Y_fromlist, Ytpred_og)
            rmse_da = root_mean_squared_error(Yt, Ytpred_da)
            r2_og = r2_score(Y_fromlist, Ytpred_og)
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


        #=======================================================================================#
        # Plotting
        #=======================================================================================#
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

        list_numpoints.append(X_fromlist.shape[0])


    """ Training Metrics """
    mission_numbers = list(np.linspace(len(inputdomain_list),1,len(inputdomain_list)))
    ax6[0].plot(mission_numbers,np.array(list_unaligned_d)/np.array(list_numpoints),marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='maroon',label=r'Pre-adaptation $d$')
    ax6[0].plot(mission_numbers,np.array(list_aligned_d)/np.array(list_numpoints),marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='navy',label=r'Post-adaptation $d$')
    ax6[0].set_xlabel('Mission Number')
    ax6[0].set_ylabel('Normalized\nSubspace Distance $d$')
    ax6[0].legend(framealpha=0.5,fontsize=6.25)
    ax6[1].errorbar(mission_numbers,list_mean_rmse_og,yerr=list_std_rmse_og,marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='red',label='No adaptation')
    ax6[1].errorbar(mission_numbers,list_mean_rmse_da,yerr=list_std_rmse_da,marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='darkorchid',label='Domain adapted')
    ax6[1].set_xlabel('Mission Number')
    ax6[1].set_ylabel('RMSE')
    ax6[1].legend(framealpha=0.5,fontsize=6.25)
    ax6[2].errorbar(mission_numbers,list_mean_r2_og,yerr=list_std_r2_og,marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='red',label='No adaptation')
    ax6[2].errorbar(mission_numbers,list_mean_r2_da,yerr=list_std_r2_da,marker='o',markerfacecolor='none',markersize=2,linestyle='-',color='darkorchid',label='Domain adapted')
    ax6[2].set_xlabel('Mission Number')
    ax6[2].set_ylabel(r'R$^2$')
    ax6[2].legend(framealpha=0.5,fontsize=6.25)

    fig6.savefig('journal_plots/100replications_rp_shift_metrics.pdf', format='pdf')