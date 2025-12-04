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

if __name__ == '__main__':
    # rp is different between missions
    # mission1_data_directory = r'data\10orbit_ra=12000_rp=100.0\Results_ctrl=0_ra=12000_rp=100.0_hl=0.150_90.0deg.csv'
    mission1_data_directory = r'data\4orbit_ra=12000_rp=100.0\Results_ctrl=0_ra=12000_rp=100.0_hl=0.150_90.0deg.csv'
    
    # mission2_data_directory = r'data\4orbit_ra=12000_rp=99.0\Results_ctrl=0_ra=12000_rp=99.0_hl=0.150_90.0deg.csv'
    # mission2_data_directory = r'data\4orbit_ra=12000_rp=97.0\Results_ctrl=0_ra=12000_rp=97.0_hl=0.150_90.0deg.csv'
    # mission2_data_directory = r'data\4orbit_ra=12000_rp=96.0\Results_ctrl=0_ra=12000_rp=96.0_hl=0.150_90.0deg.csv'
    mission2_data_directory = r'data\4orbit_ra=12000_rp=95.0\Results_ctrl=0_ra=12000_rp=95.0_hl=0.150_90.0deg.csv'
    # mission2_data_directory = r'data\4orbit_ra=12000_rp=94.0\Results_ctrl=0_ra=12000_rp=94.0_hl=0.150_90.0deg.csv'
    # mission2_data_directory = r'data\4orbit_ra=12000_rp=92.0\Results_ctrl=0_ra=12000_rp=92.0_hl=0.150_90.0deg.csv'
    # mission2_data_directory = r'data\4orbit_ra=12000_rp=90.0\Results_ctrl=0_ra=12000_rp=90.0_hl=0.150_90.0deg.csv'

    utils.plot_domain_visualization(mission1_data_directory,mission2_data_directory)

    # Convert to dataframes and remove duplicate time entries
    df1 = pd.read_csv(mission1_data_directory).drop_duplicates(subset=['time'], keep='first')
    df2 = pd.read_csv(mission2_data_directory).drop_duplicates(subset=['time'], keep='first')

    # Specify up to which data point is seen by the model
    percent_seen = 1
    index_seen = round(percent_seen*df1['time'].shape[0]) # df1['time'].shape[0]

    #=======================================================================================#
    # Mission 1 Data Preprocessing
    #=======================================================================================#
    t1 = np.array(df1['time'])
    rho1 = utils.add_percent_noise(np.array(df1['rho']).reshape(-1,1),0)
    T1 = utils.add_percent_noise(np.array(df1['T']).reshape(-1,1),0)
    S1 = utils.add_percent_noise(np.array(df1['S']).reshape(-1,1),0)
    X1 = np.concatenate((rho1,T1,S1),axis=1)
    Y1 = np.array(df1['heat_rate']).reshape(-1,1)

    # Split the data into seen/unseen
    t1_seen = t1[0:index_seen]
    t1_unseen = t1[index_seen+1:]
    X1_seen = X1[0:index_seen,:]
    X1_unseen = X1[index_seen+1:]
    Y1_seen = Y1[0:index_seen,:]
    Y1_unseen = Y1[index_seen+1:,:]

    #=======================================================================================#
    # Mission 2 Data Preprocessing
    #=======================================================================================#
    t2 = np.array(df2['time'])
    rho2 = utils.add_percent_noise(np.array(df2['rho']).reshape(-1,1),0)
    T2 = utils.add_percent_noise(np.array(df2['T']).reshape(-1,1),0)
    S2 = utils.add_percent_noise(np.array(df2['S']).reshape(-1,1),0)
    X2 = np.concatenate((rho2,T2,S2),axis=1)
    Y2 = np.array(df2['heat_rate']).reshape(-1,1)

    # Split the data into seen/unseen
    t2_seen = t2[0:index_seen]
    t2_unseen = t2[index_seen+1:]
    X2_seen = X2[0:index_seen,:]
    X2_unseen = X2[index_seen+1:,:]
    Y2_seen = Y2[0:index_seen,:]
    Y2_unseen = Y2[index_seen+1:,:]

    #=======================================================================================#
    # Subspace Alignment
    #=======================================================================================#
    Xsa, Xta, Ys, Yt = psa.procrustes_manifold_alignment(X1,X2,Y1,Y2,t1,t2,window_length=50,k=5)
    Xsa_stream, Xta_seen, Ys_stream, Yt_seen = psa.streaming_procrustes_manifold_alignment(X1,X2_seen,Y1,Y2_seen,t1,t2_seen,window_length=5,k=5) # Note that streaming performs better with smaller window size 

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

    X1_seen = scaler.fit_transform(X1_seen)
    X2_seen = scaler.fit_transform(X2_seen)
    X1_seen_torch = torch.tensor(X1_seen,dtype=torch.float32).to(device)
    X2_seen_torch = torch.tensor(X2_seen,dtype=torch.float32).to(device)
    Y1_seen_torch = torch.tensor(Y1_seen,dtype=torch.float32).to(device)
    Y2_seen_torch = torch.tensor(Y2_seen,dtype=torch.float32).to(device)

    Xsa_torch = torch.tensor(Xsa,dtype=torch.float32).to(device)
    Xta_torch = torch.tensor(Xta,dtype=torch.float32).to(device)
    Ys_torch = torch.tensor(Ys,dtype=torch.float32).to(device)
    Yt_torch = torch.tensor(Yt,dtype=torch.float32).to(device)

    Xsa_stream_torch = torch.tensor(Xsa_stream,dtype=torch.float32).to(device)
    Xta_seen_torch = torch.tensor(Xta_seen,dtype=torch.float32).to(device)
    Ys_stream_torch = torch.tensor(Ys_stream,dtype=torch.float32).to(device)
    Yt_seen_torch = torch.tensor(Yt_seen,dtype=torch.float32).to(device)

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
    model_da = NeuralNetwork(input_size=Xsa_torch.size(1), hidden_sizes=hidden_sizes, output_size=Ys_torch.size(1)).to(device)
    model_sda = NeuralNetwork(input_size=Xsa_stream_torch.size(1), hidden_sizes=hidden_sizes, output_size=Ys_stream_torch.size(1)).to(device)

    loss_og = train_model(model_og,X1_torch,Y1_torch,num_epochs,learning_rate)
    loss_da = train_model(model_da,Xsa_torch,Ys_torch,num_epochs,learning_rate)
    loss_sda = train_model(model_sda,Xsa_stream_torch,Ys_stream_torch,num_epochs,learning_rate,weight_decay=1e-3) # Note the change in L2 regularizer

    print(f"Original model Loss: {loss_og}")
    print(f"Batch adapted model loss: {loss_da}")
    print(f"Streaming domain adapted model loss: {loss_sda}")

    # Save model and settings
    save_settings = True
    if save_settings == True:
        # Save model
        torch.save(model_og.state_dict(),"models/baseline_model.pt")
        torch.save(model_da.state_dict(),"models/psa_model.pt")
        # Save input size, output size, neuron number, and layer size settings to txt
        with open("models/baseline_model_settings.txt", "w") as f:
            f.write(f"{X1_torch.size(1)}\n")
            f.write(f"{Y1_torch.size(1)}\n")
            f.write(f"{num_layers}\n")
            f.write(f"{num_neurons}\n")
            f.write(f"{hidden_sizes}")
        with open("models/psa_model_settings.txt", "w") as f:
            f.write(f"{Xsa_torch.size(1)}\n")
            f.write(f"{Ys_torch.size(1)}\n")
            f.write(f"{num_layers}\n")
            f.write(f"{num_neurons}\n")
            f.write(f"{hidden_sizes}")

    model_og.eval()
    model_da.eval()
    with torch.no_grad():
        Yspred_torch_og = model_og(X1_torch)
        Yspred_torch_da = model_da(Xsa_torch)
        Yspred_torch_sda = model_sda(Xsa_stream_torch)
        Ytpred_torch_og = model_og(X2_torch)
        Ytpred_torch_da = model_da(Xta_torch)
        Ytpred_torch_sda = model_sda(Xta_seen_torch)

    Yspred_og = Yspred_torch_og.cpu().detach().numpy() # Save to local memory (.cpu), convert to numpy array (.detach.numpy), and convert to scalar value (.item)
    Yspred_da = Yspred_torch_da.cpu().detach().numpy()
    Yspred_sda = Yspred_torch_sda.cpu().detach().numpy()
    Ytpred_og = Ytpred_torch_og.cpu().detach().numpy()
    Ytpred_da = Ytpred_torch_da.cpu().detach().numpy()
    Ytpred_sda = Ytpred_torch_sda.cpu().detach().numpy()

    Yspred_da = psa.H_to_TS(Yspred_da.T)
    Yspred_sda = psa.H_to_TS(Yspred_sda.T)
    Ytpred_da = psa.H_to_TS(Ytpred_da.T)
    Ytpred_sda = psa.H_to_TS(Ytpred_sda.T)
    Ys = psa.H_to_TS(Ys.T) # Ys == Y1 after de-Hankelising
    Ys_stream = psa.H_to_TS(Ys_stream.T)
    Yt = psa.H_to_TS(Yt.T) # Yt == Y2 after de-Hankelising
    Yt_seen = psa.H_to_TS(Yt_seen.T)

    rmse_og = root_mean_squared_error(Y2, Ytpred_og)
    rmse_da = root_mean_squared_error(Yt, Ytpred_da)
    rmse_sda = root_mean_squared_error(Yt_seen, Ytpred_sda)

    # Plotting
    fig1, ax1 = plt.subplots(2,2,width_ratios=[0.4,1])
    plt.subplots_adjust(wspace=0.25)
    # Source Domain
    ax1[0,0].scatter(Yspred_og,Ys,s=2,label=f"RMSE: {round(root_mean_squared_error(Ys, Yspred_og),4)}",color='red', rasterized=True)
    # ax1[0,0].scatter(Yspred_da,Ys,s=2,label=f"RMSE: {round(root_mean_squared_error(Ys, Yspred_da),4)}",color='purple',alpha=0.25)
    ax1[0,0].scatter(Yspred_sda,Ys_stream,s=2,label=f"RMSE: {round(root_mean_squared_error(Ys_stream, Yspred_sda),4)}",color='darkorchid', rasterized=True)
    # Plot line y=x, the ideal predicted vs. actual curve
    lims = [
        np.min([ax1[0,0].get_xlim(), ax1[0,0].get_ylim()]),  # min of both axes
        np.max([ax1[0,0].get_xlim(), ax1[0,0].get_ylim()]),  # max1 of both axes
    ]
    ax1[0,0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax1[0,0].set_aspect('equal')
    ax1[0,0].set_xlim(lims)
    ax1[0,0].set_ylim(lims)
    ax1[0,0].set_xlabel('Predicted Output')
    ax1[0,0].set_ylabel('Actual Output') 
    ax1[0,0].legend(fontsize=6.25,framealpha=0.5)

    ax1[0,1].plot(Ys,label="Validation actual",color='black')
    ax1[0,1].plot(Yspred_og,label="No adaptation",color='red')
    # ax1[0,1].plot(Yspred_da,label="Batch adapted",color='purple',alpha=0.25)
    ax1[0,1].plot(Yspred_sda,label="Streaming adapted",color='darkorchid')
    ax1[0,1].set_xlabel('Time Step')
    ax1[0,1].set_ylabel('Heat Rate (W/cm$^2$)')
    ax1[0,1].legend(fontsize=6.25,framealpha=0.5)
    
    # Target Domain
    ax1[1,0].scatter(Ytpred_og,Yt,s=2,label=f"RMSE: {round(root_mean_squared_error(Yt, Ytpred_og),4)}",color='red', rasterized=True)
    # ax1[1,0].scatter(Ytpred_da,Yt,s=2,label=f"RMSE: {round(root_mean_squared_error(Yt, Ytpred_da),4)}",color='purple',alpha=0.25, rasterized=True)
    ax1[1,0].scatter(Ytpred_sda,Yt_seen,s=2,label=f"RMSE: {round(root_mean_squared_error(Yt_seen, Ytpred_sda),4)}",color='darkorchid', rasterized=True)
    # Plot line y=x, the ideal predicted vs. actual curve
    lims = [
        np.min([ax1[1,0].get_xlim(), ax1[1,0].get_ylim()]),  # min of both axes
        np.max([ax1[1,0].get_xlim(), ax1[1,0].get_ylim()]),  # max1 of both axes
    ]
    ax1[1,0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax1[1,0].set_aspect('equal')
    ax1[1,0].set_xlim(lims)
    ax1[1,0].set_ylim(lims)
    ax1[1,0].set_xlabel('Predicted Output')
    ax1[1,0].set_ylabel('Actual Output') 
    ax1[1,0].legend(fontsize=6.25,framealpha=0.5)

    ax1[1,1].plot(Yt,label="Operational actual",color='gray')
    ax1[1,1].plot(Ytpred_og,label="No adaptation",color='red')
    # ax1[1,1].plot(Ytpred_da,label="Batch adapted",color='purple',alpha=0.25)
    ax1[1,1].plot(Ytpred_sda,label="Streaming adapted",color='darkorchid')
    ax1[1,1].set_xlabel('Time Step')
    ax1[1,1].set_ylabel('Heat Rate (W/cm$^2$)')
    ax1[1,1].legend(fontsize=6.25,framealpha=0.5)
    fig1.savefig('plots/heatrate_vs_time.pdf', format='pdf')

    # Plotting state space in source domain
    fig2, ax2 = plt.subplots(2,3,width_ratios=[1,1,1]) # number of width ratios must match the number of columns of the grid
    ax2[0,0].scatter(df1['rho'], df1['heat_rate'], s=2, label="Validation actual", color='black', rasterized=True)
    ax2[0,0].scatter(df1['rho'], Yspred_og, s=2, label="No adaptation", color='red', rasterized=True)
    # ax2[0,0].scatter(df1['rho'], Yspred_da, s=2, label="Batch adapted", color='purple',alpha=0.25, rasterized=True)
    ax2[0,0].scatter(df1['rho'], Yspred_sda, s=2, label="Streaming adapted", color='darkorchid', rasterized=True)
    ax2[0,0].set_ylabel('Heat Rate (W/cm$^2$)') 
    ax2[0,0].legend(fontsize=6.25,framealpha=0.5)

    ax2[0,1].scatter(df1['T'], df1['heat_rate'], s=2, label="Validation actual", color='black', rasterized=True)
    ax2[0,1].scatter(df1['T'], Yspred_og, s=2, label="No adaptation", color='red', rasterized=True)
    # ax2[0,1].scatter(df1['T'], Yspred_da, s=2, label="Batch adapted", color='purple',alpha=0.25, rasterized=True)
    ax2[0,1].scatter(df1['T'], Yspred_sda, s=2, label="Streaming adapted", color='darkorchid', rasterized=True)

    ax2[0,2].scatter(df1['S'], df1['heat_rate'], s=2, label="Validation actual", color='black', rasterized=True)
    ax2[0,2].scatter(df1['S'], Yspred_og, s=2, label="No adaptation", color='red', rasterized=True)
    # ax2[0,2].scatter(df1['S'], Yspred_da, s=2, label="Batch adapted", color='purple',alpha=0.25, rasterized=True)
    ax2[0,2].scatter(df1['S'], Yspred_da, s=2, label="Streaming adapted", color='darkorchid', rasterized=True)

    # Plotting state space in target domain
    ax2[1,0].scatter(df2['rho'], df2['heat_rate'], s=2, label="Operational actual", color='gray', rasterized=True)
    ax2[1,0].scatter(df2['rho'], Ytpred_og, s=2, label="No adaptation", color='red', rasterized=True)
    # ax2[1,0].scatter(df2['rho'], Ytpred_da, s=2, label="Batch adapted", color='purple',alpha=0.25, rasterized=True)
    ax2[1,0].scatter(df2['rho'][0:index_seen], Ytpred_sda, s=2, label="Streaming adapted", color='darkorchid', rasterized=True)
    ax2[1,0].set_xlabel('Atmospheric Density (kg/m$^3$)',labelpad=15)
    ax2[1,0].set_ylabel('Heat Rate (W/cm$^2$)') 
    ax2[1,0].legend(fontsize=6.25,framealpha=0.5)

    ax2[1,1].scatter(df2['T'], df2['heat_rate'], s=2, label="Operational actual", color='gray', rasterized=True)
    ax2[1,1].scatter(df2['T'], Ytpred_og, s=2, label="No adaptation", color='red', rasterized=True)
    # ax2[1,1].scatter(df2['T'], Ytpred_da, s=2, label="Batch adapted", color='purple',alpha=0.25, rasterized=True)
    ax2[1,1].scatter(df2['T'][0:index_seen], Ytpred_sda, s=2, label="Streaming adapted", color='darkorchid', rasterized=True)
    ax2[1,1].set_xlabel('Freestream Temperature (K)',labelpad=15)

    ax2[1,2].scatter(df2['S'], df2['heat_rate'], s=2, label="Operational actual", color='gray', rasterized=True)
    ax2[1,2].scatter(df2['S'], Ytpred_og, s=2, label="No adaptation", color='red', rasterized=True)
    # ax2[1,2].scatter(df2['S'], Ytpred_da, s=2, label="Batch adapted", color='purple',alpha=0.25)
    ax2[1,2].scatter(df2['S'][0:index_seen], Ytpred_sda, s=2, label="Streaming adapted", color='darkorchid', rasterized=True)
    ax2[1,2].set_xlabel('Molecular Speed Ratio',labelpad=15)
    fig2.savefig('plots/state_space.pdf', format='pdf')

    # Plotting state space in target domain
    fig3, ax3 = plt.subplots(1,4,width_ratios=[1,1,1,1],figsize=(7,1.5)) # number of width ratios must match the number of columns of the grid
    fig3.tight_layout()
    plt.subplots_adjust(wspace=0.5)

    # Actual vs. predicted
    ax3[0].scatter(Ytpred_og,Yt,s=2,label=f"RMSE: {round(root_mean_squared_error(Yt, Ytpred_og),4)}",color='red', rasterized=True)
    # ax3[1,0].scatter(Ytpred_da,Yt,s=2,label=f"RMSE: {round(root_mean_squared_error(Yt, Ytpred_da),4)}",color='purple',alpha=0.25, rasterized=True)
    ax3[0].scatter(Ytpred_sda,Yt_seen,s=2,label=f"RMSE: {round(root_mean_squared_error(Yt_seen, Ytpred_sda),4)}",color='darkorchid', rasterized=True)
    # Plot line y=x, the ideal predicted vs. actual curve
    lims = [
        np.min([ax3[0].get_xlim(), ax3[0].get_ylim()]),  # min of both axes
        np.max([ax3[0].get_xlim(), ax3[0].get_ylim()]),  # max1 of both axes
    ]
    ax3[0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax3[0].set_aspect('equal')
    ax3[0].set_xlim(lims)
    ax3[0].set_ylim(lims)
    ax3[0].set_xlabel('Predicted Output')
    ax3[0].set_ylabel('Actual Output') 
    ax3[0].legend(fontsize=6.25,framealpha=0.5)

    # Plotting state space in target domain
    ax3[1].scatter(df2['rho'], df2['heat_rate'], s=2, label="Operational actual", color='gray', rasterized=True)
    ax3[1].scatter(df2['rho'], Ytpred_og, s=2, label="No adaptation", color='red', rasterized=True)
    # ax3[1].scatter(df2['rho'], Ytpred_da, s=2, label="Batch adapted", color='purple',alpha=0.25)
    ax3[1].scatter(df2['rho'][0:index_seen], Ytpred_sda, s=2, label="Streaming adapted", color='darkorchid', rasterized=True)
    ax3[1].set_xlabel('Atmospheric\n Density (kg/m$^3$)',labelpad=15)
    ax3[1].set_ylabel('Heat Rate (W/cm$^2$)') 
    ax3[1].legend(fontsize=6.25,framealpha=0.5)

    ax3[2].scatter(df2['T'], df2['heat_rate'], s=2, label="Operational actual", color='gray', rasterized=True)
    ax3[2].scatter(df2['T'], Ytpred_og, s=2, label="No adaptation", color='red', rasterized=True)
    # ax3[2].scatter(df2['T'], Ytpred_da, s=2, label="Batch adapted", color='purple',alpha=0.25)
    ax3[2].scatter(df2['T'][0:index_seen], Ytpred_sda, s=2, label="Streaming adapted", color='darkorchid', rasterized=True)
    ax3[2].set_xlabel('Freestream\n Temperature (K)',labelpad=15)

    ax3[3].scatter(df2['S'], df2['heat_rate'], s=2, label="Operational actual", color='gray', rasterized=True)
    ax3[3].scatter(df2['S'], Ytpred_og, s=2, label="No adaptation", color='red', rasterized=True)
    # ax3[1,2].scatter(df2['S'], Ytpred_da, s=2, label="Batch adapted", color='purple',alpha=0.25)
    ax3[3].scatter(df2['S'][0:index_seen], Ytpred_sda, s=2, label="Streaming adapted", color='darkorchid', rasterized=True)
    ax3[3].set_xlabel('Molecular\n Speed Ratio',labelpad=15)
    fig3.savefig('plots/actualpred_state_space.pdf', format='pdf', bbox_inches='tight')