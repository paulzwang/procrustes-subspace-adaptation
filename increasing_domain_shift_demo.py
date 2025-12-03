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
    mission1_data_directory = r'data\4orbit_ra=12000_rp=100.0\Results_ctrl=0_ra=12000_rp=100.0_hl=0.150_90.0deg.csv'
    mission2_data_directory = r'data\4orbit_ra=12000_rp=97.0\Results_ctrl=0_ra=12000_rp=97.0_hl=0.150_90.0deg.csv'
    mission3_data_directory = r'data\4orbit_ra=12000_rp=96.0\Results_ctrl=0_ra=12000_rp=96.0_hl=0.150_90.0deg.csv'
    mission4_data_directory = r'data\4orbit_ra=12000_rp=95.0\Results_ctrl=0_ra=12000_rp=95.0_hl=0.150_90.0deg.csv'
    mission5_data_directory = r'data\4orbit_ra=12000_rp=94.0\Results_ctrl=0_ra=12000_rp=94.0_hl=0.150_90.0deg.csv'
    shift_list = ["6 km shift in periapsis", "5 km shift in periapsis", "4 km shift in periapsis", "3 km shift in periapsis"] # Model training performed on descending shifts

    # Convert to dataframes and remove duplicate time entries
    df1 = pd.read_csv(mission1_data_directory).drop_duplicates(subset=['time'], keep='first')
    df2 = pd.read_csv(mission2_data_directory).drop_duplicates(subset=['time'], keep='first')
    df3 = pd.read_csv(mission3_data_directory).drop_duplicates(subset=['time'], keep='first')
    df4 = pd.read_csv(mission4_data_directory).drop_duplicates(subset=['time'], keep='first')
    df5 = pd.read_csv(mission5_data_directory).drop_duplicates(subset=['time'], keep='first')

    #=======================================================================================#
    # Mission 1 Data Preprocessing
    #=======================================================================================#
    t1 = np.array(df1['time'])
    X1 = np.concatenate((np.array(df1['rho']).reshape(-1,1),np.array(df1['T']).reshape(-1,1),np.array(df1['S']).reshape(-1,1)),axis=1)
    Y1 = np.array(df1['heat_rate']).reshape(-1,1)
    #=======================================================================================#
    # Mission 2 Data Preprocessing
    #=======================================================================================#
    t2 = np.array(df2['time'])
    X2 = np.concatenate((np.array(df2['rho']).reshape(-1,1),np.array(df2['T']).reshape(-1,1),np.array(df2['S']).reshape(-1,1)),axis=1)
    Y2 = np.array(df2['heat_rate']).reshape(-1,1)
    #=======================================================================================#
    # Mission 3 Data Preprocessing
    #=======================================================================================#
    t3 = np.array(df3['time'])
    X3 = np.concatenate((np.array(df3['rho']).reshape(-1,1),np.array(df3['T']).reshape(-1,1),np.array(df3['S']).reshape(-1,1)),axis=1)
    Y3 = np.array(df3['heat_rate']).reshape(-1,1)
    #=======================================================================================#
    # Mission 4 Data Preprocessing
    #=======================================================================================#
    t4 = np.array(df4['time'])
    X4 = np.concatenate((np.array(df4['rho']).reshape(-1,1),np.array(df4['T']).reshape(-1,1),np.array(df4['S']).reshape(-1,1)),axis=1)
    Y4 = np.array(df4['heat_rate']).reshape(-1,1)
    #=======================================================================================#
    # Mission 5 Data Preprocessing
    #=======================================================================================#
    t5 = np.array(df5['time'])
    X5 = np.concatenate((np.array(df5['rho']).reshape(-1,1),np.array(df5['T']).reshape(-1,1),np.array(df5['S']).reshape(-1,1)),axis=1)
    Y5 = np.array(df5['heat_rate']).reshape(-1,1)

    inputdomain_list = [X5, X4, X3, X2]
    outputdomain_list = [Y5, Y4, Y3, Y2]
    t_list = [t5, t4, t3, t2]

    #=======================================================================================#
    # Instantiate Plotting
    #=======================================================================================#
    linecolor_list = np.linspace(0.8,0.1,len(inputdomain_list)) # Create an array of values to customize line color in RGB
    fig0, ax0 = plt.subplots(2,2,figsize=(7,3))
    plt.subplots_adjust(wspace=0.25)
    fig1, ax1 = plt.subplots(len(inputdomain_list),2,width_ratios=[0.4,1])
    plt.subplots_adjust(wspace=0.1)
    fig2, ax2 = plt.subplots(1,3,width_ratios=[1,1,1],figsize=(7,1.5)) # number of width ratios must match the number of columns of the grid
    fig3, ax3 = plt.subplots(len(inputdomain_list),3,width_ratios=[1,1,1])

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
        Xsa, Xta, Ys, Yt = psa.streaming_procrustes_manifold_alignment(X1,X_fromlist,Y1,Y_fromlist,t1,t_fromlist,window_length,k)

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

        Xsa_torch = torch.tensor(Xsa,dtype=torch.float32).to(device)
        Xta_torch = torch.tensor(Xta,dtype=torch.float32).to(device)
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
        model_da = NeuralNetwork(input_size=Xsa_torch.size(1), hidden_sizes=hidden_sizes, output_size=Ys_torch.size(1)).to(device)

        loss_og = train_model(model_og,X1_torch,Y1_torch,num_epochs,learning_rate)
        loss_da = train_model(model_da,Xsa_torch,Ys_torch,num_epochs,learning_rate,weight_decay=1e-3)

        print(f"Original model Loss: {loss_og}")
        print(f"Domain adapted model loss: {loss_da}")

        model_og.eval()
        model_da.eval()
        with torch.no_grad():
            Yspred_torch_og = model_og(X1_torch)
            Yspred_torch_da = model_da(Xsa_torch)
            Ytpred_torch_og = model_og(X_fromlist_torch)
            Ytpred_torch_da = model_da(Xta_torch)

        Yspred_og = Yspred_torch_og.cpu().detach().numpy() # Save to local memory (.cpu), convert to numpy array (.detach.numpy), and convert to scalar value (.item)
        Yspred_da = Yspred_torch_da.cpu().detach().numpy()
        Ytpred_og = Ytpred_torch_og.cpu().detach().numpy()
        Ytpred_da = Ytpred_torch_da.cpu().detach().numpy()

        Yspred_da = psa.H_to_TS(Yspred_da.T)
        Ytpred_da = psa.H_to_TS(Ytpred_da.T)
        Ys = psa.H_to_TS(Ys.T)
        Yt = psa.H_to_TS(Yt.T)

        rmse_og = root_mean_squared_error(Y_fromlist, Ytpred_og)
        rmse_da = root_mean_squared_error(Yt, Ytpred_da)

        #=======================================================================================#
        # Plotting
        #=======================================================================================#
        # Getting inputs from list
        rhot = inputdomain_list[i][:,0] # Cannot use X_fromlist bc it is now mean-centered and scaled
        Tt = inputdomain_list[i][:,1]
        St = inputdomain_list[i][:,2]
        heatt = outputdomain_list[i]

        """ Input Domains and Shift Visualization """
        ax0[0,0].plot(t_list[i], rhot, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i]))) # Line color is in RGB, iterations increase green and decrease blue
        ax0[0,1].plot(t_list[i], Tt, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])))
        ax0[1,0].plot(t_list[i], St, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])))
        ax0[1,1].plot(t_list[i], heatt, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])))
        if i == len(inputdomain_list)-1:
            ax0[0,0].plot(df1['time'], df1['rho'], label="Validation domain", color='0')
            ax0[0,1].plot(df1['time'], df1['T'], label="Validation domain",  color='0')
            ax0[1,0].plot(df1['time'], df1['S'], label="Validation domain", color='0')
            ax0[1,1].plot(df1['time'], df1['heat_rate'], label="Validation domain", color='0')
            ax0[0,0].set_ylabel('Atmospheric\nDensity (kg/m$^3$)',fontsize=9) 
            ax0[0,1].set_ylabel('Freestream\nTemperature (K)',fontsize=9) 
            ax0[1,0].set_xlabel('Time (s)',fontsize=9)
            ax0[1,0].set_ylabel('Molecular\nSpeed Ratio',fontsize=9) 
            ax0[1,1].set_xlabel('Time (s)',fontsize=9)
            ax0[1,1].set_ylabel('Heat Rate (W/cm$^2$)',fontsize=9) 
            ax0[0,0].legend(fontsize=4,loc='upper left',framealpha=0.5)

        """ Actual vs. Predicted and Heat Rate vs. Time """
        # Plotting in target domain
        ax1[i,0].scatter(Ytpred_og,Yt,s=1,label=f"RMSE: {round(root_mean_squared_error(Yt, Ytpred_og),4)}",color='red', rasterized=True)
        ax1[i,0].scatter(Ytpred_da,Yt,s=1,label=f"RMSE: {round(root_mean_squared_error(Yt, Ytpred_da),4)}",color='darkorchid', rasterized=True)
        # Plot line y=x, the ideal predicted vs. actual curve
        lims = [
            np.min([ax1[i,0].get_xlim(), ax1[i,0].get_ylim()]),  # min of both axes
            np.max([ax1[i,0].get_xlim(), ax1[i,0].get_ylim()]),  # max1 of both axes
        ]
        ax1[i,0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax1[i,0].set_aspect('equal')
        ax1[i,0].set_xlim(lims)
        ax1[i,0].set_ylim(lims)
        if i == len(inputdomain_list)-1:
            ax1[i,0].set_xlabel('Predicted Output',fontsize=9)
            ax1[i,0].set_ylabel('Actual Output',fontsize=9) 
        ax1[i,0].legend(fontsize=4,framealpha=0.5)
        ax1[i,1].plot(Yt,label=shift_list[i],color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])))
        ax1[i,1].plot(Ytpred_og,label="No adaptation",color='red')
        ax1[i,1].plot(Ytpred_da,label="Domain adapted",color='darkorchid')
        if i == len(inputdomain_list)-1:
            ax1[i,1].set_xlabel('Time Step',fontsize=9)
            ax1[i,1].set_ylabel('Heat Rate (W/cm$^2$)',fontsize=9)
        ax1[i,1].legend(fontsize=4,loc='upper left',framealpha=0.5)

        
        """ State Space Comparison """
        # Plotting state space in target domain
        ax2[0].scatter(rhot, Yt, s=1, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
        ax2[1].scatter(Tt, Yt, s=1, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
        ax2[2].scatter(St, Yt, s=1, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
        if i == len(inputdomain_list)-1:
            ax2[0].set_xlabel('Atmospheric\n Density (kg/m$^3$)',labelpad=5,fontsize=9)
            ax2[0].set_ylabel('Heat Rate (W/cm$^2$)',fontsize=9) 
            ax2[0].scatter(df1['rho'], df1['heat_rate'], s=1, label="Validation domain", color='0', rasterized=True)
            ax2[1].scatter(df1['T'], df1['heat_rate'], s=1, label="Validation domain", color='0', rasterized=True)
            ax2[1].set_xlabel('Freestream\n Temperature (K)',labelpad=5,fontsize=9)
            ax2[2].scatter(df1['S'], df1['heat_rate'], s=1, label="Validation domain", color='0', rasterized=True)
            ax2[2].set_xlabel('Molecular\n Speed Ratio',labelpad=5,fontsize=9)
            ax2[0].legend(fontsize=4,loc='upper left',framealpha=0.5)

        """ State Space Predictions """
        # Plotting state space in target domain
        ax3[i,0].scatter(rhot, Yt, s=1, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
        ax3[i,0].scatter(rhot, Ytpred_og, s=1, label="No adaptation", color='red', rasterized=True)
        ax3[i,0].scatter(rhot, Ytpred_da, s=1, label="Domain adapted", color='darkorchid', rasterized=True)
        if i == len(inputdomain_list)-1:
            ax3[i,0].set_xlabel('Atmospheric\n Density (kg/m$^3$)',labelpad=5,fontsize=9)
            ax3[i,0].set_ylabel('Heat Rate (W/cm$^2$)',fontsize=9) 
        ax3[i,0].legend(fontsize=4,loc='upper left',framealpha=0.5)
        ax3[i,1].scatter(Tt, Yt, s=1, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
        ax3[i,1].scatter(Tt, Ytpred_og, s=1, label="No adaptation", color='red', rasterized=True)
        ax3[i,1].scatter(Tt, Ytpred_da, s=1, label="Domain adapted", color='darkorchid', rasterized=True)
        if i == len(inputdomain_list)-1:
            ax3[i,1].set_xlabel('Freestream\n Temperature (K)',labelpad=5,fontsize=9)
        ax3[i,2].scatter(St, Yt, s=1, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
        ax3[i,2].scatter(St, Ytpred_og, s=1, label="No adaptation", color='red', rasterized=True)
        ax3[i,2].scatter(St, Ytpred_da, s=1, label="Domain adapted", color='darkorchid', rasterized=True)
        if i == len(inputdomain_list)-1:
            ax3[i,2].set_xlabel('Molecular\n Speed Ratio',labelpad=5,fontsize=9)

    fig0.savefig('increasing_shifts_dataviz.pdf', format='pdf', bbox_inches='tight')
    fig1.savefig('increasing_shifts_heatratevstime.pdf', format='pdf')
    fig2.savefig('increasing_shifts_statespacecomparison.pdf', format='pdf', bbox_inches='tight')
    fig3.savefig('increasing_shifts_statespaceprediction.pdf', format='pdf')