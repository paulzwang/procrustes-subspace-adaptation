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
    list_rmse_og = []
    list_rmse_da = []
    list_r2_og = []
    list_r2_da = []
    list_unaligned_d = []
    list_aligned_d = []



    #=======================================================================================#
    # Instantiate Plotting
    #=======================================================================================#
    linecolor_list = np.linspace(0.8,0.1,len(inputdomain_list)) # Create an array of values to customize line color in RGB
    fig0, ax0 = plt.subplots(4,1,layout='constrained') # time series visualization
    fig1, ax1 = plt.subplots(4,4,layout='constrained') # actual vs predicted and heat rate vs. time
    plt.subplots_adjust(wspace=0.1)
    fig2, ax2 = plt.subplots(1,3,figsize=(7,2),layout='constrained') # state space visualization
    fig5, ax5 = plt.subplots(4,2,subplot_kw=dict(projection='3d'),figsize=(3.25,5.5),layout='constrained') # manifolds part 2
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

        model_og = NeuralNetwork(input_size=X1_torch.size(1), hidden_sizes=hidden_sizes, output_size=Y1_torch.size(1)).to(device)
        model_da = NeuralNetwork(input_size=Xa_torch.size(1), hidden_sizes=hidden_sizes, output_size=Ys_torch.size(1)).to(device)

        loss_og = train_model(model_og,X1_torch,Y1_torch,num_epochs,learning_rate)
        loss_da = train_model(model_da,Xa_torch,Ys_torch,num_epochs,learning_rate,weight_decay=0)

        print(f"Original model Loss: {loss_og}")
        print(f"Domain adapted model loss: {loss_da}")

        model_og.eval()
        model_da.eval()
        with torch.no_grad():
            Yspred_torch_og = model_og(X1_torch)
            Yspred_torch_da = model_da(Xa_torch)
            Ytpred_torch_og = model_og(X_fromlist_torch)
            Ytpred_torch_da = model_da(Za_torch)

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
        r2_og = r2_score(Y_fromlist, Ytpred_og)
        r2_da = r2_score(Yt, Ytpred_da)

        #=======================================================================================#
        # Plotting
        #=======================================================================================#
        # Getting inputs from list
        rhot = inputdomain_list[i][:,0] # Cannot use X_fromlist bc it is now mean-centered and scaled
        Tt = inputdomain_list[i][:,1]
        St = inputdomain_list[i][:,2]
        heatt = outputdomain_list[i]

        """ Input Domains and Shift Visualization """
        ax0[0].plot(t_list[i], rhot, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i]))) # Line color is in RGB, iterations increase green and decrease blue
        ax0[1].plot(t_list[i], Tt, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])))
        ax0[2].plot(t_list[i], St, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])))
        ax0[3].plot(t_list[i], heatt, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])))
        if i == len(inputdomain_list)-1:
            ax0[0].plot(df1['time'], df1['rho'], label=r"DRM, $r_p=100$ km", color='0')
            ax0[1].plot(df1['time'], df1['T'], label=r"DRM, $r_p=100$ km",  color='0')
            ax0[2].plot(df1['time'], df1['S'], label=r"DRM, $r_p=100$ km", color='0')
            ax0[3].plot(df1['time'], df1['heat_rate'], label=r"DRM, $r_p=100$ km", color='0')
            ax0[0].set_ylabel('Atmospheric\nDensity (kg/m$^3$)') 
            ax0[1].set_ylabel('Freestream\nTemperature (K)') 
            ax0[2].set_ylabel('Molecular\nSpeed Ratio') 
            ax0[3].set_xlabel('Time (s)')
            ax0[3].set_ylabel('Heat Rate (W/cm$^2$)') 
            ax0[0].legend(framealpha=0.5,fontsize=4,ncol=3,loc='upper left')

        
        """ State Space Comparison """
        # Plotting state space in target domain
        ax2[0].scatter(rhot, Yt, s=0.25, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
        ax2[1].scatter(Tt, Yt, s=0.25, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
        ax2[2].scatter(St, Yt, s=0.25, label=shift_list[i], color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
        if i == len(inputdomain_list)-1:
            ax2[0].set_xlabel('Atmospheric\nDensity (kg/m$^3$)',labelpad=5)
            ax2[0].set_ylabel('Heat Rate (W/cm$^2$)') 
            ax2[0].scatter(df1['rho'], df1['heat_rate'], s=0.25, label=r"DRM, $r_p=100$ km", color='0', rasterized=True)
            ax2[1].scatter(df1['T'], df1['heat_rate'], s=0.25, label=r"DRM, $r_p=100$ km", color='0', rasterized=True)
            ax2[1].set_xlabel('Freestream\nTemperature (K)',labelpad=5)
            ax2[1].set_ylabel('Heat Rate (W/cm$^2$)') 
            ax2[2].scatter(df1['S'], df1['heat_rate'], s=0.25, label=r"DRM, $r_p=100$ km", color='0', rasterized=True)
            ax2[2].set_xlabel('Molecular\nSpeed Ratio',labelpad=5)
            ax2[2].set_ylabel('Heat Rate (W/cm$^2$)') 
            ax2[0].legend(fontsize=4,ncol=3,loc='upper left',framealpha=0.5)

        if i >= 6:
            """ Manifolds for Mission 1-4"""
            scolors = Hx_proj[2,:] #np.linspace(0,Hx_proj.shape[1],num=Hx_proj.shape[1])
            tcolors = Hz_proj[2,:] #np.linspace(0,Hz_proj.shape[1],num=Hz_proj.shape[1])

            ax5[i-6,0].scatter(Hx_proj[0,:],Hx_proj[1,:],Hx_proj[2,:],s=2,marker='.',c=scolors,cmap='viridis',label='$H_{X,\mathrm{proj}}$',rasterized=True,depthshade=False)
            ax5[i-6,0].scatter(Hz_proj[0,:],Hz_proj[1,:],Hz_proj[2,:],s=2,marker='.',c=tcolors,cmap='plasma',label='$H_{Z,\mathrm{proj}}$',rasterized=True,depthshade=False)
            ax5[i-6,0].legend(title=f'Mission {10-i},\nunaligned manifolds', title_fontsize=6, loc='upper left',framealpha=0.5,fontsize=4)
            ax5[i-6,0].tick_params(pad=-3)

            scolors = Xa[:,2] #np.linspace(0,Xa.shape[0],num=Xa.shape[0])
            tcolors = Za[:,2] #np.linspace(0,Za.shape[0],num=Za.shape[0])
            ax5[i-6,1].scatter(Xa[:,0],Xa[:,1],Xa[:,2],s=2,marker='.',c=scolors,cmap='viridis',label='$X_a$',rasterized=True,depthshade=False)
            ax5[i-6,1].scatter(Za[:,0],Za[:,1],Za[:,2],s=2,marker='.',c=tcolors,cmap='plasma',label='$Z_a$',rasterized=True,depthshade=False)
            ax5[i-6,1].legend(title=f'Mission {10-i},\naligned manifolds', title_fontsize=6, loc='upper left',framealpha=0.5,fontsize=4)
            ax5[i-6,1].tick_params(pad=-3)


            """ Actual vs. Predicted, State Space Prediction for Mission 1-4"""
            # Plotting in target domain
            ax1[i-6,0].scatter(Ytpred_og,Yt,s=0.25,label=f"RMSE: {round(rmse_og,4)}",color='red', rasterized=True)
            ax1[i-6,0].scatter(Ytpred_da,Yt,s=0.25,label=f"RMSE: {round(rmse_da,4)}",color='darkorchid', rasterized=True)
            # Plot line y=x, the ideal predicted vs. actual curve
            lims = [
                np.min([ax1[i-6,0].get_xlim(), ax1[i-6,0].get_ylim()]),  # min of both axes
                np.max([ax1[i-6,0].get_xlim(), ax1[i-6,0].get_ylim()]),  # max1 of both axes
            ]
            ax1[i-6,0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax1[i-6,0].set_aspect('equal')
            ax1[i-6,0].set_xlim(lims)
            ax1[i-6,0].set_ylim(lims)
            if i == len(inputdomain_list)-1:
                ax1[i-6,0].set_xlabel(r'Predicted $\dot Q$',fontsize=9)
                ax1[i-6,0].set_ylabel(r'Actual $\dot Q$',fontsize=9) 
            ax1[i-6,0].legend(fontsize=4,loc='lower right',framealpha=0.5)
            ax1[i-6,0].set_title(f'Mission {len(inputdomain_list) - i}: {shift_list[i]}',fontsize=9)
            
            ax1[i-6,1].scatter(rhot, Yt, s=0.25, label=f'Mission {10-i} ground truth', color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
            ax1[i-6,1].scatter(rhot, Ytpred_og, s=0.25, label="No adaptation", color='red', rasterized=True)
            ax1[i-6,1].scatter(rhot, Ytpred_da, s=0.25, label="Domain adapted", color='darkorchid', rasterized=True)
            if i == len(inputdomain_list)-1:
                ax1[i-6,1].set_xlabel('Atmospheric\n Density (kg/m$^3$)',labelpad=5,fontsize=9)
                ax1[i-6,1].set_ylabel('Heat Rate (W/cm$^2$)',fontsize=9) 
            ax1[i-6,1].legend(fontsize=4,loc='upper left',framealpha=0.5)
            ax1[i-6,2].scatter(Tt, Yt, s=0.25, label=f'Mission {10-i} ground truth', color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
            ax1[i-6,2].scatter(Tt, Ytpred_og, s=0.25, label="No adaptation", color='red', rasterized=True)
            ax1[i-6,2].scatter(Tt, Ytpred_da, s=0.25, label="Domain adapted", color='darkorchid', rasterized=True)
            if i == len(inputdomain_list)-1:
                ax1[i-6,2].set_xlabel('Freestream\n Temperature (K)',labelpad=5,fontsize=9)
            ax1[i-6,3].scatter(St, Yt, s=0.25, label=f'Mission {10-i} ground truth', color=(0, 0+float(linecolor_list[i]), 1-float(linecolor_list[i])), rasterized=True)
            ax1[i-6,3].scatter(St, Ytpred_og, s=0.25, label="No adaptation", color='red', rasterized=True)
            ax1[i-6,3].scatter(St, Ytpred_da, s=0.25, label="Domain adapted", color='darkorchid', rasterized=True)
            if i == len(inputdomain_list)-1:
                ax1[i-6,3].set_xlabel('Molecular\n Speed Ratio',labelpad=5,fontsize=9)
        
        """ Collect RMSE, R2, and subspace distances """
        list_rmse_og.append(rmse_og)
        list_rmse_da.append(rmse_da)
        list_r2_og.append(r2_og)
        list_r2_da.append(r2_da)
        list_unaligned_d.append(linalg.norm(Hx_proj - Hz_proj))
        list_aligned_d.append(linalg.norm(Hx_proj_aligned - Hz_proj))


    fig0.savefig('journal_plots/increasing_rp_shifts_dataviz.pdf', format='pdf')
    fig1.savefig('journal_plots/increasing_rp_shifts_prediction_results.pdf', format='pdf')
    fig2.savefig('journal_plots/increasing_rp_shifts_statespacecomparison.pdf', format='pdf')
    fig5.savefig('journal_plots/increasing_rp_shifts_manifolds.pdf', format='pdf')