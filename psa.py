import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import make_interp_spline
import scipy.linalg as linalg
from scipy.linalg import orth

def interpolate_inputs(X,Z,ts,tt,interptype='time'):
    visualization = False

    if interptype == 'time':
        """ Common Time Interpolation Scheme """
        big_time = np.unique(np.concatenate((ts,tt),axis=0))
        bs = make_interp_spline(ts,X)
        bt = make_interp_spline(tt,Z)
        X_interp = bs(big_time)
        Z_interp = bt(big_time)

        if visualization == True:
            """ Visualizing interpolation """
            plt.figure()
            plt.scatter(big_time,X_interp[:,0],s=2,label='$X_s$')
            plt.scatter(big_time,Z_interp[:,0],s=2,label='$X_t$')
            plt.legend()
            plt.show()

    elif interptype == 'removal':
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
            if visualization == True:
                """ Visualizing interpolation """
                plt.figure()
                plt.scatter(ts[idx],X[idx,0],s=1)
                plt.scatter(ts[idx_zeros],X[idx_zeros,0],s=2)
                plt.figure()
                plt.scatter(ts_interp, X_interp[:,0],s=2)
                plt.scatter(tt_interp, Z_interp[:,0],s=2)
                plt.scatter(ts[indices_remove],X[indices_remove,0],s=2,marker='x',color='red')
                plt.show()
        else:
            # Remove zeros from Z
            idx = np.arange(Z[:,0].shape[0])
            idx_zeros = idx[Z[:,0] <= 0]
            indices_remove = np.linspace(idx_zeros[0],idx_zeros[-1],numpoints_remove,dtype=np.int16)
            X_interp = X
            Z_interp = np.delete(Z,indices_remove,axis=0)
            ts_interp = ts
            tt_interp = np.delete(tt,indices_remove,axis=0)
            if visualization == True:
                """ Visualizing interpolation """
                plt.figure()
                plt.scatter(tt[idx_zeros],Z[idx_zeros,0],s=2)
                plt.scatter(tt[idx],Z[idx,0],s=1)
                plt.figure()
                plt.scatter(ts_interp, X_interp[:,0],s=2)
                plt.scatter(tt_interp, Z_interp[:,0],s=2)
                plt.scatter(tt[indices_remove],Z[indices_remove,0],s=2,marker='x',color='red')
                plt.show()

    else:
        """ Common Time Interpolation Scheme """
        big_time = np.unique(np.concatenate((ts,tt),axis=0))
        bs = make_interp_spline(ts,X)
        bt = make_interp_spline(tt,Z)
        X_interp = bs(big_time)
        Z_interp = bt(big_time)
        
        if visualization == True:
            """ Visualizing interpolation """
            plt.figure()
            plt.scatter(big_time,X_interp[:,0],s=2,label='$X_s$')
            plt.scatter(big_time,Z_interp[:,0],s=2,label='$X_t$')
            plt.legend()
            plt.show()
    return X_interp, Z_interp

def find_trajectory_matrix(time_series,window_length):
    """
    Transform 1D input time series into a trajectory matrix
    """
    L = window_length # The window length.
    N = len(time_series)
    K = N - L + 1 # The number of columns in the trajectory matrix.
    X = np.column_stack([time_series[i:i+L] for i in range(0,K)])
    return X

def H_to_TS(X_i):
    """
    Reconstructs input time series from trajectory matrix
    """
    # Reverse the column ordering of X_i
    X_rev = X_i[::-1]
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])

def plot_manifolds(Xa,Za,Hx_proj,Hz_proj):
    Xa = Xa.cpu().detach().numpy()
    Za = Za.cpu().detach().numpy()
    Hx_proj = Hx_proj.cpu().detach().numpy()
    Hz_proj = Hz_proj.cpu().detach().numpy()

    scolors = Hx_proj[2,:] #np.linspace(0,Hx_proj.shape[1],num=Hx_proj.shape[1])
    tcolors = Hz_proj[2,:] #np.linspace(0,Hz_proj.shape[1],num=Hz_proj.shape[1])
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1,projection='3d')
    ax.scatter(Hx_proj[0,:],Hx_proj[1,:],Hx_proj[2,:],s=8,marker='.',c=scolors,cmap='viridis',label='$H_{X,\mathrm{proj}}$',rasterized=True,depthshade=False)
    ax.scatter(Hz_proj[0,:],Hz_proj[1,:],Hz_proj[2,:],s=8,marker='.',c=tcolors,cmap='plasma',label='$H_{Z,\mathrm{proj}}$',rasterized=True,depthshade=False)
    ax.legend(framealpha=0.5)

    scolors = Xa[:,2] #np.linspace(0,Xa.shape[0],num=Xa.shape[0])
    tcolors = Za[:,2] #np.linspace(0,Za.shape[0],num=Za.shape[0])
    ax = fig.add_subplot(1,2,2,projection='3d')
    ax.scatter(Xa[:,0],Xa[:,1],Xa[:,2],s=8,marker='.',c=scolors,cmap='viridis',label='$X_a$',rasterized=True,depthshade=False)
    ax.scatter(Za[:,0],Za[:,1],Za[:,2],s=8,marker='.',c=tcolors,cmap='plasma',label='$Z_a$',rasterized=True,depthshade=False)
    ax.legend(framealpha=0.5)

    plt.show()

def batch_procrustes_subspace_adaptation(X, Z, Ys, Yt, ts, tt, window_length, k=5, rotation=True, scaling=True):
    """
    Procrustes subspace adaptation with full domain seen. Operational subspace found by PCA (SVD of the input matrix Z)
    """
    # Set training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    L = window_length

    # Mean center and scale data to unit variance
    scaler = StandardScaler(with_std=True)
    X = scaler.fit_transform(X)
    Z = scaler.fit_transform(Z)

    # Interpolate inputs
    X_interp, Z_interp = interpolate_inputs(X,Z,ts,tt)

    # Inputs are in the column space of X
    for i in range(0,np.size(X,1)):
        Hx_Li = torch.tensor(find_trajectory_matrix(X_interp[:,i],L),dtype=torch.float32).to(device)
        Hz_Li = torch.tensor(find_trajectory_matrix(Z_interp[:,i],L),dtype=torch.float32).to(device)
        Hxi = torch.tensor(find_trajectory_matrix(X[:,i],L),dtype=torch.float32).to(device)
        Hzi = torch.tensor(find_trajectory_matrix(Z[:,i],L),dtype=torch.float32).to(device)

        # Stack Hankel matrices. Two sets of stacks are generated; one for linked data and one for the full datasets.
        # Linked data is used to find rotation matrix.
        if i==0:
            Hx_L = Hx_Li
            Hz_L = Hz_Li
            Hx = Hxi
            Hz = Hzi
        else:
            Hx_L = torch.cat((Hx_L,Hx_Li),dim=0)
            Hz_L = torch.cat((Hz_L,Hz_Li),dim=0)
            Hx = torch.cat((Hx,Hxi),dim=0)
            Hz = torch.cat((Hz,Hzi),dim=0)

    Us,_,_ = torch.linalg.svd(Hx_L)
    Ut,_,_ = torch.linalg.svd(Hz_L)
    Hx_sub = Us[:,0:k]
    Hz_sub = Ut[:,0:k]

    Hx_proj = Hx_sub.T @ Hx_L
    Hz_proj = Hz_sub.T @ Hz_L

    U,S,V = torch.linalg.svd(Hx_proj @ Hz_proj.T)
    Q = V.T @ U.T
    s = torch.trace(torch.diag(S))/torch.trace(Hx_proj @ Hx_proj.T)

    if rotation and scaling:
        Xa = s * Q @ (Hx_sub.T @ Hx)
    elif not rotation and scaling:
        Xa = s * (Hx_sub.T @ Hx)
    elif rotation and not scaling:
        Xa = Q @ (Hx_sub.T @ Hx)
    elif not rotation and not scaling:
        Xa = Hx_sub.T @ Hx
    else:
        Xa = s * Q @ (Hx_sub.T @ Hx)

    Za = Hz_sub.T @ Hz
    Xa = Xa.T
    Za = Za.T
    # Hankelise outputs
    Ys_H = find_trajectory_matrix(Ys,L)
    Yt_H = find_trajectory_matrix(Yt,L)

    return Xa.cpu().detach().numpy(), Za.cpu().detach().numpy(), Ys_H.T, Yt_H.T

def ojas(Y, Uhat, U, eta=0.5):
    """
    Oja's algorithm for subspace tracking
    Y is the streamed data, U is the true subspace, and Uhat is the streaming estimate
    """
    n_samples, n_features = Y.shape
    
    errors = []
    for i in range(n_samples):
        y = Y[i, :].reshape(-1, 1)  # Column vector
        
        # Oja's update
        # Uhat += eta * (y @ (y.T @ Uhat) - (Uhat @ (Uhat.T @ y)) @ (y.T @ Uhat))
        Uhat += eta * y @ (Uhat.T @ y).T
        
        # Re-orthogonalize U
        Uhat = orth(Uhat)
        
        # Compute error (subspace distance). Normalize error wrt to first sample error.
        # error = np.linalg.norm(np.eye(k) - U.T @ U, 'fro')
        # errors.append(error)

        error = linalg.norm(U - Uhat @ (Uhat.T @ U),ord='fro')**2
        errors.append(error)

    return Uhat, np.array(errors)

def streaming_procrustes_subspace_adaptation(X, Z, Ys, Yt, ts, tt, window_length, k=5, interptype='time', rotation=True, scaling=True, manifold_visual=False):
    """
    Procrustes subspace adaptation with streaming domain. Operational subspace is found by Oja's algorithm using partially streamed Z
    """
    # Set training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    L = window_length

    # Interpolate inputs
    X_interp, Z_interp = interpolate_inputs(X,Z,ts,tt,interptype=interptype)

    # Mean center and scale data to unit variance
    scaler = StandardScaler(with_std=True)
    X = scaler.fit_transform(X)
    Z = scaler.fit_transform(Z)
    X_interp = scaler.fit_transform(X_interp)
    Z_interp = scaler.fit_transform(Z_interp)

    # Inputs are in the column space of X
    for i in range(0,np.size(X,1)):
        Hx_Li = torch.tensor(find_trajectory_matrix(X_interp[:,i],L),dtype=torch.float32).to(device)
        Hz_Li = torch.tensor(find_trajectory_matrix(Z_interp[:,i],L),dtype=torch.float32).to(device)
        Hxi = torch.tensor(find_trajectory_matrix(X[:,i],L),dtype=torch.float32).to(device)
        Hzi = torch.tensor(find_trajectory_matrix(Z[:,i],L),dtype=torch.float32).to(device)

        # Stack Hankel matrices. Two sets of stacks are generated; one for linked data and one for the full datasets.
        # Linked data is used to find rotation matrix.
        if i==0:
            Hx_L = Hx_Li
            Hz_L = Hz_Li
            Hx = Hxi
            Hz = Hzi
        else:
            Hx_L = torch.cat((Hx_L,Hx_Li),dim=0)
            Hz_L = torch.cat((Hz_L,Hz_Li),dim=0)
            Hx = torch.cat((Hx,Hxi),dim=0)
            Hz = torch.cat((Hz,Hzi),dim=0)

    Us,_,_ = torch.linalg.svd(Hx)
    Ut,_,_ = torch.linalg.svd(Hz_L)
    Hx_sub = Us[:,0:k]
    
    # Target domain subspace streaming
    Uhat = linalg.orth(np.random.randn(Hx_sub.shape[0],Hx_sub.shape[1]))
    Utrue = Ut[:,0:k]
    Hz_sub,_ = ojas(Hz_L.T.cpu().detach().numpy(), Uhat,Utrue.cpu().detach().numpy(), eta=0.001)
    Hz_sub = torch.tensor(Hz_sub,dtype=torch.float32).to(device)

    Hx_proj = Hx_sub.T @ Hx_L
    Hz_proj = Hz_sub.T @ Hz_L

    U,S,V = torch.linalg.svd(Hx_proj @ Hz_proj.T)
    Q = V.T @ U.T
    s = torch.trace(torch.diag(S))/torch.trace(Hx_proj @ Hx_proj.T)

    if rotation and scaling:
        Xa = s * Q @ (Hx_sub.T @ Hx)
    elif not rotation and scaling:
        Xa = s * (Hx_sub.T @ Hx)
    elif rotation and not scaling:
        Xa = Q @ (Hx_sub.T @ Hx)
    elif not rotation and not scaling:
        Xa = Hx_sub.T @ Hx
    else:
        Xa = s * Q @ (Hx_sub.T @ Hx)

    Za = Hz_sub.T @ Hz
    Xa = Xa.T
    Za = Za.T
    # Hankelise outputs
    Ys_H = find_trajectory_matrix(Ys,L)
    Yt_H = find_trajectory_matrix(Yt,L)

    # Plot manifolds
    if manifold_visual == True:
        plot_manifolds(Xa,Za,Hx_proj,Hz_proj) # Interpolated data
        plot_manifolds(Xa,Za,Hx_sub.T@Hx,Hz_sub.T@Hz) # Non-interpolated data

    # Compute Grassmannian distances between embeddings
    print(f'Grassmannian distance between unaligned projections: {linalg.norm(Hx_proj - Hz_proj)}')
    print(f'Grassmannian distance between aligned projections: {linalg.norm((s*Q @ (Hx_sub.T @ Hx_L)) - (Hz_sub.T @ Hz_L))}') # Take distance between interpolated projections

    return Xa.cpu().detach().numpy(), Za.cpu().detach().numpy(), Ys_H.T, Yt_H.T