import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import make_interp_spline
import scipy.linalg as linalg
from scipy.linalg import orth

def interpolate_inputs(X,Z,ts,tt):
    if len(ts) > len(tt):
        bs = make_interp_spline(ts,X)
        X_interp = bs(tt)
        Z_interp = Z
    else:
        bt = make_interp_spline(tt,Z) 
        X_interp = X
        Z_interp = bt(ts)
    return X_interp, Z_interp

def find_trajectory_matrix(time_series,window_length):
    """
    Function to construct a trajectory matrix with Hankel matrix structure.
    Time series must be a vector (1D array).
    """
    L = window_length # The window length.
    N = len(time_series)
    K = N - L + 1 # The number of columns in the trajectory matrix.
    # Create the trajectory matrix by pulling the relevant subseries of F, and stacking them as columns
    X = np.column_stack([time_series[i:i+L] for i in range(0,K)])
    # Note: the i+L above gives us up to i+L-1, as numpy array upper bounds are exclusive.
    return X

def H_to_TS(X_i):
    """
    "De-Hankelises" an input matrix, returning the original time series.
    Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series.
    """
    # Reverse the column ordering of X_i
    X_rev = X_i[::-1]
    return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0]+1, X_i.shape[1])])

def procrustes_manifold_alignment(X, Z, Ys, Yt, ts, tt, window_length, k=5, rotation=True, scaling=True):
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
    Hx_sub = Us[:,0:k]# Us[:,0:5]
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
    Oja's algorithm for subspace tracking.
    
    Parameters:
    Y : array-like, shape (n_samples, n_features)
        The input data matrix where each row is a data point.
    U : array-like, shape (n_features, k)
        The initial estimate of the subspace (orthonormal basis).
    eta : float
        The learning rate.
        
    Returns:
    errors : list
        List of subspace estimation errors at each step.
    times : list
        List of cumulative times at each step.
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

def streaming_procrustes_manifold_alignment(X, Z, Ys, Yt, ts, tt, window_length, k=5, rotation=True, scaling=True):
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

    Us,_,_ = torch.linalg.svd(Hx) # Change this line to use Hx
    Ut,_,_ = torch.linalg.svd(Hz_L)
    Hx_sub = Us[:,0:k]# Us[:,0:5]
    
    # Target domain subspace streaming
    Uhat = linalg.orth(np.random.randn(Hx_sub.shape[0],Hx_sub.shape[1]))
    Utrue = Ut[:,0:k]
    Hz_sub,_ = ojas(Hz_L.T.cpu().detach().numpy(),Uhat,Utrue.cpu().detach().numpy())
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

    return Xa.cpu().detach().numpy(), Za.cpu().detach().numpy(), Ys_H.T, Yt_H.T