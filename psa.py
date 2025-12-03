import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import make_interp_spline
import scipy.linalg as linalg
from scipy.linalg import orth

def interpolate_inputs(Xs,Xt,ts,tt):
    if len(ts) > len(tt):
        bs = make_interp_spline(ts,Xs)
        Xs_interp = bs(tt)
        Xt_interp = Xt
    else:
        bt = make_interp_spline(tt,Xt) 
        Xs_interp = Xs
        Xt_interp = bt(ts)
    return Xs_interp, Xt_interp

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

def procrustes_manifold_alignment(Xs, Xt, Ys, Yt, ts, tt, window_length, k=5, rotation=True, scaling=True):
    # Set training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    L = window_length

    # Mean center and scale data to unit variance
    scaler = StandardScaler(with_std=True)
    Xs = scaler.fit_transform(Xs)
    Xt = scaler.fit_transform(Xt)

    # Interpolate inputs
    Xs_interp, Xt_interp = interpolate_inputs(Xs,Xt,ts,tt)

    # Inputs are in the column space of X
    for i in range(0,np.size(Xs,1)):
        Hs_Li = torch.tensor(find_trajectory_matrix(Xs_interp[:,i],L),dtype=torch.float32).to(device)
        Ht_Li = torch.tensor(find_trajectory_matrix(Xt_interp[:,i],L),dtype=torch.float32).to(device)
        Hsi = torch.tensor(find_trajectory_matrix(Xs[:,i],L),dtype=torch.float32).to(device)
        Hti = torch.tensor(find_trajectory_matrix(Xt[:,i],L),dtype=torch.float32).to(device)

        # Stack Hankel matrices. Two sets of stacks are generated; one for linked data and one for the full datasets.
        # Linked data is used to find rotation matrix.
        if i==0:
            Hs_L = Hs_Li
            Ht_L = Ht_Li
            Hs = Hsi
            Ht = Hti
        else:
            Hs_L = torch.cat((Hs_L,Hs_Li),dim=0)
            Ht_L = torch.cat((Ht_L,Ht_Li),dim=0)
            Hs = torch.cat((Hs,Hsi),dim=0)
            Ht = torch.cat((Ht,Hti),dim=0)

    Us,_,_ = torch.linalg.svd(Hs_L)
    Ut,_,_ = torch.linalg.svd(Ht_L)
    Hs_sub = Us[:,0:k]# Us[:,0:5]
    Ht_sub = Ut[:,0:k]

    Hs_proj = Hs_sub.T @ Hs_L
    Ht_proj = Ht_sub.T @ Ht_L

    U,S,V = torch.linalg.svd(Hs_proj @ Ht_proj.T)
    Q = V.T @ U.T
    s = torch.trace(torch.diag(S))/torch.trace(Hs_proj @ Hs_proj.T)

    if rotation and scaling:
        Xsa = s * Q @ (Hs_sub.T @ Hs)
    elif not rotation and scaling:
        Xsa = s * (Hs_sub.T @ Hs)
    elif rotation and not scaling:
        Xsa = Q @ (Hs_sub.T @ Hs)
    elif not rotation and not scaling:
        Xsa = Hs_sub.T @ Hs
    else:
        Xsa = s * Q @ (Hs_sub.T @ Hs)

    Xta = Ht_sub.T @ Ht
    Xsa = Xsa.T
    Xta = Xta.T
    # Hankelise outputs
    Ys_H = find_trajectory_matrix(Ys,L)
    Yt_H = find_trajectory_matrix(Yt,L)

    return Xsa.cpu().detach().numpy(), Xta.cpu().detach().numpy(), Ys_H.T, Yt_H.T

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

def streaming_procrustes_manifold_alignment(Xs, Xt, Ys, Yt, ts, tt, window_length, k=5, rotation=True, scaling=True):
    # Set training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    L = window_length

    # Mean center and scale data to unit variance
    scaler = StandardScaler(with_std=True)
    Xs = scaler.fit_transform(Xs)
    Xt = scaler.fit_transform(Xt)

    # Interpolate inputs
    Xs_interp, Xt_interp = interpolate_inputs(Xs,Xt,ts,tt)

    # Inputs are in the column space of X
    for i in range(0,np.size(Xs,1)):
        Hs_Li = torch.tensor(find_trajectory_matrix(Xs_interp[:,i],L),dtype=torch.float32).to(device)
        Ht_Li = torch.tensor(find_trajectory_matrix(Xt_interp[:,i],L),dtype=torch.float32).to(device)
        Hsi = torch.tensor(find_trajectory_matrix(Xs[:,i],L),dtype=torch.float32).to(device)
        Hti = torch.tensor(find_trajectory_matrix(Xt[:,i],L),dtype=torch.float32).to(device)

        # Stack Hankel matrices. Two sets of stacks are generated; one for linked data and one for the full datasets.
        # Linked data is used to find rotation matrix.
        if i==0:
            Hs_L = Hs_Li
            Ht_L = Ht_Li
            Hs = Hsi
            Ht = Hti
        else:
            Hs_L = torch.cat((Hs_L,Hs_Li),dim=0)
            Ht_L = torch.cat((Ht_L,Ht_Li),dim=0)
            Hs = torch.cat((Hs,Hsi),dim=0)
            Ht = torch.cat((Ht,Hti),dim=0)

    Us,_,_ = torch.linalg.svd(Hs) # Change this line to use Hs
    Ut,_,_ = torch.linalg.svd(Ht_L)
    Hs_sub = Us[:,0:k]# Us[:,0:5]
    
    # Target domain subspace streaming
    Uhat = linalg.orth(np.random.randn(Hs_sub.shape[0],Hs_sub.shape[1]))
    Utrue = Ut[:,0:k]
    Ht_sub,_ = ojas(Ht_L.T.cpu().detach().numpy(),Uhat,Utrue.cpu().detach().numpy())
    Ht_sub = torch.tensor(Ht_sub,dtype=torch.float32).to(device)

    Hs_proj = Hs_sub.T @ Hs_L
    Ht_proj = Ht_sub.T @ Ht_L

    U,S,V = torch.linalg.svd(Hs_proj @ Ht_proj.T)
    Q = V.T @ U.T
    s = torch.trace(torch.diag(S))/torch.trace(Hs_proj @ Hs_proj.T)

    if rotation and scaling:
        Xsa = s * Q @ (Hs_sub.T @ Hs)
    elif not rotation and scaling:
        Xsa = s * (Hs_sub.T @ Hs)
    elif rotation and not scaling:
        Xsa = Q @ (Hs_sub.T @ Hs)
    elif not rotation and not scaling:
        Xsa = Hs_sub.T @ Hs
    else:
        Xsa = s * Q @ (Hs_sub.T @ Hs)

    Xta = Ht_sub.T @ Ht
    Xsa = Xsa.T
    Xta = Xta.T
    # Hankelise outputs
    Ys_H = find_trajectory_matrix(Ys,L)
    Yt_H = find_trajectory_matrix(Yt,L)

    return Xsa.cpu().detach().numpy(), Xta.cpu().detach().numpy(), Ys_H.T, Yt_H.T