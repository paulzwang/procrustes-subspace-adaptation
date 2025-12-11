import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import utils
from utils import NeuralNetwork
from utils import train_model
import psa

import torch 
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import make_interp_spline
import scipy.linalg as linalg
from scipy.linalg import orth

#==========================================================================================================#

mission1_data_directory = r'data\4orbit_ra=12000_rp=100.0\Results_ctrl=0_ra=12000_rp=100.0_hl=0.150_90.0deg.csv'
mission2_data_directory = r'data\4orbit_ra=12000_rp=92.0\Results_ctrl=0_ra=12000_rp=92.0_hl=0.150_90.0deg.csv'
df1 = pd.read_csv(mission1_data_directory).drop_duplicates(subset=['time'], keep='first')
df2 = pd.read_csv(mission2_data_directory).drop_duplicates(subset=['time'], keep='first')

# Plotting
fig0, ax0 = plt.subplots(1,3,figsize=(15,5))
ax0[0].plot(df1['time'], df1['rho'], label="Source Domain", color='black')
ax0[0].plot(df2['time'], df2['rho'], label="Target Domain", color='gray')
ax0[0].set_xlabel('Time ($s$)')
ax0[0].set_ylabel('Atmospheric Density (kg/m$^3$)') 
ax0[0].legend()

ax0[1].plot(df1['time'], df1['heat_rate'], label="Source Domain", color='black')
ax0[1].plot(df2['time'], df2['heat_rate'], label="Target Domain", color='gray')
ax0[1].set_xlabel('Time (s)')
ax0[1].set_ylabel('Heat Rate (W/cm$^2$)')
ax0[1].legend() 

# Plotting state space
ax0[2].scatter(df1['rho'], df1['heat_rate'], s=2, label="Source Domain", color='black')
ax0[2].scatter(df2['rho'], df2['heat_rate'], s=2, label="Target Domain", color='gray')
ax0[2].set_xlabel('Atmospheric Density ($kg/m^3$)')
ax0[2].set_ylabel('Heat Rate ($W/{cm}^3$)') 
ax0[2].legend()
fig0.tight_layout()

#==========================================================================================================#

from psa import find_trajectory_matrix, H_to_TS, interpolate_inputs

t1 = np.array(df1['time'])
rho1 = np.array(df1['rho']).reshape(-1,1)
Q1 = np.array(df1['heat_rate']).reshape(-1,1)

t2 = np.array(df2['time'])
rho2 = np.array(df2['rho']).reshape(-1,1)
Q2 = np.array(df2['heat_rate']).reshape(-1,1)

# Mean center and scale inputs to unit variance
scaler = StandardScaler(with_std=True)
rho1 = scaler.fit_transform(rho1)
rho2 = scaler.fit_transform(rho2)

# Interpolate inputs
rho1_interp, rho2_interp = interpolate_inputs(rho1,rho2,t1,t2)

#==========================================================================================================#

# Non-interpolated trajectory matrices
H_rho1 = torch.tensor(find_trajectory_matrix(rho1,window_length=50),dtype=torch.float32)
H_rho2 = torch.tensor(find_trajectory_matrix(rho2,window_length=50),dtype=torch.float32)

# Interpolated trajectory matrices
H_rho1_interp = find_trajectory_matrix(rho1_interp,window_length=50)
H_rho2_interp = find_trajectory_matrix(rho2_interp,window_length=50)

# Plotting trajectory matrices
print(f'Plotting trajectory matrices. Their dimensions are {H_rho1_interp.shape}.')
fig0, ax0 = plt.subplots(2,1,figsize=(20,10))
fig0.subplots_adjust(hspace=-0.85)
ax0[0].matshow(H_rho1_interp)
ax0[1].matshow(H_rho2_interp)
ax0[0].set_title('Source Domain')
ax0[1].set_title('Target Domain')

# Plotting reshaped trajectory matrices
H_rho1_square = H_rho1_interp.reshape(440,440)
H_rho2_square = H_rho2_interp.reshape(440,440)
print(f'Plotting reshaped trajectory matrices for purely visualization purposes. Their dimensions are: {H_rho1_square.shape}.')
fig1, ax1 = plt.subplots(1,2)
ax1[0].matshow(H_rho1_square)
ax1[1].matshow(H_rho2_square)
ax1[0].set_title('Source Domain')
ax1[1].set_title('Target Domain')

#==========================================================================================================#

H_rho1_interp = torch.tensor(H_rho1_interp,dtype=torch.float32)
H_rho2_interp = torch.tensor(H_rho2_interp,dtype=torch.float32)
Us,Ss,Vs = torch.linalg.svd(H_rho1_interp)
Ut,St,Vt = torch.linalg.svd(H_rho2_interp)

#==========================================================================================================#

ds = np.linalg.matrix_rank(H_rho1_interp)
H_rho1_elem = np.array([Ss[i] * np.outer(Us[:,i], Vs[:,i]) for i in range(0,ds)])

num_subplots = 6
fig2, ax2 = plt.subplots(num_subplots,1,figsize=(20,16))
fig2.subplots_adjust(hspace=-0.9)
for i in range(num_subplots):
    ax2[i].matshow(H_rho1_elem[i])
    ax2[i].set_title(f"Source Domain, $H_{str(i)}$")
    # Remove x-axis ticks and labels
    ax2[i].set_xticks([])
    ax2[i].set_xticklabels([])
    # Remove y-axis ticks and labels
    ax2[i].set_yticks([])
    ax2[i].set_yticklabels([])

fig3, ax3 = plt.subplots(1,num_subplots,figsize=(20,16))
for i in range(num_subplots):
    ax3[i].matshow(H_rho1_elem[i].reshape(440,440))
    ax3[i].set_title(f"Source Domain, Reshaped $H_{str(i)}$")
    # Remove x-axis ticks and labels
    ax3[i].set_xticks([])
    ax3[i].set_xticklabels([])
    # Remove y-axis ticks and labels
    ax3[i].set_yticks([])
    ax3[i].set_yticklabels([])
fig3.tight_layout()

#==========================================================================================================#

dt = np.linalg.matrix_rank(H_rho2_interp)
H_rho2_elem = np.array([St[i] * np.outer(Ut[:,i], Vt[:,i]) for i in range(0,dt)])

num_subplots = 6
fig4, ax4 = plt.subplots(num_subplots,1,figsize=(20,16))
fig4.subplots_adjust(hspace=-0.9)
for i in range(num_subplots):
    ax4[i].matshow(H_rho2_elem[i])
    ax4[i].set_title(f"Target Domain, $H_{str(i)}$")
    # Remove x-axis ticks and labels
    ax4[i].set_xticks([])
    ax4[i].set_xticklabels([])
    # Remove y-axis ticks and labels
    ax4[i].set_yticks([])
    ax4[i].set_yticklabels([])

fig5, ax5 = plt.subplots(1,num_subplots,figsize=(20,16))
for i in range(num_subplots):
    ax5[i].matshow(H_rho2_elem[i].reshape(440,440))
    ax5[i].set_title(f"Target Domain, Reshaped $H_{str(i)}$")
    # Remove x-axis ticks and labels
    ax5[i].set_xticks([])
    ax5[i].set_xticklabels([])
    # Remove y-axis ticks and labels
    ax5[i].set_yticks([])
    ax5[i].set_yticklabels([])
fig5.tight_layout()

#==========================================================================================================#

# Calculate contribution of each principal component
sigma_sumsq = (Ss**2).sum()
print(Ss**2 / sigma_sumsq * 100) 
plt.plot(Ss**2 / sigma_sumsq * 100)
plt.xlabel('Component Number')
plt.ylabel('Explained Variance')

#==========================================================================================================#

k = 5
H_rho1_sub = Us[:,0:k]
H_rho1_sub = torch.tensor(H_rho1_sub,dtype=torch.float32)

#==========================================================================================================#

Y = H_rho2_interp.T.cpu().detach().numpy()
n_samples, n_features = Y.shape
eta = 0.5
errors = []
Uhat = linalg.orth(np.random.randn(H_rho1_sub.shape[0],H_rho1_sub.shape[1]))
Utrue = Ut[:,0:k].cpu().detach().numpy()
for i in range(n_samples):
    y = Y[i, :].reshape(-1, 1)  # Column vector
    # Oja's update
    Uhat += eta * y @ (Uhat.T @ y).T 
    # Re-orthogonalize U
    Uhat = orth(Uhat)
    error = linalg.norm(Utrue - Uhat @ (Uhat.T @ Utrue),ord='fro')**2
    errors.append(error)
H_rho2_sub = Uhat
H_rho2_sub = torch.tensor(H_rho2_sub,dtype=torch.float32)

#==========================================================================================================#

""" Procrustes Subspace Adaptation """
H_rho1_proj = H_rho1_sub.T @ H_rho1_interp
H_rho2_proj = H_rho2_sub.T @ H_rho2_interp

U,S,V = torch.linalg.svd(H_rho1_proj @ H_rho2_proj.T)
Q = V.T @ U.T
s = torch.trace(torch.diag(S))/torch.trace(H_rho1_proj @ H_rho1_proj.T)

Xa = s * Q @ (H_rho1_sub.T @ H_rho1_interp)

Za = H_rho2_sub.T @ H_rho2_interp
Xa = Xa.T
Za = Za.T
# Hankelise outputs
Ys_H = find_trajectory_matrix(Q1,window_length=50)
Yt_H = find_trajectory_matrix(Q2,window_length=50)

Xa = Xa.cpu().detach().numpy()
Za = Za.cpu().detach().numpy()
Ys_H = Ys_H.T
Yt_h = Yt_H.T

print(H_rho1.shape, H_rho2.shape)
print(H_rho1_sub.shape,H_rho2_sub.shape)
print(H_rho1_proj.shape,H_rho2_proj.shape)
print(Xa.shape,Za.shape)

""" Plotting Subspace Projections and Latent Space Projections """
colnum = 160
sproj1_reshaped = H_rho1_proj.T.cpu().detach().numpy().reshape(-1,colnum)
sproj2_reshaped = H_rho2_proj.T.cpu().detach().numpy().reshape(-1,colnum)
lproj1_reshaped = Xa.reshape(-1,colnum)
lproj2_reshaped = Za.reshape(-1,colnum)

datasets = [sproj1_reshaped, sproj2_reshaped, lproj1_reshaped, lproj2_reshaped]
# create a single norm to be shared across all images
norm = colors.Normalize(vmin=np.min(datasets), vmax=np.max(datasets))

fig, axs = plt.subplots(2, 2)
fig.suptitle('Multiple images')

images = []
for ax, data in zip(axs.flat, datasets):
    images.append(ax.matshow(data, norm=norm))

fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

# plt.show()

#==========================================================================================================#

""" Plotting discrepancy in subspace/latent space """
datasets = [np.abs(sproj1_reshaped - sproj2_reshaped), np.abs(lproj1_reshaped - lproj2_reshaped)]

# create a single norm to be shared across all images
norm = colors.Normalize(vmin=np.min(datasets), vmax=np.max(datasets))

fig, axs = plt.subplots(2, 1)
fig.suptitle('Multiple images')

images = []
for ax, data in zip(axs.flat, datasets):
    images.append(ax.matshow(data, norm=norm, cmap='plasma'))

fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

# plt.show()

""" Plotting distribution """
binnum = round(np.sqrt(len(df1['rho'])))
plt.figure()
plt.hist(df1['rho'], bins=binnum, alpha=0.6)
plt.hist(df2['rho'], bins=binnum, alpha=0.6)

binnum = round(np.sqrt(len(rho1)))
plt.figure()
plt.hist(rho1, bins=binnum, alpha=0.6)
plt.hist(rho2, bins=binnum, alpha=0.6)

binnum = round(np.sqrt(len(sproj1_reshaped.ravel())))
plt.figure()
plt.hist(sproj1_reshaped.ravel(), bins=binnum, alpha=0.6)
plt.hist(sproj2_reshaped.ravel(), bins=binnum, alpha=0.6)
plt.figure()
plt.hist(lproj1_reshaped.ravel(), bins=binnum, alpha=0.6)
plt.hist(lproj2_reshaped.ravel(), bins=binnum, alpha=0.6)
plt.show()


# fig6, ax6 = plt.subplots(2,2)
# colnum = 160
# vmin = np.min(H_rho1_proj.ravel().cpu().detach().numpy())
# vmax = np.max(H_rho1_proj.ravel().cpu().detach().numpy())
# cmap = 'viridis'


# H_rho1_proj_np = H_rho1_proj.cpu().detach().numpy()
# H_rho2_proj_np = H_rho2_proj.cpu().detach().numpy()
# im = ax6[0,0].matshow(H_rho1_proj_np.T.reshape(-1,colnum), vmin=vmin, vmax=vmax, cmap=cmap)
# ax6[1,0].matshow(H_rho2_proj_np.T.reshape(-1,colnum), vmin=vmin, vmax=vmax, cmap=cmap)
# ax6[0,1].matshow(Xa.reshape(-1,colnum), vmin=vmin, vmax=vmax, cmap=cmap)
# ax6[1,1].matshow(Za.reshape(-1,colnum), vmin=vmin, vmax=vmax, cmap=cmap)
# fig6.colorbar(im)
# fig6.tight_layout()

# vmin = np.min(H_rho1_proj.ravel().cpu().detach().numpy())
# vmax = np.max(H_rho1_proj.ravel().cpu().detach().numpy())
# cmap = 'inferno'

# fig7, ax7 = plt.subplots(1,2,figsize=(20,16))
# im1 = ax7[0].matshow(np.abs(H_rho1_proj_np.T.reshape(-1,colnum)-H_rho2_proj_np.T.reshape(-1,colnum)), cmap=cmap)
# im2 = ax7[1].matshow(np.abs(Xa.reshape(-1,colnum)-Za.reshape(-1,colnum)), cmap=cmap)
# fig7.colorbar(im1)
# fig7.colorbar(im2)

# plt.show()