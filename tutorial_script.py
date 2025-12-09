import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from utils import NeuralNetwork
from utils import train_model
import psa

import torch 
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

mission1_data_directory = r'data\4orbit_ra=12000_rp=100.0\Results_ctrl=0_ra=12000_rp=100.0_hl=0.150_90.0deg.csv'
mission2_data_directory = r'data\4orbit_ra=12000_rp=95.0\Results_ctrl=0_ra=12000_rp=95.0_hl=0.150_90.0deg.csv'
df1 = pd.read_csv(mission1_data_directory)
df2 = pd.read_csv(mission2_data_directory)

# Plotting
fig1, ax1 = plt.subplots(1,1)
ax1.plot(df1['time'], df1['rho'], label="Source Domain", color='black')
ax1.plot(df2['time'], df2['rho'], label="Target Domain", color='gray')
ax1.set_xlabel('Time ($s$)')
ax1.set_ylabel('Atmospheric Density (kg/m$^3$)') 
ax1.legend()

fig2, ax2 = plt.subplots(1,1)
ax2.plot(df1['time'], df1['heat_rate'], label="Source Domain", color='black')
ax2.plot(df2['time'], df2['heat_rate'], label="Target Domain", color='gray')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Heat Rate (W/cm$^2$)')
ax2.legend() 

# Plotting state space
fig3, ax3 = plt.subplots(1,1)
ax3.scatter(df1['rho'], df1['heat_rate'], s=2, label="Source Domain", color='black')
ax3.scatter(df2['rho'], df2['heat_rate'], s=2, label="Target Domain", color='gray')
ax3.set_xlabel('Atmospheric Density ($kg/m^3$)')
ax3.set_ylabel('Heat Rate ($W/{cm}^3$)') 
ax3.legend()

#plt.show()

L = 50 # The window length.
N = len(df1['rho'])
K = N - L + 1 # The number of columns in the trajectory matrix.
# Create the trajectory matrix by pulling the relevant subseries of F, and stacking them as columns
X = np.column_stack([df1['rho'][i:i+L] for i in range(0,K)])
# Note: the i+L above gives us up to i+L-1, as numpy array upper bounds are exclusive.

ax = plt.matshow(X)
plt.xlabel("$L$-Lagged Vectors")
plt.ylabel("$K$-Lagged Vectors")
plt.show()