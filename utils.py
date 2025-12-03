import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import erf
import pandas as pd
import matplotlib.pyplot as plt

def plot_domain_visualization(source_directory,target_directory):
    df1 = pd.read_csv(source_directory)
    df2 = pd.read_csv(target_directory)

    # Plotting
    fig1, ax1 = plt.subplots(2,2)
    ax1[0,0].plot(df1['time'], df1['rho'], label="Source Domain", color='black')
    ax1[0,0].plot(df2['time'], df2['rho'], label="Target Domain", color='gray')
    ax1[0,0].set_xlabel('Time ($s$)')
    ax1[0,0].set_ylabel('Atmospheric Density (kg/m$^3$)') 
    ax1[0,0].legend()

    ax1[0,1].plot(df1['time'], df1['T'], label="Source Domain",  color='black')
    ax1[0,1].plot(df2['time'], df2['T'], label="Target Domain", color='gray')
    ax1[0,1].set_xlabel('Time ($s$)')
    ax1[0,1].set_ylabel('Freestream Temperature (K)') 
    ax1[0,1].legend()

    ax1[1,0].plot(df1['time'], df1['S'], label="Source Domain", color='black')
    ax1[1,0].plot(df2['time'], df2['S'], label="Target Domain", color='gray')
    ax1[1,0].set_xlabel('Time (s)')
    ax1[1,0].set_ylabel('Molecular Speed Ratio') 
    ax1[1,0].legend()

    ax1[1,1].plot(df1['time'], df1['aoa'], label="Source Domain", color='black')
    ax1[1,1].plot(df2['time'], df2['aoa'], label="Target Domain", color='gray')
    ax1[1,1].set_xlabel('Time (s)')
    ax1[1,1].set_ylabel('Angle of Attack (deg)') 
    ax1[1,1].legend()

    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(df1['time'], df1['heat_rate'], label="Source Domain", color='black')
    ax2.plot(df2['time'], df2['heat_rate'], label="Target Domain", color='gray')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Heat Rate (W/cm$^2$)')
    ax2.legend() 

    # Plotting state space
    fig3, ax3 = plt.subplots(2,2)
    ax3[0,0].scatter(df1['rho'], df1['heat_rate'], s=2, label="Source Domain", color='black')
    ax3[0,0].scatter(df2['rho'], df2['heat_rate'], s=2, label="Target Domain", color='gray')
    ax3[0,0].set_xlabel('Atmospheric Density ($kg/m^3$)')
    ax3[0,0].set_ylabel('Heat Rate ($W/{cm}^3$)') 
    ax3[0,0].legend()

    ax3[0,1].scatter(df1['T'], df1['heat_rate'], s=2, label="Source Domain", color='black')
    ax3[0,1].scatter(df2['T'], df2['heat_rate'], s=2, label="Target Domain", color='gray')
    ax3[0,1].set_xlabel('Freestream Temperature ($K$)')
    ax3[0,1].set_ylabel('Heat Rate ($W/{cm}^3$)') 
    ax3[0,1].legend()

    ax3[1,0].scatter(df1['S'], df1['heat_rate'], s=2, label="Source Domain", color='black')
    ax3[1,0].scatter(df2['S'], df2['heat_rate'], s=2, label="Target Domain", color='gray')
    ax3[1,0].set_xlabel('Molecular Speed Ratio')
    ax3[1,0].set_ylabel('Heat Rate ($W/{cm}^3$)') 
    ax3[1,0].legend()

    ax3[1,1].scatter(df1['aoa'], df1['heat_rate'], s=2, label="Source Domain", color='black')
    ax3[1,1].scatter(df2['aoa'], df2['heat_rate'], s=2, label="Target Domain", color='gray')
    ax3[1,1].set_xlabel('Angle of Attack (deg)')
    ax3[1,1].set_ylabel('Heat Rate ($W/{cm}^3$)') 
    ax3[1,1].legend()

    plt.show()

def add_percent_noise(arr, percent_noise):
    """
    Adds noise to a NumPy array where the noise magnitude is a percentage
    of the absolute values in the original array.

    Args:
        arr (np.ndarray): The input NumPy array.
        percent_noise (float): The percentage of noise to add (e.g., 0.1 for 10%).

    Returns:
        np.ndarray: The array with added noise.
    """
    # Generate random noise with the same shape as the input array.
    # Using np.random.randn for Gaussian noise with mean 0 and standard deviation 1.
    noise = np.random.randn(*arr.shape)

    # Scale the noise by the desired percentage of the absolute values of the array.
    # This ensures that the noise magnitude is proportional to the original values.
    scaled_noise = noise * (np.abs(arr) * percent_noise)

    # Add the scaled noise to the original array.
    noisy_arr = arr + scaled_noise

    return noisy_arr

def calculate_heat_rate(rho,T,S,aoa):
    taf = 1 # Thermal accomodation factor, default value is 1
    R = 188.92  # J/KgK
    gamma = 1.33 # Isentropic exponent
    calculated_heat_rate = (taf*rho*R*T) * ((R*T/(2.0*np.pi))**0.5) * (
        (S**2.0 + (gamma)/(gamma - 1.0) - (gamma + 1.0) / (2.0 * (gamma - 1))) * (
            np.exp(-(S * np.sin(aoa)) ** 2.0) + (np.pi ** 0.5) * (S * np.sin(aoa)) * 
            (1 + erf(S * np.sin(aoa)))) - 0.5 * np.exp(-(S * np.sin(aoa)) ** 2.0)) * 1e-4  # W/cm^2

    return calculated_heat_rate

# Define the neural network model class with flexible architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(NeuralNetwork, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        x = x.type(torch.float32)
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        outputs = self.output_layer(x)
        return outputs

def train_model(model,X,Y,num_epochs,learning_rate,weight_decay=1e-5):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        model.train()
        # Zero gradient
        optimizer.zero_grad()
        # Make predictions
        outputs = model(X)
        # Compute the loss and its gradients
        loss = loss_fn(outputs,Y)
        loss.backward()
        # Adjust learning weights
        optimizer.step()
    return loss.item()