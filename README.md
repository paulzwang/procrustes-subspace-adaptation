# Procrustes Subspace Adaptation

This repository contains Python code used to demonstrate the results presented in Section 5 of the following AIAA SciTech conference paper.

> Paul Wang, Olivia J. Pinon Fischer, and Dimitri N. Mavris. "Adaptive Digital Twins: Continuous Subspace Learning for Dynamic Domains". _AIAA Scitech 2026 Forum_, January 2026.

## Content 
### Demo Files
- `single_shift_demo.py` contains code to run Procrustes subspace adaptation for a single domain shift (i.e., a shift downwards in the aerobraking orbit periapsis).
- `increasing_shift_demo.py` contains code to run Procrustes subspace adaptation on multiple domain shifts (i.e., multiple shifts in periapsis).
- `tutorial.ipynb` is a Python notebook that gives a step-by-step tutorial on the subspace adaptation learning algorithm. [This is a work in progress]

### Functions
- `psa.py` contains the learning algorithm for Procrustes subspace adaptation.
- `utils.py` contains functions for the PyTorch model architecture, model training, and data visualization.

### Data and Models
- The folder `data` contains all aerobraking trajectory data as CSV files. Subfolders within `data` indicate the number of orbits, the apoapsis radius, and the periapsis radius for the aerobraking mission, each of which contains corresponding CSV trajectory data. All data was generated from Falcone et al.'s Aerobraking Trajectory Simulator (https://github.com/gfalcon2/Aerobraking-Trajectory-Simulator)
- The folder `models` contains saved PyTorch model weights and biases as `.pt` files for the baseline neural network and the subspace adapted network. The input layer size, output layer size, number of layers, number of neurons, and hidden layer size are saved under `model_settings.txt` files in the `models` folder.

## Usage
1. Python is required to run all code. Install all packages in `requirements.txt` into the Python environment.
2. Run `single_shift_demo.py` or `increasing_shift_demo.py` to produce results, which can be found in the `plots` folder as PDFs.