"""
Plot the strain localization operator E and stress localization operator S at different temperatures
"""
#%%
from operator import itemgetter

import numpy.linalg as la
import matplotlib.pyplot as plt
from microstructures import *
from utilities import read_h5, mode_identification, mode_processing

np.random.seed(0)
file_name, data_path, temp1, temp2, n_tests, sampling_alphas = itemgetter(
    "file_name", "data_path", "temp1", "temp2", "n_tests", "sampling_alphas"
)(microstructures[0])
print(file_name, "\t", data_path)

test_temperatures = np.linspace(temp1, temp2, num=n_tests)
test_alphas = np.linspace(0, 1, num=n_tests)

mesh, samples = read_h5(file_name, data_path, test_temperatures)
mat_id = mesh["mat_id"]
n_gauss = mesh["n_gauss"]
strain_dof = mesh["strain_dof"]
nodal_dof = mesh["nodal_dof"]
n_elements = mesh["n_elements"]
n_integration_points = mesh["n_integration_points"]
global_gradient = mesh["global_gradient"]
n_gp = mesh["n_integration_points"]
disc = mesh["combo_discretisation"]

#%% Mode identification

# TODO: Read plastic snapshots from h5 file
N_modes = 10
plastic_snapshots = np.random.rand(n_integration_points, strain_dof, N_modes)

# Mode identification using POD
r_min = 1e-3
plastic_modes = mode_identification(plastic_snapshots, r_min)

# TODO: save identified plastic modes to h5 file

#%% Mode processing to compute system matrices (after snapshots have been computed using FANS)

# TODO: compute system matrices for multiple temperatures in an efficient way
strain_localization = samples[0]["strain_localization"]
mat_stiffness = samples[0]["mat_stiffness"]
A_bar, D_xi, tau_theta, C_bar = mode_processing(strain_localization, mat_stiffness, mesh, plastic_modes)

# TODO: save system matrices for multiple temperatures as tabular data
