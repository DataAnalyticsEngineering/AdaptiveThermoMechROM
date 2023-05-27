"""
Demo code for plastic mode identification and processing, i.e. computation of the system matrices
"""
#%%
from operator import itemgetter

import numpy.linalg as la
import matplotlib.pyplot as plt
from microstructures import *
from utilities import read_h5, read_snapshots, mode_identification, mode_processing, save_tabular_data

np.random.seed(0)
file_name, data_path, temp1, temp2, n_tests, sampling_alphas = itemgetter(
    "file_name", "data_path", "temp1", "temp2", "n_tests", "sampling_alphas"
)(microstructures[0])
print(file_name, "\t", data_path)

temperatures = np.linspace(temp1, temp2, num=n_tests)

mesh, samples = read_h5(file_name, data_path, temperatures)
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
plastic_snapshots = read_snapshots(file_name, data_path, temperatures)
print('plastic_snapshots.shape:', plastic_snapshots.shape)

# Mode identification using POD
r_min = 6e-5
plastic_modes = mode_identification(plastic_snapshots, r_min)
print('plastic_modes.shape:', plastic_modes.shape)

# TODO: save identified plastic modes to h5 file

#%% Mode processing to compute system matrices (after eigenstress problems have been solved using FANS)

# TODO: compute system matrices for multiple temperatures in an efficient way
sample = samples[0]  # For now, choose one arbitrary sample
# plastic_modes = sample['plastic_modes'] (until we have real data)
N_modes = plastic_modes.shape[2]
strain_localization = sample["strain_localization"]
# Add dummy data to strain_localization until we have real data:
strain_localization = np.concatenate([strain_localization, np.random.rand(n_integration_points, strain_dof, N_modes)], axis=2)
mat_stiffness = sample["mat_stiffness"]
mat_thermal_strain = sample["mat_thermal_strain"]
A_bar, D_xi, tau_theta, C_bar = mode_processing(strain_localization, mat_stiffness, mat_thermal_strain, plastic_modes, mesh)
print('A_bar.shape:', A_bar.shape)
print('D_xi.shape:', D_xi.shape)
print('tau_theta.shape:', tau_theta.shape)
print('C_bar.shape:', C_bar.shape)

# TODO: compute system matrices for multiple intermediate temperatures in an efficient way

# TODO: save system matrices for multiple temperatures as tabular data
save_tabular_data(file_name, data_path, temperatures, A_bar, D_xi, tau_theta, C_bar)
