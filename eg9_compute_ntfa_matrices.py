"""
Demo code for plastic mode identification and processing, i.e. computation of the system matrices
"""
#%%
from operator import itemgetter

import numpy.linalg as la
import matplotlib.pyplot as plt
import time
from microstructures import *
from utilities import read_h5, read_snapshots, mode_identification, compute_tabular_data, save_tabular_data

np.random.seed(0)
for microstructure in microstructures[-2:]:
    file_name, data_path, temp1, temp2, n_tests, sampling_alphas = itemgetter(
        "file_name", "data_path", "temp1", "temp2", "n_tests", "sampling_alphas"
    )(microstructure)
    print(file_name, "\t", data_path)

    sample_temperatures = np.linspace(temp1, temp2, num=n_tests)

    mesh, samples = read_h5(file_name, data_path, sample_temperatures)
    mat_id = mesh["mat_id"]
    n_gauss = mesh["n_gauss"]
    strain_dof = mesh["strain_dof"]
    nodal_dof = mesh["nodal_dof"]
    n_elements = mesh["n_elements"]
    n_integration_points = mesh["n_integration_points"]
    global_gradient = mesh["global_gradient"]
    n_gp = mesh["n_integration_points"]
    disc = mesh["combo_discretisation"]
    vol_frac = mesh['volume_fraction'][0]

    # Mode identification

    # Read plastic snapshots from h5 file
    plastic_snapshots = read_snapshots(file_name, data_path)
    print('plastic_snapshots.shape:', plastic_snapshots.shape)

    # Identification of plastic modes
    r_min = 1e-8
    plastic_modes_svd = mode_identification(plastic_snapshots, vol_frac, r_min)
    print('plastic_modes_svd.shape:', plastic_modes_svd.shape)

    # Compare computed plastic modes with plastic modes from h5 file
    plastic_modes = samples[0]['plastic_modes']
    assert np.allclose(plastic_modes, plastic_modes_svd), 'Identified plastic modes do not match plastic modes in h5 file'

    # Mode processing to compute system matrices (after eigenstress problems have been solved using FANS)

    n_temp = 100
    temperatures = np.linspace(temp1, temp2, num=n_temp)
    start_time = time.time()
    # TODO: compute system matrices for multiple intermediate temperatures in an efficient way
    A_bar, D_xi, tau_theta, C_bar = compute_tabular_data(samples, mesh, temperatures)
    elapsed_time = time.time() - start_time
    print(f'Computed tabular data for {n_temp} temperatures in {elapsed_time}s')
    print('A_bar.shape:', A_bar.shape)
    print('D_xi.shape:', D_xi.shape)
    print('tau_theta.shape:', tau_theta.shape)
    print('C_bar.shape:', C_bar.shape)

    # Save system matrices for multiple temperatures as tabular data
    save_tabular_data(file_name, data_path, temperatures, A_bar, D_xi, tau_theta, C_bar)
    # Tabular data is saved to the input h5 file and can be copied to a new h5 file using e.g.
    # h5copy -i input/file.h5 -o input/file_ntfa.h5 -s ms_1p/dset0_ntfa -d ms_1p/dset0_ntfa -p

#%% Compare interpolated NTFA matrices with exact NTFA matrices
# TODO: compute errors
