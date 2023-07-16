"""
Demo code for plastic mode identification and processing, i.e. computation of the system matrices
"""
#%%
from operator import itemgetter

import numpy.linalg as la
import matplotlib.pyplot as plt
import time
from microstructures import *
from utilities import read_h5
from ntfa import read_snapshots, mode_identification, compute_tabular_data, save_tabular_data

np.random.seed(0)
for microstructure in microstructures:
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
    vol_frac0, vol_frac1 = mesh['volume_fraction'][0], mesh['volume_fraction'][1]

    # Mode identification

    # Read plastic snapshots from h5 file
    plastic_snapshots = read_snapshots(file_name, data_path)
    print('plastic_snapshots.shape:', plastic_snapshots.shape)

    # Identification of plastic modes
    r_min = 1e-3
    plastic_modes_svd = mode_identification(plastic_snapshots, vol_frac0, r_min)
    print('plastic_modes_svd.shape:', plastic_modes_svd.shape)

    # Compare computed plastic modes with plastic modes from h5 file
    plastic_modes = samples[0]['plastic_modes']
    plastic_modes = plastic_modes_svd
    assert np.allclose(plastic_modes / np.sign(np.expand_dims(np.expand_dims(plastic_modes[0,0,:], axis=0), axis=0)), plastic_modes_svd), 'Identified plastic modes do not match plastic modes in h5 file'

    # Mode processing to compute system matrices

    n_temp = 1000
    temperatures = np.linspace(temp1, temp2, num=n_temp)
    start_time = time.time()
    # TODO: compute system matrices for multiple intermediate temperatures in an efficient way
    C_bar, tau_theta, A_bar, tau_xi, D_xi, D_theta, A0, A1, C0, C1 = compute_tabular_data(samples, mesh, temperatures)
    elapsed_time = time.time() - start_time
    print(f'Computed tabular data for {n_temp} temperatures in {elapsed_time}s')
    print('C_bar.shape:', C_bar.shape)
    print('tau_theta.shape:', tau_theta.shape)
    print('A_bar.shape:', A_bar.shape)
    print('tau_xi.shape:', tau_xi.shape)
    print('D_xi.shape:', D_xi.shape)
    print('D_theta.shape:', D_theta.shape)
    print('A_bar/A0/A1 error:', np.linalg.norm(vol_frac0 * A0[:, strain_dof + 1:] + vol_frac1 * A1[:, strain_dof + 1:] - A_bar) / np.linalg.norm(A_bar))

    # Save system matrices for multiple temperatures as tabular data
    save_tabular_data(file_name, data_path, temperatures, C_bar, tau_theta, A_bar, tau_xi, D_xi, D_theta, A0, A1, C0, C1)
    # Tabular data is saved to the input h5 file and can be copied to a new h5 file using e.g.
    # h5copy -i input/file.h5 -o input/file_ntfa.h5 -s ms_1p/dset0_ntfa -d ms_1p/dset0_ntfa -p

#%% Compare interpolated NTFA matrices with exact NTFA matrices
# TODO: compute errors
