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
from ntfa import read_tabular_data

file_name_10, data_path_10, temp1_10, temp2_10, n_tests_10, sampling_alphas_10 = itemgetter(
        "file_name", "data_path", "temp1", "temp2", "n_tests", "sampling_alphas"
)(microstructures[0])

temperatures_10, C_bar_10, tau_theta_10, A_bar_10, tau_xi_10, D_xi_10, D_theta_10, A0_10, A1_10, C0_10, C1_10 = \
    read_tabular_data(file_name_10, data_path_10)

file_name_100, data_path_100, temp1_100, temp2_100, n_tests_100, sampling_alphas_100 = itemgetter(
        "file_name", "data_path", "temp1", "temp2", "n_tests", "sampling_alphas"
)(microstructures[-1])

temperatures_100, C_bar_100, tau_theta_100, A_bar_100, tau_xi_100, D_xi_100, D_theta_100, A0_100, A1_100, C0_100, C1_100 = \
    read_tabular_data(file_name_100, data_path_100)

C_bar_diff = C_bar_10 - C_bar_100
tau_theta_diff = tau_theta_10 - tau_theta_100
# A_bar_diff
# tau_xi_diff
# D_xi_diff
D_theta_diff = tau_theta_10 - tau_theta_100
# A0_diff
# A1_diff
C0_diff = C0_10 - C0_100
C1_10 = C1_10 - C1_100
print('C_bar error:', np.linalg.norm(C_bar_diff, axis=(0,1)) / np.linalg.norm(C_bar_100, axis=(0,1)))
print('tau_theta error:', np.linalg.norm(tau_theta_diff, axis=0) / np.linalg.norm(tau_theta_100, axis=0))
print('D_theta error:', np.linalg.norm(D_theta_diff, axis=0) / np.linalg.norm(D_theta_100, axis=0))
print(C_bar_10.shape, C_bar_100.shape)