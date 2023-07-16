"""
Interpolate the homogenized response (i.e. effective C and effective eps) at arbitrary temperatures
based on the approximations in eg3_hierarchical_sampling.py or eg4_hierarchical_sampling_efficient.py
"""
# %%
from operator import itemgetter
import time
import h5py
from scipy import interpolate
from microstructures import *

# Offline stage: Load precomputed optimal data
load_start = time.time()
ms_id = 6
level = 4
file_name, data_path, temp1, temp2, n_samples, sampling_alphas = itemgetter('file_name', 'data_path', 'temp1', 'temp2', 'n_tests',
                                                                            'sampling_alphas')(microstructures[ms_id])
opt_file = path(f'output/opt_{file_name.name}')
print(f'Loading precomputed data from {file_name}\t{data_path}')

sample_temperatures = np.linspace(temp1, temp2, num=n_samples)

opt_C = np.zeros((n_samples, 6, 6))
opt_eps = np.zeros((n_samples, 6))

try:
    with h5py.File(opt_file, 'r') as file:
        for idx, temperature in enumerate(sample_temperatures):
            opt_C[idx] = file[f'{data_path}_level{level}/eff_stiffness_{temperature:07.2f}'][:]
            # eg4_*.py saves (-1 * eff_thermal_strain) in the corresponding h5 file
            opt_eps[idx] = -1.0 * file[f'{data_path}_level{level}/eff_thermal_strain_{temperature:07.2f}'][:]
except Exception:
    print(f'Could not load precomputed data. Run eg4_*.py first for ms_id={ms_id}, level={level}.')
    exit()

load_time = time.time() - load_start
print(f'Loaded precomputed data at {n_samples} temperatures in {load_time:.5f} s')

# %%
# Online stage: linear interpolation between sampled data
online_start = time.time()
n_tests = 1000
test_temperatures = np.linspace(temp1, temp2, num=n_tests)


def staggered_model_online(test_temperatures, sample_temperatures, opt_C, opt_eps):
    """Interpolate the effective stiffness C and the effective thermal expansion eps
    at a list of temperatures in `test_temperatures` using linear interpolation
    based on precomputed `opt_C` and `opt_eps` at `sample_temperatures`.

    Args:
        test_temperatures (np.ndarray): array of temperatures with shape (N,)
        sample_temperatures (np.ndarray): array of temperatures with shape (n,)
        opt_C (np.ndarray): optimal C at n temperatures with shape (n,6,6)
        opt_eps (np.ndarray): optimal eps at n temperatures with shape (n,6)

    Returns:
        np.ndarray: array of interpolated C with shape (N,6,6)
        np.ndarray: array of interpolated eps with shape (N,6)
    """
    # Linear interpolation for C
    interp_C = interpolate.interp1d(sample_temperatures, opt_C, axis=0)
    approx_C = interp_C(test_temperatures)
    # Linear interpolation for eps
    interp_eps = interpolate.interp1d(sample_temperatures, opt_eps, axis=0)
    approx_eps = interp_eps(test_temperatures)
    return approx_C, approx_eps


approx_C, approx_eps = staggered_model_online(test_temperatures, sample_temperatures, opt_C, opt_eps)

online_time = time.time() - online_start
print(f'Interpolated C and eps at {n_tests} temperatures in {online_time:.5f} s')

# Print interpolated C and eps at a specific data point
idx = 123
np.set_printoptions(precision=4)
print(f'Interpolated C at {test_temperatures[idx]:.4f} K:')
print(approx_C[idx])
print(f'Interpolated eps at {test_temperatures[idx]:.4f} K:')
print(approx_eps[idx])
