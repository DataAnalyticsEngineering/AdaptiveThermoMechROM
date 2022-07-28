# %%
import h5py
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from utilities import plot_and_save, cm, read_h5
from matplotlib.ticker import FormatStrFormatter
from operator import itemgetter
from microstructures import *
from utilities import read_h5, construct_stress_localization, volume_average, plot_and_save, cm, compute_err_indicator_efficient

n_hierarchical_levels = 5
ms_id = 9
file_name, data_path, temp1, temp2, n_tests, sampling_alphas = itemgetter('file_name', 'data_path', 'temp1', 'temp2', 'n_tests',
                                                                          'sampling_alphas')(microstructures[ms_id])
print(file_name, '\t', data_path)
opt_file = path(f'output/opt_{file_name.name}')
ann_file = path(f'output/ann_{file_name.name}')

test_temperatures = np.linspace(temp1, temp2, num=n_tests)

refC = np.zeros((n_tests, 6, 6))
ref_eps = np.zeros((n_tests, 6))

optC = np.zeros((n_tests, n_hierarchical_levels, 6, 6))
opt_eps = np.zeros((n_tests, n_hierarchical_levels, 6))

annC = np.zeros((n_tests, n_hierarchical_levels, 6, 6))
ann_eps = np.zeros((n_tests, n_hierarchical_levels, 6))

for level in range(n_hierarchical_levels):
    with h5py.File(file_name, 'r') as file:
        for idx, temperature in enumerate(test_temperatures):
            refC[idx] = file[f'{data_path}/eff_stiffness_{temperature:07.2f}'][:]
            ref_eps[idx] = file[f'{data_path}/eff_thermal_strain_{temperature:07.2f}'][:]

    with h5py.File(opt_file, 'r') as file:
        for idx, temperature in enumerate(test_temperatures):
            optC[idx, level] = file[f'{data_path}_level{level}/eff_stiffness_{temperature:07.2f}'][:]
            opt_eps[idx, level] = -file[f'{data_path}_level{level}/eff_thermal_strain_{temperature:07.2f}'][:]

    with h5py.File(ann_file, 'r') as file:
        for idx, temperature in enumerate(test_temperatures):
            annC[idx, level] = file[f'{data_path}_level{level}/eff_stiffness_{temperature:07.2f}'][:]
            ann_eps[idx, level] = file[f'{data_path}_level{level}/eff_thermal_strain_{temperature:07.2f}'][:]

#%%
err_eff_C_opt = np.zeros((n_tests, n_hierarchical_levels))
err_eff_eps_opt = np.zeros((n_tests, n_hierarchical_levels))

err_eff_C_ann = np.zeros((n_tests, n_hierarchical_levels))
err_eff_eps_ann = np.zeros((n_tests, n_hierarchical_levels))

err = lambda x, y: la.norm(x - y) * 100 / la.norm(y)
err_energy = lambda x, y, C: (y @ C @ y - x @ C @ x) / (y @ C @ y) * 100

for level in range(n_hierarchical_levels):
    for idx, temperature in enumerate(test_temperatures):

        invL = la.inv(la.cholesky(refC[idx]))

        err_eff_C_opt[idx, level] = la.norm(invL @ optC[idx, level] @ invL.T - np.eye(6)) / la.norm(np.eye(6)) * 100
        err_eff_C_ann[idx, level] = la.norm(invL @ annC[idx, level] @ invL.T - np.eye(6)) / la.norm(np.eye(6)) * 100

        err_eff_eps_opt[idx, level] = err(opt_eps[idx, level], ref_eps[idx])
        err_eff_eps_ann[idx, level] = err(ann_eps[idx, level], ref_eps[idx])

        # err_eff_eps_opt[idx, level] = err_energy(opt_eps[idx, level], ref_eps[idx], refC[idx])
        # err_eff_eps_ann[idx, level] = err_energy(ann_eps[idx, level], ref_eps[idx], refC[idx])

#%%
level = 4

xlabel = 'Temperature [K]'
styles = ['-', '--', ':', '-.', ':', ':', ':']
markers = ['s', 'd', '+', 'x', 'o']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']

ylabel = 'Relative error $e_{\overline{\mathbb{C}}}$ [\%]'
fig_name = f'eg5_{ms_id}_hierarchical_sampling_err_eff_stiffness'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
# axins = ax.inset_axes([0.53, 0.42, 0.4, 0.3])
axins = ax.inset_axes([0.23, 0.42, 0.4, 0.3])
# for level in range(n_hierarchical_levels):
plt.plot(test_temperatures, err_eff_C_opt[:, level], label=f'O$_4$ {level + 2} samples', marker=markers[level], color=colors[1],
         linestyle='--', markevery=8)
axins.plot(test_temperatures, err_eff_C_opt[:, level], label=f'O$_4$ {level + 2} samples', marker=markers[level], color=colors[1],
           linestyle='--', markevery=8)
plt.plot(test_temperatures, err_eff_C_ann[:, level], label=f'ANN {level + 2} samples', marker=markers[level], color=colors[0],
         linestyle='-', markevery=8)
axins.grid('on')
axins.grid(ls='--', color='gray', linewidth=0.5)
axins.set_xlim(temp1, temp2)
axins.set_ylim(0.0, 0.020)
axins.get_xaxis().set_visible(False)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, 3.5], loc='upper right')
# plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, 1.5], loc='upper right')
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, 2.2], loc='upper right')

#%%
ylabel = r'Relative error $e_{\overline{\boldmath{\varepsilon}}_{\uptheta}}$ [\%]'
fig_name = f'eg5_{ms_id}_hierarchical_sampling_err_eff_thermal_strain'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
axins = ax.inset_axes([0.53, 0.42, 0.4, 0.3])
# for level in range(n_hierarchical_levels):
plt.plot(test_temperatures, err_eff_eps_opt[:, level], label=f'O$_4$ {level + 2} samples', marker=markers[level], color=colors[1],
         linestyle='--', markevery=8)
axins.plot(test_temperatures, err_eff_eps_opt[:, level], label=f'O$_4$ {level + 2} samples', marker=markers[level],
           color=colors[1], linestyle='--', markevery=8)
plt.plot(test_temperatures, err_eff_eps_ann[:, level], label=f'ANN {level + 2} samples', marker=markers[level], color=colors[0],
         linestyle='-', markevery=8)
axins.grid('on')
axins.grid(ls='--', color='gray', linewidth=0.5)
axins.set_xlim(temp1, temp2)
axins.set_ylim(0.0, 0.2)
axins.get_xaxis().set_visible(False)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, 15], loc='upper right')
# plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, 4], loc='upper right')
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, 7], loc='upper right')
