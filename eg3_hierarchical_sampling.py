"""
Test hierarchical sampling on RVE with octahedral inclusion
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from operator import itemgetter
from microstructures import *
from optimize_alpha import opt4
from interpolate_fluctuation_modes import interpolate_fluctuation_modes
from utilities import read_h5, construct_stress_localization, volume_average, plot_and_save, cm, compute_err_indicator, \
    compute_residual, compute_residual_efficient, compute_err_indicator_efficient

np.random.seed(0)
file_name, data_path, temp1, temp2, n_tests, sampling_alphas = itemgetter('file_name', 'data_path', 'temp1', 'temp2', 'n_tests',
                                                                          'sampling_alphas')(microstructures[6])
print(file_name, '\t', data_path)

n_loading_directions = 1
n_hierarchical_levels = 5
test_temperatures = np.linspace(temp1, temp2, num=n_tests)
test_alphas = np.linspace(0, 1, num=n_tests)

mesh, ref = read_h5(file_name, data_path, test_temperatures)
mat_id = mesh['mat_id']
n_gauss = mesh['n_gauss']
strain_dof = mesh['strain_dof']
global_gradient = mesh['global_gradient']
n_gp = mesh['n_integration_points']
n_phases = len(np.unique(mat_id))
n_modes = ref[0]['strain_localization'].shape[-1]

strains = np.random.normal(size=(n_loading_directions, strain_dof))
strains /= la.norm(strains, axis=1)[:, None]

err_nodal_force = np.zeros((n_hierarchical_levels, n_tests, n_loading_directions))
err_indicators, err_eff_S, err_eff_C, err_eff_eps = [np.zeros((n_hierarchical_levels, n_tests)) for _ in range(4)]

alpha_levels = [np.linspace(0, 1, num=2)]

for level in range(n_hierarchical_levels):
    print(f'\n --- {level = :.2f} --- \n')
    alphas = alpha_levels[level]
    interpolate_temp = lambda x1, x2, alpha: x1 + alpha * (x2 - x1)
    temperatures = interpolate_temp(temp1, temp2, alphas)
    _, samples = read_h5(file_name, data_path, temperatures)
    for idx, alpha in enumerate(test_alphas):
        print(f'{alpha = :.2f}')
        temperature = test_temperatures[idx]

        upper_bound = np.searchsorted(alphas, alpha)
        id1 = upper_bound if upper_bound > 0 else 1
        id0 = id1 - 1

        E0 = samples[id0]['strain_localization']
        E1 = samples[id1]['strain_localization']
        E01 = np.ascontiguousarray(np.concatenate((E0, E1), axis=-1))

        sampling_C = np.stack((samples[id0]['mat_stiffness'], samples[id1]['mat_stiffness'])).transpose([1, 0, 2, 3])
        sampling_eps = np.stack((samples[id0]['mat_thermal_strain'], samples[id1]['mat_thermal_strain'])).transpose([1, 0, 2, 3])

        # reference values
        ref_C = ref[idx]['mat_stiffness']
        ref_eps = ref[idx]['mat_thermal_strain']
        normalization_factor_mech = ref[idx]['normalization_factor_mech']
        effSref = np.vstack((ref[idx]['eff_stiffness'], -ref[idx]['eff_stiffness'] @ ref[idx]['eff_thermal_strain'])).T

        # interpolated quantities using an implicit interpolation scheme with four DOF
        approx_C, approx_eps = opt4(sampling_C, sampling_eps, ref_C, ref_eps)
        Eopt4, _ = interpolate_fluctuation_modes(E01, approx_C, approx_eps, mat_id, n_gauss, strain_dof, n_modes, n_gp)
        Sopt4 = construct_stress_localization(Eopt4, ref_C, ref_eps, mat_id, n_gauss, strain_dof)
        effSopt = volume_average(Sopt4)

        err_indicators[level, idx] = np.mean(np.max(np.abs(compute_err_indicator_efficient(Sopt4, global_gradient)),
                                                    axis=0)) / normalization_factor_mech * 100

        for strain_idx, strain in enumerate(strains):
            zeta = np.hstack((strain, 1))
            stress_opt4 = np.einsum('ijk,k', Sopt4, zeta, optimize='optimal')
            residual = compute_residual_efficient(stress_opt4, global_gradient)

            err_nodal_force[level, idx, strain_idx] = la.norm(residual, np.inf) / normalization_factor_mech * 100

        err = lambda x, y: la.norm(x - y) * 100 / la.norm(y)
        err_eff_S[level, idx] = err(effSopt, effSref)

        Capprox = effSopt[:6, :6]
        Cref = effSref[:6, :6]
        invL = la.inv(la.cholesky(Cref))
        err_eff_C[level, idx] = la.norm(invL @ Capprox @ invL.T - np.eye(6)) / la.norm(np.eye(6)) * 100

        err_eff_eps[level, idx] = err(la.solve(Capprox, effSopt[:, -1]), la.solve(Cref, effSref[:, -1]))

    # max_err_idx = np.argmax(np.mean(err_nodal_force[level], axis=1))
    max_err_idx = np.argmax(err_indicators[level])
    alpha_levels.append(np.sort(np.hstack((alphas, test_alphas[max_err_idx]))))
    print(f'{np.max(np.mean(err_nodal_force[level], axis=1)) = }')
    print(f'{np.max(err_indicators[level]) = }')

np.savez_compressed('output/eg3', n_hierarchical_levels=n_hierarchical_levels, test_temperatures=test_temperatures,
                    err_nodal_force=err_nodal_force, err_indicators=err_indicators, err_eff_S=err_eff_S,
                    alpha_levels=np.asarray(alpha_levels, dtype=object))

# %%
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from utilities import plot_and_save, cm

# loaded_qoi = np.load('output/eg3.npz', allow_pickle=True)
# n_hierarchical_levels = loaded_qoi['n_hierarchical_levels']
# test_temperatures = loaded_qoi['test_temperatures']
# err_nodal_force = loaded_qoi['err_nodal_force']
# err_indicators = loaded_qoi['err_indicators']
# err_eff_S = loaded_qoi['err_eff_S']
# alpha_levels = loaded_qoi['alpha_levels']

temp1 = test_temperatures[0]
temp2 = test_temperatures[-1]
interpolate_temp = lambda x1, x2, alpha: x1 + alpha * (x2 - x1)

for level in range(n_hierarchical_levels):
    print(f'alphas of level {level}: {alpha_levels[level]}')
print('\n')
for level in range(n_hierarchical_levels):
    print(f'temperatures of level {level}: {interpolate_temp(temp1, temp2, alpha_levels[level])}')
print('\n')
for level in range(n_hierarchical_levels):
    print(f'level {level}')
    print(f'{np.max(np.mean(err_nodal_force[level], axis=1)) = }')
    print(f'{np.max(err_indicators[level]) = }')

xlabel = 'Temperature [K]'
markers = ['s', 'd', '+', 'x', 'o']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']

fig_name = 'eg3_hierarchical_sampling_err_nodal_force'
ylabel = 'Relative error $e_\mathsf{f}$ [\%]'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, np.mean(err_nodal_force[level], axis=1), label=f'{level + 2} samples', marker=markers[level],
             color=colors[level], markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(np.mean(err_nodal_force, axis=-1))], loc='upper left')

fig_name = 'eg3_hierarchical_sampling_err_indicator'
ylabel = 'Relative error $e_\mathsf{I}$ [\%]'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, err_indicators[level], label=f'{level + 2} samples', marker=markers[level], color=colors[level],
             markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_indicators)], loc='upper left')

fig_name = 'eg3_hierarchical_sampling_err_nodal_vs_indicator'
ylabel = 'Normalized error [-]'
err_indicators /= np.max(err_indicators)
err_nodal_force_mat = np.mean(err_nodal_force, axis=-1)
err_nodal_force_mat /= np.max(err_nodal_force_mat)
plt.figure(figsize=(10 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, err_nodal_force_mat[level], label=rf'$e_\mathsf f$ {level + 2} samples', marker=markers[level],
             color=colors[level], linestyle='-', markevery=8)
    plt.plot(test_temperatures, err_indicators[level], label=rf'$e_\mathsf I$ {level + 2} samples', marker=markers[level],
             color=colors[level], linestyle=':', markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_indicators)], loc='upper left')

fig_name = 'eg3_hierarchical_sampling_err_eff_stress_localization'
ylabel = r'Relative error $e_{\overline{\mathsf{S}}}$ [\%]'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, err_eff_S[level], label=f'{level + 2} samples', marker=markers[level], color=colors[level],
             markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_eff_S)], loc='upper left')

ylabel = r'Relative error $e_{\overline{\mathbb{C}}}$ [\%]'
fig_name = 'eg3_hierarchical_sampling_err_eff_stiffness'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, err_eff_C[level], label=f'{level + 2} samples', marker=markers[level], color=colors[level],
             markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_eff_C)], loc='upper left')

ylabel = r'Relative error $e_{\overline{\boldmath{\varepsilon}}_{\uptheta}}$ [\%]'
fig_name = 'eg3_hierarchical_sampling_err_eff_thermal_strain'
plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
for level in range(n_hierarchical_levels):
    plt.plot(test_temperatures, err_eff_eps[level], label=f'{level + 2} samples', marker=markers[level], color=colors[level],
             markevery=8)
plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_eff_eps)], loc='upper left')
