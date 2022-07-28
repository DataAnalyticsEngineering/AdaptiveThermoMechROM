"""
Test a straightforward implementation of all approaches
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from operator import itemgetter
from microstructures import *
from optimize_alpha import opt1, opt2, opt4, naive
from interpolate_fluctuation_modes import interpolate_fluctuation_modes
from utilities import read_h5, construct_stress_localization, volume_average, plot_and_save, cm, compute_residual, \
    compute_residual_efficient

np.random.seed(0)
file_name, data_path, temp1, temp2, n_tests, sampling_alphas = itemgetter('file_name', 'data_path', 'temp1', 'temp2', 'n_tests',
                                                                          'sampling_alphas')(microstructures[0])
print(file_name, '\t', data_path)

n_loading_directions = 10
n_tests = 10
test_temperatures = np.linspace(temp1, temp2, num=n_tests)
test_alphas = np.linspace(0, 1, num=n_tests)

mesh, ref = read_h5(file_name, data_path, test_temperatures)
mat_id = mesh['mat_id']
n_gauss = mesh['n_gauss']
strain_dof = mesh['strain_dof']
global_gradient = mesh['global_gradient']
n_gp = mesh['n_integration_points']
n_modes = ref[0]['strain_localization'].shape[-1]

_, samples = read_h5(file_name, data_path, [temp1, temp2], get_mesh=False)

strains = np.random.normal(size=(n_loading_directions, mesh['strain_dof']))
strains /= la.norm(strains, axis=1)[:, None]

n_approaches = 5
err_E, err_S, err_eff_S = [np.zeros((n_approaches, n_tests)) for _ in range(3)]
err_eff_stress, err_f = [np.zeros((n_approaches, n_tests * n_loading_directions)) for _ in range(2)]

for idx, alpha in enumerate(test_alphas):
    print(f'{alpha = :.2f}')
    temperature = test_temperatures[idx]

    interpolate_temp = lambda x1, x2: x1 + alpha * (x2 - x1)

    E0 = samples[0]['strain_localization']
    E1 = samples[1]['strain_localization']
    E01 = np.ascontiguousarray(np.concatenate((E0, E1), axis=-1))

    sampling_C = np.stack((samples[0]['mat_stiffness'], samples[1]['mat_stiffness'])).transpose([1, 0, 2, 3])
    sampling_eps = np.stack((samples[0]['mat_thermal_strain'], samples[1]['mat_thermal_strain'])).transpose([1, 0, 2, 3])

    # reference values
    Eref = ref[idx]['strain_localization']
    ref_C = ref[idx]['mat_stiffness']
    ref_eps = ref[idx]['mat_thermal_strain']
    normalization_factor_mech = ref[idx]['normalization_factor_mech']

    Sref = construct_stress_localization(Eref, ref_C, ref_eps, mat_id, n_gauss, strain_dof)
    effSref = volume_average(Sref)

    # interpolated quantities using an explicit interpolation scheme with one DOF
    approx_C, approx_eps = naive(alpha, sampling_C, sampling_eps, ref_C, ref_eps)
    Enaive = interpolate_temp(E0, E1)
    Snaive = construct_stress_localization(Enaive, ref_C, ref_eps, mat_id, n_gauss, strain_dof)
    effSnaive = volume_average(Snaive)

    # interpolated quantities using an explicit interpolation scheme with one DOF
    Eopt0, _ = interpolate_fluctuation_modes(E01, approx_C, approx_eps, mat_id, n_gauss, strain_dof, n_modes, n_gp)
    Sopt0 = construct_stress_localization(Eopt0, ref_C, ref_eps, mat_id, n_gauss, strain_dof)
    effSopt0 = volume_average(Sopt0)

    # interpolated quantities using an implicit interpolation scheme with one DOF
    approx_C, approx_eps = opt1(sampling_C, sampling_eps, ref_C, ref_eps)
    Eopt1, _ = interpolate_fluctuation_modes(E01, approx_C, approx_eps, mat_id, n_gauss, strain_dof, n_modes, n_gp)
    Sopt1 = construct_stress_localization(Eopt1, ref_C, ref_eps, mat_id, n_gauss, strain_dof)
    effSopt1 = volume_average(Sopt1)

    # interpolated quantities using an implicit interpolation scheme with two DOF
    approx_C, approx_eps = opt2(sampling_C, sampling_eps, ref_C, ref_eps)
    Eopt2, _ = interpolate_fluctuation_modes(E01, approx_C, approx_eps, mat_id, n_gauss, strain_dof, n_modes, n_gp)
    Sopt2 = construct_stress_localization(Eopt2, ref_C, ref_eps, mat_id, n_gauss, strain_dof)
    effSopt2 = volume_average(Sopt2)

    # interpolated quantities using an implicit interpolation scheme with four DOF
    approx_C, approx_eps = opt4(sampling_C, sampling_eps, ref_C, ref_eps)
    Eopt4, _ = interpolate_fluctuation_modes(E01, approx_C, approx_eps, mat_id, n_gauss, strain_dof, n_modes, n_gp)
    Sopt4 = construct_stress_localization(Eopt4, ref_C, ref_eps, mat_id, n_gauss, strain_dof)
    effSopt4 = volume_average(Sopt4)

    err = lambda x, y: np.mean(la.norm(x - y, axis=(-1, -2)) / la.norm(y, axis=(-1, -2))) * 100
    err_vec = lambda x, y: np.mean(la.norm(x - y, axis=(-1)) / la.norm(y, axis=(-1))) * 100

    err_E[:, idx] = [err(Enaive, Eref), err(Eopt0, Eref), err(Eopt1, Eref), err(Eopt2, Eref), err(Eopt4, Eref)]
    err_S[:, idx] = [err(Snaive, Sref), err(Sopt0, Sref), err(Sopt1, Sref), err(Sopt2, Sref), err(Sopt4, Sref)]
    err_eff_S[:, idx] = [err(effSnaive, effSref), err(effSopt0, effSref), err(effSopt1, effSref), \
                         err(effSopt2, effSref), err(effSopt4, effSref)]

    for strain_idx, strain in enumerate(strains):
        zeta = np.hstack((strain, 1))

        eff_stress_ref = effSref @ zeta
        err_eff_stress[:, idx * n_loading_directions + strain_idx] = \
            [err_vec(effSnaive @ zeta, eff_stress_ref), err_vec(effSopt0 @ zeta, eff_stress_ref), \
             err_vec(effSopt1 @ zeta, eff_stress_ref), err_vec(effSopt2 @ zeta, eff_stress_ref), \
             err_vec(effSopt4 @ zeta, eff_stress_ref)]

        stress_naive = np.einsum('ijk,k', Snaive, zeta, optimize='optimal')
        stress_opt0 = np.einsum('ijk,k', Sopt0, zeta, optimize='optimal')
        stress_opt1 = np.einsum('ijk,k', Sopt1, zeta, optimize='optimal')
        stress_opt2 = np.einsum('ijk,k', Sopt2, zeta, optimize='optimal')
        stress_opt4 = np.einsum('ijk,k', Sopt4, zeta, optimize='optimal')

        residuals = compute_residual_efficient([stress_naive, stress_opt0, stress_opt1, stress_opt2, stress_opt4],
                                               mesh['global_gradient'])

        err_f[:, idx * n_loading_directions + strain_idx] = la.norm(residuals, np.inf, axis=0) / normalization_factor_mech * 100

np.savez_compressed('output/eg2', err_f=err_f, err_eff_S=err_eff_S, err_eff_stress=err_eff_stress, err_E=err_E, err_S=err_S,
                    n_loading_directions=n_loading_directions, n_approaches=n_approaches)
# %%
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from utilities import plot_and_save, cm, ecdf

# loaded_qoi = np.load('output/eg2.npz')
# err_f = loaded_qoi['err_f']
# err_eff_S = loaded_qoi['err_eff_S']
# err_eff_stress = loaded_qoi['err_eff_stress']
# err_E = loaded_qoi['err_E']
# err_S = loaded_qoi['err_S']
# n_loading_directions = loaded_qoi['n_loading_directions']
# n_approaches = loaded_qoi['n_approaches']
markevery = 8 * n_loading_directions

xlabel = '$x$ [\%]'
labels = ['N', r'O$_0$', 'O$_1$', 'O$_2$', 'O$_4$']
markers = ['s', 'd', '+', 'x', 'o']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']

fig_name = 'eg2_err_nodal_force'
ylabel = r'$P(e_\mathsf{f}<x)$ [-]'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
axins = ax.inset_axes([0.45, 0.15, 0.2, 0.3])
for idx in range(n_approaches):
    x, y = ecdf(err_f[idx])
    print(f'err_f {np.max(x[y<=0.99]) = :2.2e}')
    plt.step(x, y, label=labels[idx], marker=markers[idx], color=colors[idx], markevery=markevery)
    axins.step(x, y, label=labels[idx], marker=markers[idx], color=colors[idx], markevery=markevery)
axins.set_xlim(-0.05, 0.5)
axins.set_ylim(0.6, 1)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plot_and_save(xlabel, ylabel, fig_name, [-0.1, np.max(err_f) * 1.1], [0, 1], loc='lower right')

fig_name = 'eg2_err_strain_localization'
ylabel = r'$P(e_\mathsf{E}<x)$ [-]'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
axins = ax.inset_axes([0.1, 0.65, 0.2, 0.3])
for idx in range(5):
    plt.step(*ecdf(err_E[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
    axins.step(*ecdf(err_E[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
axins.set_xlim(-0.05, 0.5)
axins.set_ylim(0.6, 1)
axins.get_yaxis().set_visible(False)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plot_and_save(xlabel, ylabel, fig_name, [-0.1, np.max(err_E) * 1.1], [0, 1], loc='lower right')

fig_name = 'eg2_err_stress_localization'
ylabel = r'$P(e_\mathsf{S}<x)$ [-]'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
axins = ax.inset_axes([0.1, 0.65, 0.2, 0.3])
for idx in range(5):
    plt.step(*ecdf(err_S[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
    axins.step(*ecdf(err_S[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
axins.set_xlim(-0.05, 0.2)
axins.set_ylim(0.6, 1)
axins.get_yaxis().set_visible(False)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plot_and_save(xlabel, ylabel, fig_name, [-0.1, np.max(err_S) * 1.1], [0, 1], loc='lower right')

fig_name = 'eg2_err_eff_stress_localization'
ylabel = r'$P(e_{\overline{\mathsf{S}}}<x)$ [-]'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
axins = ax.inset_axes([0.1, 0.65, 0.2, 0.3])
for idx in range(5):
    plt.step(*ecdf(err_eff_S[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
    axins.step(*ecdf(err_eff_S[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=8)
axins.set_xlim(-0.05, 0.2)
axins.set_ylim(0.6, 1)
axins.get_yaxis().set_visible(False)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plot_and_save(xlabel, ylabel, fig_name, [-0.1, np.max(err_eff_S) * 1.1], [0, 1], loc='lower right')

fig_name = 'eg2_err_eff_stress'
ylabel = r'$P(e_{\overline{\mathsf{\sigma}}}<x)$ [-]'
fig, ax = plt.subplots(figsize=(6 * cm, 6 * cm), dpi=600)
axins = ax.inset_axes([0.45, 0.15, 0.2, 0.3])
for idx in range(5):
    plt.step(*ecdf(err_eff_stress[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=markevery)
    axins.step(*ecdf(err_eff_stress[idx]), label=labels[idx], marker=markers[idx], color=colors[idx], markevery=markevery)
axins.set_xlim(-0.05, 0.2)
axins.set_ylim(0.6, 1)
ax.indicate_inset_zoom(axins, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor=(0.3, 0.3, 0.3, 0.3), lw=0.1)
plot_and_save(xlabel, ylabel, fig_name, [-0.1, np.max(err_eff_stress) * 1.1], [0, 1], loc='lower right')

with np.printoptions(precision=4, suppress=True, formatter={'float': '{:>2.2e}'.format}, linewidth=100):
    print(f'{np.max(err_f,axis=1) = }')
    print(f'{np.max(err_E,axis=1) = }')
    print(f'{np.max(err_S,axis=1) = }')
    print(f'{np.max(err_eff_S,axis=1) = }')
    print(f'{np.max(err_eff_stress,axis=1) = }')
