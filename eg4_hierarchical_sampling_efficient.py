import h5py
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from operator import itemgetter
from matplotlib.ticker import FormatStrFormatter

from interpolate_fluctuation_modes import update_affine_decomposition, effective_S, effective_stress_localization, \
    interpolate_fluctuation_modes, get_phi, transform_strain_localization
from microstructures import *
from optimize_alpha import opt4_alphas, opt4
from utilities import read_h5, construct_stress_localization, volume_average, plot_and_save, cm, compute_err_indicator_efficient

np.random.seed(0)
# np.set_printoptions(precision=3)

for ms_id in [7, 8, 9]:
    file_name, data_path, temp1, temp2, n_tests, sampling_alphas = itemgetter('file_name', 'data_path', 'temp1', 'temp2',
                                                                              'n_tests',
                                                                              'sampling_alphas')(microstructures[ms_id])
    print(file_name, '\t', data_path)
    out_file = path(f'output/opt_{file_name.name}')
    given_alpha_levels = True if sampling_alphas is not None else False

    # debuging options
    # given_alpha_levels = False
    # n_tests = 10

    n_hierarchical_levels = len(sampling_alphas) if sampling_alphas is not None else 5
    test_temperatures = np.linspace(temp1, temp2, num=n_tests)
    test_alphas = np.linspace(0, 1, num=n_tests)

    # read reference solutions
    mesh, refs = read_h5(file_name, data_path, test_temperatures)
    mat_id = mesh['mat_id']
    n_gauss = mesh['n_gauss']
    strain_dof = mesh['strain_dof']
    global_gradient = mesh['global_gradient']
    n_gp = mesh['n_integration_points']
    n_phases = len(np.unique(mat_id))
    n_modes = refs[0]['strain_localization'].shape[-1]

    # extract temperature dependent data from the reference solutions
    # such as: material stiffness and thermal strain at each temperature and for all phases
    ref_Cs = np.zeros((n_tests, *refs[0]['mat_stiffness'].shape))  # n_tests x n_phases x 6 x 6
    ref_epss = np.zeros((n_tests, *refs[0]['mat_thermal_strain'].shape))  # n_tests x n_phases x 6 x 1
    effSref = np.zeros((n_tests, strain_dof, n_modes))
    normalization_factor_mech = np.zeros((n_tests))
    for idx, alpha in enumerate(test_alphas):
        ref_Cs[idx] = refs[idx]['mat_stiffness']
        ref_epss[idx] = refs[idx]['mat_thermal_strain']
        normalization_factor_mech[idx] = refs[idx]['normalization_factor_mech']
        effSref[idx] = np.hstack(
            (refs[idx]['eff_stiffness'], -np.reshape(refs[idx]['eff_stiffness'] @ refs[idx]['eff_thermal_strain'], (-1, 1))))

    err_indicators, err_eff_S, err_eff_C, err_eff_eps = [np.zeros((n_hierarchical_levels, n_tests)) for _ in range(4)]
    interpolate_temp = lambda x1, x2, alpha: x1 + alpha * (x2 - x1)
    err = lambda x, y: la.norm(x - y) / la.norm(y) * 100

    # alpha_all_levels is initialized with the first level of two samples
    alpha_all_levels = [np.linspace(0, 1, num=2)] if not given_alpha_levels else sampling_alphas

    file = h5py.File(out_file, 'w')

    for level in range(n_hierarchical_levels):
        print(f'\n --- {level = :.2f} --- \n')

        # read sampling data given current sampling points. note that samples are reread in the next hierarchical level
        # but as long as everything is stored is h5 & no solvers are called there's no need for optimizing performance here
        alphas = alpha_all_levels[level]
        temperatures = interpolate_temp(temp1, temp2, alphas)
        n_samples = len(alphas)
        _, samples = read_h5(file_name, data_path, temperatures, get_mesh=False)
        # lists that contain quantities from sampling pairs
        E01s, sampling_Cs, sampling_epss = [], [], []
        for id0 in range(n_samples - 1):
            id1 = id0 + 1
            E0 = samples[id0]['strain_localization']
            E1 = samples[id1]['strain_localization']
            E01s.append(np.ascontiguousarray(np.concatenate((E0, E1), axis=-1)))
            # n_samples of [n_phases x 2 x 6 x 6]
            sampling_Cs.append(np.stack((samples[id0]['mat_stiffness'], samples[id1]['mat_stiffness'])).transpose([1, 0, 2, 3]))
            # n_samples of [n_phases x 2 x 6 x 1]
            sampling_epss.append(
                np.stack((samples[id0]['mat_thermal_strain'], samples[id1]['mat_thermal_strain'])).transpose([1, 0, 2, 3]))

        # alphas_indexing will contain the id of each pair of samples needed to solve the problem at a specific temperature
        # temperatures are determined by the values contained in tes_alphas
        alphas_indexing = np.searchsorted(alphas, test_alphas) - 1
        alphas_indexing[0] = 0

        current_sampling_id = None
        K0, K1, F0, F1, F2, F3, S001, S101, S103, S002, S102, S104 = [None for _ in range(12)]

        for idx, alpha in enumerate(test_alphas):
            print(f'{alpha = :.2f}')

            sampling_C = sampling_Cs[alphas_indexing[idx]]
            sampling_eps = sampling_epss[alphas_indexing[idx]]

            # interpolated quantities using an implicit interpolation scheme with four DOF
            alpha_C, alpha_eps = opt4_alphas(sampling_C, sampling_eps, ref_Cs[idx], ref_epss[idx])
            alpha_C_eps = alpha_C * alpha_eps

            # Assemble the linear system only when new samples are considered
            if alphas_indexing[idx] != current_sampling_id:
                current_sampling_id = alphas_indexing[idx]

                K0, K1, F0, F1, F2, F3, S001, S101, S103, S002, S102, S104 = update_affine_decomposition(
                    E01s[current_sampling_id], sampling_C, sampling_eps, n_modes, n_phases, n_gp, strain_dof, mat_id, n_gauss)

            phi = get_phi(K0, K1, F0, F1, F2, F3, alpha_C, alpha_eps, alpha_C_eps)

            speed = 1
            if speed == 0:
                C, eps = ref_Cs[idx], ref_epss[idx]
                # C, eps = opt4(sampling_C, sampling_eps, ref_Cs[idx], ref_epss[idx])
                _, effSopt = interpolate_fluctuation_modes(E01s[current_sampling_id], C, eps, mat_id, n_gauss, strain_dof,
                                                           n_modes, n_gp)
            elif speed == 1:
                effSopt = effective_stress_localization(E01s[current_sampling_id], phi, ref_Cs[idx], ref_epss[idx], mat_id,
                                                        n_gauss, n_gp, strain_dof, n_modes)
            elif speed == 2:
                # matches the result from interpolate_fluctuation_modes with a difference
                # that depends on using ref_Cs[idx],ref_epss[idx] instead of alphas
                effSopt, phi = effective_S(phi, S001, S101, S103, S002, S102, S104, alpha_C, np.squeeze(alpha_eps, axis=-1),
                                           np.squeeze(alpha_C_eps, axis=-1))
            else:
                raise NotImplementedError()

            if not given_alpha_levels:
                Eopt4 = transform_strain_localization(E01s[current_sampling_id], phi, n_gp, strain_dof, n_modes)
                Sopt4 = construct_stress_localization(Eopt4, ref_Cs[idx], ref_epss[idx], mat_id, n_gauss, strain_dof)
                err_indicators[level,
                               idx] = np.mean(np.max(np.abs(compute_err_indicator_efficient(Sopt4, global_gradient)),
                                                     axis=0)) / normalization_factor_mech[idx] * 100

            err_eff_S[level, idx] = err(effSopt, effSref[idx])

            Capprox = effSopt[:6, :6]
            Cref = effSref[idx][:6, :6]
            invL = la.inv(la.cholesky(Cref))

            err_eff_C[level, idx] = la.norm(invL @ Capprox @ invL.T - np.eye(6)) / la.norm(np.eye(6)) * 100
            err_eff_eps[level, idx] = err(la.solve(Capprox, effSopt[:, -1]), la.solve(Cref, effSref[idx][:, -1]))

            # TODO remove dtype='f'
            group = file.require_group(f'{data_path}_level{level}')
            # group.attrs['sampling_strategy'] = "model description"
            temperature = test_temperatures[idx]
            dset_stiffness = group.require_dataset(f'eff_stiffness_{temperature:07.2f}', (6, 6), dtype='f')
            dset_thermal_strain = group.require_dataset(f'eff_thermal_strain_{temperature:07.2f}', (6, ), dtype='f')
            dset_stiffness[:] = Capprox.T
            dset_thermal_strain[:] = la.solve(Capprox, effSopt[:, -1])

        if not given_alpha_levels:
            max_err_idx = np.argmax(err_indicators[level])
            alpha_all_levels.append(np.unique(np.sort(np.hstack((alphas, test_alphas[max_err_idx])))))

    file.close()
    idx = [idx for idx, microstructure in enumerate(microstructures) if file_name == microstructure['file_name']][0]
    np.savez_compressed(f'output/eg4_{idx}', n_hierarchical_levels=n_hierarchical_levels, test_temperatures=test_temperatures,
                        err_indicators=err_indicators, err_eff_S=err_eff_S, err_eff_C=err_eff_C, err_eff_eps=err_eff_eps,
                        alpha_all_levels=np.asarray(alpha_all_levels, dtype=object))

    # %%
    import numpy as np
    import numpy.linalg as la
    import matplotlib.pyplot as plt
    from utilities import plot_and_save, cm
    from matplotlib.ticker import FormatStrFormatter

    # loaded_qoi = np.load(f'output/eg4_{idx}.npz', allow_pickle=True)
    # n_hierarchical_levels = loaded_qoi['n_hierarchical_levels']
    # test_temperatures = loaded_qoi['test_temperatures']
    # err_indicators = loaded_qoi['err_indicators']
    # err_eff_S = loaded_qoi['err_eff_S']
    # err_eff_C = loaded_qoi['err_eff_C']
    # err_eff_eps = loaded_qoi['err_eff_eps']
    # alpha_all_levels = loaded_qoi['alpha_all_levels']

    temp1 = test_temperatures[0]
    temp2 = test_temperatures[-1]
    interpolate_temp = lambda x1, x2, alpha: x1 + alpha * (x2 - x1)

    for level in range(n_hierarchical_levels):
        print(f'alphas of level {level}: {alpha_all_levels[level]}')
    print('\n')
    for level in range(n_hierarchical_levels):
        print(f'temperatures of level {level}: {interpolate_temp(temp1, temp2, alpha_all_levels[level])}')
    print('\n')
    for level in range(n_hierarchical_levels):
        print(f'level {level}')
        print(f'{np.max(err_indicators[level]) = }')

    xlabel = 'Temperature [K]'
    styles = ['-', '-', '--', '-.', ':', ':', ':', ':']
    markers = ['s', 'd', '+', 'x', 'o']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    if not given_alpha_levels:
        ylabel = 'Relative error $e_\mathsf{I}$ [\%]'
        fig_name = f'eg4_{idx}_hierarchical_sampling_err_indicator'
        plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
        for level in range(n_hierarchical_levels):
            plt.plot(test_temperatures, err_indicators[level], label=f'{level + 2} samples', marker=markers[level],
                     color=colors[level], linestyle=styles[level], markevery=8)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_indicators)], loc='upper left')

    ylabel = 'Relative error $e_{\overline{\mathsf{S}}}$ [\%]'
    fig_name = f'eg4_{idx}_hierarchical_sampling_err_eff_stress_localization'
    plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
    for level in range(n_hierarchical_levels):
        plt.plot(test_temperatures, err_eff_S[level], label=f'{level + 2} samples', marker=markers[level], color=colors[level],
                 linestyle=styles[level], markevery=8)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_eff_S)], loc='upper left')

    ylabel = 'Relative error $e_{\overline{\mathbb{C}}}$ [\%]'
    fig_name = f'eg4_{idx}_hierarchical_sampling_err_eff_stiffness'
    plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
    for level in range(n_hierarchical_levels):
        plt.plot(test_temperatures, err_eff_C[level], label=f'{level + 2} samples', marker=markers[level], color=colors[level],
                 linestyle=styles[level], markevery=8)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_eff_C)], loc='upper left')

    ylabel = r'Relative error $e_{\overline{\boldmath{\varepsilon}}_{\uptheta}}$ [\%]'
    fig_name = f'eg4_{idx}_hierarchical_sampling_err_eff_thermal_strain'
    plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
    for level in range(n_hierarchical_levels):
        plt.plot(test_temperatures, err_eff_eps[level], label=f'{level + 2} samples', marker=markers[level], color=colors[level],
                 linestyle=styles[level], markevery=8)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plot_and_save(xlabel, ylabel, fig_name, [temp1, temp2], [0, np.max(err_eff_eps)], loc='upper left')

    print(np.max(err_indicators))
    print(np.max(err_eff_S))
    print(np.max(err_eff_C))
    print(np.max(err_eff_eps))
