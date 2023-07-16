"""
NTFA
"""
import numpy as np
import h5py
from utilities import *
from material_parameters import *

def read_snapshots(file_name, data_path):
    """
    Read an H5 file that contains responses of simulated microstructures
    :param file_name: e.g. "input/simple_3d_rve_combo.h5"
    :param data_path: the path to the simulation results within the h5 file, e.g. '/ms_1p/dset0_sim'
    :return:
        strain_snapshots: plastic strain snapshots eps_p 
            with shape (n_integration_points, strain_dof, n_frames)
    """
    # TODO: read snapshots from H5 file. Because of the sheer amount of data it may be better to use a separate h5 file for the snapshots.
    plastic_snapshots = None
    with h5py.File(file_name, 'r') as file:
        plastic_snapshots = np.transpose(file[f'{data_path}/plastic_strains'][:], (0, 2, 1))
    # For now, use dummy data:
    # n_integration_points, strain_dof, n_frames = 512, 6, 100
    # plastic_snapshots = create_dummy_plastic_modes(n_integration_points, strain_dof, n_frames)
    # TODO: Reorder snapshots as follows: | 1st strain path: last timestep to first timestep | 2nd strain path: last timestep to first timestep | ...
    # or: reorder snapshots already in FANS?
    return plastic_snapshots


def mode_identification_iterative(plastic_snapshots, vol_frac, r_min=1e-8):
    """
    Identification of plastic strain modes µ using an iterative algorithm and renormalization
    :param strain_snapshots: plastic strain snapshots eps_p (ordered as described in `read_snapshots`)
        with shape (n_integration_points, strain_dof, n_frames)
    :param r_min: stop criterion
    :return:
        plastic_modes: plastic strain modes µ with shape (n_integration_points, strain_dof, n_modes)
    """
    n_integration_points, strain_dof, n_frames = plastic_snapshots.shape
    n_modes = 0
    plastic_modes = np.zeros((n_integration_points, strain_dof, n_modes))
    for i in range(n_frames):
        eps_i = plastic_snapshots[:, :, i]
        r = volume_average(inner_product(eps_i, eps_i))  # TODO: average only over Omega_p?
        k = np.zeros(n_modes)  # Coefficients k_j
        for j in range(n_modes):
            k[j] = volume_average(inner_product(eps_i, plastic_modes[:, :, j]))
            r = r - k[j]**2
            if r < r_min:
                break
        if r > r_min:
            n_modes = n_modes + 1  # increment number of modes
            # Generate new strain mode:
            plastic_mode = (eps_i - np.tensordot(k, plastic_modes, axes=(0,2))) / np.sqrt(r)
            plastic_modes = np.concatenate([plastic_modes, np.expand_dims(plastic_mode, 2)], axis=2)
    # Renormalize plastic modes
    for i in range(n_modes):
        weighting_factor = vol_frac / volume_average(norm_2(plastic_modes[:, :, i]))
        plastic_modes[:, :, i] = plastic_modes[:, :, i] * weighting_factor * np.sign(plastic_modes[0, 0, i])
    return plastic_modes


def mode_identification(plastic_snapshots, vol_frac, r_min=1e-8):
    """
    Identification of plastic strain modes µ using POD and renormalization
    :param strain_snapshots: plastic strain snapshots eps_p (ordered as described in `read_snapshots`)
        with shape (n_integration_points, strain_dof, n_frames)
    :param r_min: stop criterion
    :return:
        plastic_modes: plastic strain modes µ with shape (n_integration_points, strain_dof, n_modes)
    """
    n_integration_points, strain_dof, n_frames = plastic_snapshots.shape
    plastic_snapshots_rs = plastic_snapshots.transpose(1, 0, 2).reshape((strain_dof * n_integration_points, n_frames))
    u, s, v = np.linalg.svd(plastic_snapshots_rs, full_matrices=False)
    s = s / s[0]
    n_modes = np.argwhere(s > r_min).size
    plastic_modes = u[:, :n_modes].reshape((strain_dof, n_integration_points, n_modes)).transpose(1, 0, 2)
    # Renormalize plastic modes
    for i in range(n_modes):
        weighting_factor = vol_frac / volume_average(norm_2(plastic_modes[:, :, i]))
        plastic_modes[:, :, i] = plastic_modes[:, :, i] * weighting_factor * np.sign(plastic_modes[0, 0, i])
    return plastic_modes


def compute_ntfa_matrices(strain_localization, stress_localization, plastic_modes, mat_thermal_strain, mesh):
    """
    Processing of the plastic strain modes µ to compute the matrices A_bar, D_xi, C_bar and the vector tau_theta
    as tabular data at given temperatures
    :param strain_localization: strain localization 3D array
        with shape (n_integration_points, strain_dof, 7 + n_modes)
    :param stress_localization: stress localization 3D array
        with shape (n_integration_points, strain_dof, 7 + n_modes)
    :param mat_stiffness: stiffness tensors of the phases
        with shape (n_phases, strain_dof, strain_dof)
    :param mat_id: material phase identification
        with shape (n_elements,)
    :param mesh:
    :param plastic_modes: plastic strain modes µ
        with shape (n_integration_points, strain_dof, N_modes)
    :param eigen_strains: solutions of the auxiliary eigenstress problems eps_star
        with shape (n_integration_points, strain_dof, N_modes)
    :param ...:
    :return:
        C_bar with shape (strain_dof, strain_dof)
        tau_theta with shape (strain_dof,)
        A_bar with shape (strain_dof, strain_dof)
        tau_xi with shape (n_modes,)
        D_xi with shape (strain_dof, n_modes)
        D_theta with shape (1,)
    """
    strain_dof = mesh['strain_dof']
    mat_id = mesh['mat_id']
    n_modes = plastic_modes.shape[2]
    n_gp = mesh['n_integration_points']
    n_gauss = mesh['n_gauss']
    
    # slice strain localization operator E into E_eps, E_theta, E_xi
    E_eps = strain_localization[:, :, :strain_dof]
    E_theta = strain_localization[:, :, strain_dof]
    E_xi = strain_localization[:, :, strain_dof + 1:]

    
    # slice stress localization operator S into S_eps, S_theta, S_xi
    S_eps = stress_localization[:, :, :strain_dof]
    S_theta = stress_localization[:, :, strain_dof]
    S_xi = stress_localization[:, :, strain_dof + 1:]

    I = np.eye(6)
    # Compute C_bar via < (E_eps + I).T @ S_eps >
    C_bar = volume_average((E_eps + I).transpose((0, 2, 1)) @ S_eps)

    # Compute tau_theta via < (E_eps + I).T @ S_theta >
    tau_theta = volume_average(np.einsum('nij,nj->ni', (E_eps + I).transpose((0, 2, 1)), S_theta))

    # Compute A_bar via < (E_eps + I).T @ S_eps >
    A_bar = volume_average((E_eps + I).transpose((0, 2, 1)) @ S_xi)

    # Compute tau_xi via < (E_theta - P_theta).T @ S_xi >
    # Account for the phase-wise thermal strain by an explicit summation over all integration points
    tau_xi = np.zeros((1, n_modes))
    for gp_id in prange(n_gp):
        phase_id = mat_id[gp_id // n_gauss]
        tau_xi += (E_theta[gp_id] - mat_thermal_strain[phase_id].T) @ S_xi[gp_id] / n_gp

    # Compute D_xi via < (E_xi - P_xi).T @ S_xi >
    D_xi = volume_average((E_xi - plastic_modes).transpose((0, 2, 1)) @ S_xi)

    # Compute D_theta via < (E_theta - P_theta).T @ S_theta >
    # Account for the phase-wise thermal strain by an explicit summation over all integration points
    D_theta = 0.
    for gp_id in prange(n_gp):
        phase_id = mat_id[gp_id // n_gauss]
        D_theta += (E_theta[gp_id] - mat_thermal_strain[phase_id].T) @ S_theta[gp_id] / n_gp

    return C_bar, tau_theta, A_bar, tau_xi, D_xi, D_theta


def compute_phase_average_stresses(strain_localization, mat_stiffness, mat_thermal_strain, plastic_modes, zeta, mesh):
    combo_strain_loc0, combo_strain_loc1 = None, None
    stress_localization0, stress_localization1, combo_stress_loc0, combo_stress_loc1 = construct_stress_localization_phases(
        strain_localization, mat_stiffness, mat_thermal_strain, plastic_modes, combo_strain_loc0, combo_strain_loc1, mesh)

    combo_stress0 = np.einsum('ijk,k', combo_stress_loc0, zeta, optimize='optimal') if combo_stress_loc0 is not None else None
    combo_stress1 = np.einsum('ijk,k', combo_stress_loc1, zeta, optimize='optimal') if combo_stress_loc1 is not None else None

    stress0 = np.einsum('ijk,k', stress_localization0, zeta, optimize='optimal')
    stress1 = np.einsum('ijk,k', stress_localization1, zeta, optimize='optimal')
    average_stress_0, average_stress_1 = volume_average_phases(stress0, stress1, combo_stress0, combo_stress1, mesh)
    return average_stress_0, average_stress_1


def compute_phase_average_stress_localizations(strain_localization, mat_stiffness, mat_thermal_strain, plastic_modes, mesh):
    assert mesh['vox_type'] == 'normal', 'For now, only normal voxels are supported'
    combo_strain_loc0, combo_strain_loc1 = None, None
    stress_localization0, stress_localization1, _, _ = construct_stress_localization_phases(
        strain_localization, mat_stiffness, mat_thermal_strain, plastic_modes, combo_strain_loc0, combo_strain_loc1, mesh)
    average_stress_localization0, average_stress_localization1 = volume_average(stress_localization0), volume_average(stress_localization1)
    return average_stress_localization0, average_stress_localization1


def compute_tabular_data_for_ms(ms_id, temperatures):
    """
    Perform `compute_tabular_data` for microstructure with id `ms_id`
    :param ms_id:
    :param temperatures:
    """
    file_name, data_path, temp1, temp2, n_tests, sampling_alphas = itemgetter('file_name', 'data_path', 'temp1', 'temp2', 'n_tests',
                                                                          'sampling_alphas')(microstructures[ms_id])
    sample_temperatures = np.linspace(temp1, temp2, num=n_tests)
    sample_alphas = np.linspace(0, 1, num=n_tests)
    mesh, samples = read_h5(file_name, data_path, sample_temperatures)
    return compute_tabular_data(samples, mesh, temperatures)


#@jit(nopython=True, cache=True, parallel=True, nogil=True)
def compute_tabular_data(samples, mesh, temperatures):
    """
    Compute tabular data for a given list of temperatures
    :param samples:
    :param mesh:
    :param temperatures:
    :return:
        C_bar, tau_theta, A_bar, tau_xi, D_xi, D_theta, A0, A1, C0, C1
    """
    assert mesh['vox_type'] == 'normal', 'For now, only normal voxels are supported'
    mat_id = mesh['mat_id']
    n_gauss = mesh['n_gauss']
    strain_dof = mesh['strain_dof']
    global_gradient = mesh['global_gradient']
    n_gp = mesh['n_integration_points']
    n_phases = len(np.unique(mat_id))
    n_modes = samples[0]['plastic_modes'].shape[-1]
    n_temps = len(temperatures)
    C_bar = np.zeros((strain_dof, strain_dof, n_temps))
    tau_theta = np.zeros((strain_dof, n_temps))
    A_bar = np.zeros((strain_dof, n_modes, n_temps))
    tau_xi = np.zeros((n_modes, n_temps))
    D_xi = np.zeros((n_modes, n_modes, n_temps))
    D_theta = np.zeros((n_temps))
    C0 = np.zeros((strain_dof, strain_dof, n_temps))
    C1 = np.zeros((strain_dof, strain_dof, n_temps))
    A0 = np.zeros((strain_dof, 7 + n_modes, n_temps))
    A1 = np.zeros((strain_dof, 7 + n_modes, n_temps))
    vol_frac0 = mesh['volume_fraction'][0]
    vol_frac1 = mesh['volume_fraction'][1]
    plastic_modes = samples[0]['plastic_modes']
    # interpolate_temp = lambda x1, x2, alpha: x1 + alpha * (x2 - x1)
    # dns_temperatures = interpolate_temp(temp1, temp2, sample_alphas)
    sample_temperatures = np.array([sample['temperature'] for sample in samples])
    temp1, temp2 = min(sample_temperatures), max(sample_temperatures)
    sample_alphas = (sample_temperatures - temp1) / (temp2 - temp1)
    for idx in prange(n_temps):
        temperature = temperatures[idx]
        ref_C = np.stack(([stiffness_cu(temperature), stiffness_wsc(temperature)]))
        ref_eps = np.expand_dims(np.stack(([thermal_strain_cu(temperature), thermal_strain_wsc(temperature)])), axis=2)
        alpha = (temperature - temp1) / (temp2 - temp1)
        upper_bound = np.searchsorted(sample_alphas, alpha)
        if np.floor(alpha) == alpha:
            # sample for given temperature exists, no need for interpolation
            id = upper_bound
            C, eps = ref_C, ref_eps
            E = samples[id]['strain_localization']
            S = construct_stress_localization(E, ref_C, ref_eps, plastic_modes, mat_id, n_gauss, strain_dof)
        else:
            id1 = upper_bound if upper_bound > 0 else 1
            id0 = id1 - 1

            E0 = samples[id0]['strain_localization']
            E1 = samples[id1]['strain_localization']
            E01 = np.ascontiguousarray(np.concatenate((E0, E1), axis=-1))

            sampling_C = np.stack((samples[id0]['mat_stiffness'], samples[id1]['mat_stiffness'])).transpose([1, 0, 2, 3])
            sampling_eps = np.stack((samples[id0]['mat_thermal_strain'], samples[id1]['mat_thermal_strain'])).transpose([1, 0, 2, 3])

            # interpolated quantities using an implicit interpolation scheme with four DOF
            C, eps = opt4(sampling_C, sampling_eps, ref_C, ref_eps)
            E, _ = interpolate_fluctuation_modes(E01, C, eps, plastic_modes, mat_id, n_gauss, strain_dof, n_modes, n_gp)
            S = construct_stress_localization(E, ref_C, ref_eps, plastic_modes, mat_id, n_gauss, strain_dof)

        # Compute NTFA matrices
        C_bar[:, :, idx], tau_theta[:, idx], A_bar[:, :, idx], tau_xi[:, idx], D_xi[:, :, idx], D_theta[idx] = \
            compute_ntfa_matrices(E, S, plastic_modes, eps, mesh)
        
        # Compute phase average stresses
        A0_full, A1_full = compute_phase_average_stress_localizations(E, ref_C, ref_eps, plastic_modes, mesh)
        A0[:, :, idx], A1[:, :, idx] = A0_full, A1_full

        # Save phase-wise stiffness tensors
        C0[:, :, idx], C1[:, :, idx] = ref_C[0, :, :], ref_C[1, :, :]
    return C_bar, tau_theta, A_bar, tau_xi, D_xi, D_theta, A0, A1, C0, C1


@jit(nopython=True, cache=True, parallel=True, nogil=True)
def compute_tabular_data_efficient(samples, mesh, temperatures):
    """
    WIP
    mat_id = mesh['mat_id']
    n_gauss = mesh['n_gauss']
    strain_dof = mesh['strain_dof']
    global_gradient = mesh['global_gradient']
    n_gp = mesh['n_integration_points']
    n_phases = len(np.unique(mat_id))
    n_modes = samples[0]['strain_localization'].shape[-1]
    n_temps = len(temperatures)
    A_bar = np.zeros((strain_dof, n_modes - 7, n_temps))
    D_xi = np.zeros((n_modes - 7, n_modes - 7, n_temps))
    tau_theta = np.zeros((strain_dof, n_temps))
    C_bar = np.zeros((strain_dof, strain_dof, n_temps))
    # interpolate_temp = lambda x1, x2, alpha: x1 + alpha * (x2 - x1)
    # dns_temperatures = interpolate_temp(temp1, temp2, sample_alphas)
    sample_temperatures = np.array([sample['temperature'] for sample in samples])
    temp1, temp2 = min(sample_temperatures), max(sample_temperatures)
    sample_alphas = (sample_temperatures - temp1) / (temp2 - temp1)
    # for idx, temperature in enumerate(temperatures):
    for idx in prange(n_temps):
        temperature = temperatures[idx]
        alpha = (temperature - temp1) / (temp2 - temp1)
        upper_bound = np.searchsorted(sample_alphas, alpha)
        id1 = upper_bound if upper_bound > 0 else 1
        id0 = id1 - 1

        E0 = samples[id0]['strain_localization']
        E1 = samples[id1]['strain_localization']
        E01 = np.ascontiguousarray(np.concatenate((E0, E1), axis=-1))

        sampling_C = np.stack((samples[id0]['mat_stiffness'], samples[id1]['mat_stiffness'])).transpose([1, 0, 2, 3])
        sampling_eps = np.stack((samples[id0]['mat_thermal_strain'], samples[id1]['mat_thermal_strain'])).transpose([1, 0, 2, 3])
        plastic_modes = samples[id0]['plastic_modes']  # TODO: does exist?
        # normalization_factor_mech = samples[idx]['normalization_factor_mech']  # TODO: does exist?
        ref_C = np.stack(([stiffness_cu(temperature), stiffness_wsc(temperature)]))
        ref_eps = np.expand_dims(np.stack(([thermal_strain_cu(temperature), thermal_strain_wsc(temperature)])), axis=2)

        # interpolated quantities using an implicit interpolation scheme with four DOF
        approx_C, approx_eps = opt4(sampling_C, sampling_eps, ref_C, ref_eps)
        Eopt4, _ = interpolate_fluctuation_modes(E01, approx_C, approx_eps, plastic_modes, mat_id, n_gauss, strain_dof, n_modes,
                                                 n_gp)
        Sopt4 = construct_stress_localization(Eopt4, ref_C, ref_eps, plastic_modes, mat_id, n_gauss, strain_dof)
        # effSopt = volume_average(Sopt4)
        A_bar[:,:,idx], D_xi[:,:,idx], tau_theta[:,idx], C_bar[:,:,idx] = compute_ntfa_matrices(Sopt4, plastic_modes, mesh)
    return C_bar, tau_theta, A_bar, tau_xi, D_xi, D_theta
    """
    pass


def save_tabular_data(file_name, data_path, temperatures, C_bar, tau_theta, A_bar, tau_xi, D_xi, D_theta, A0, A1, C0, C1):
    """
    Save tabular data
    :param file_name: e.g. "input/simple_3d_rve_combo.h5"
    :param data_path:
    :param temperatures:
    :param C_bar: tabular data for C_bar with shape (strain_dof, strain_dof, n_temp)
    :param tau_theta: tabular data for tau_theta with shape (strain_dof, n_temp)
    :param A_bar: tabular data for A_bar with shape (strain_dof, n_modes, n_temp)
    :param tau_xi: tabular data for tau_xi with shape (n_modes, n_temp)
    :param D_xi: tabular data for D_xi with shape (n_modes, n_modes, n_temp)
    :param D_theta: tabular data for D_theta with shape (n_temp)
    :param A0: tabular data for A0 with shape (strain_dof, n_modes, n_temp)
    :param A1: tabular data for A1 with shape (strain_dof, n_modes, n_temp)
    :param C0: tabular data for C0 with shape (strain_dof, strain_dof, n_temp)
    :param C1: tabular data for C1 with shape (strain_dof, strain_dof, n_temp)
    """
    with h5py.File(file_name, 'a') as file:
        dset_sim = file[data_path]
        dset = dset_sim.parent
        ntfa_path = re.sub('_sim$', '_ntfa', dset_sim.name)
        if ntfa_path in file:
            # Delete h5 group with tabular data if it already exists
            del file[ntfa_path]
        dset_ntfa = dset.create_group(ntfa_path)
        [dset_ntfa.attrs.create(key, value) for key, value in dset_sim.attrs.items()]
        dset_temperatures = dset_ntfa.create_dataset('temperatures', data=temperatures)
        dset_C_bar = dset_ntfa.create_dataset('C_bar', data=C_bar)
        dset_tau_theta = dset_ntfa.create_dataset('tau_theta', data=tau_theta)
        dset_A_bar = dset_ntfa.create_dataset('A_bar', data=A_bar)
        dset_tau_xi = dset_ntfa.create_dataset('tau_xi', data=tau_xi)
        dset_D_xi = dset_ntfa.create_dataset('D_xi', data=D_xi)
        dset_D_theta = dset_ntfa.create_dataset('D_theta', data=D_theta)
        dset_A0 = dset_ntfa.create_dataset('A0', data=A0)
        dset_A1 = dset_ntfa.create_dataset('A1', data=A1)
        dset_C0 = dset_ntfa.create_dataset('C0', data=C0)
        dset_C1 = dset_ntfa.create_dataset('C1', data=C1)


def read_tabular_data(file_name, data_path):
    """
    Save tabular data
    :param file_name: e.g. "input/simple_3d_rve_combo.h5"
    :param data_path:
    :return:
        temperatures:
        C_bar: tabular data for C_bar with shape (strain_dof, strain_dof, n_temp)
        tau_theta: tabular data for tau_theta with shape (strain_dof, n_temp)
        A_bar: tabular data for A_bar with shape (strain_dof, n_modes, n_temp)
        tau_xi: tabular data for tau_xi with shape (n_modes, n_temp)
        D_xi: tabular data for D_xi with shape (n_modes, n_modes, n_temp)
        D_theta: tabular data for D_theta with shape (n_temp)
        A0: tabular data for A0 with shape (strain_dof, n_modes, n_temp)
        A1: tabular data for A1 with shape (strain_dof, n_modes, n_temp)
        C0: tabular data for C0 with shape (strain_dof, strain_dof, n_temp)
        C1: tabular data for C1 with shape (strain_dof, strain_dof, n_temp)
    """
    with h5py.File(file_name, 'r') as file:
        if '_ntfa' in data_path:
            ntfa_path = data_path
        else:
            ntfa_path = re.sub('_sim$', '_ntfa', data_path)
        dset_ntfa = file[ntfa_path]
        temperatures = dset_ntfa['temperatures'][:]
        C_bar = dset_ntfa['C_bar'][:]
        tau_theta = dset_ntfa['tau_theta'][:]
        A_bar = dset_ntfa['A_bar'][:]
        tau_xi = dset_ntfa['tau_xi'][:]
        D_xi = dset_ntfa['D_xi'][:]
        D_theta = dset_ntfa['D_theta'][:]
        A0 = dset_ntfa['A0'][:]
        A1 = dset_ntfa['A1'][:]
        C0 = dset_ntfa['C0'][:]
        C1 = dset_ntfa['C1'][:]
    return temperatures, C_bar, tau_theta, A_bar, tau_xi, D_xi, D_theta, A0, A1, C0, C1

"""
TO BE REMOVED:
"""

def create_dummy_plastic_snapshots(n_integration_points, strain_dof, n_frames=100):
    """
    TODO: remove when FANS can compute plastic snapshots
    """
    # plastic_snapshots = np.random.rand(n_integration_points, strain_dof, n_frames)
    plastic_snapshots = np.random.rand(n_integration_points, strain_dof)[:, :, np.newaxis] * np.random.rand(n_frames) \
        + 1e-2 * np.random.rand(n_integration_points, strain_dof, n_frames)
    return plastic_snapshots


def create_dummy_plastic_modes(n_integration_points, strain_dof, N_modes):
    """
    TODO: remove when FANS can compute plastic snapshots
    """
    plastic_modes = np.random.rand(n_integration_points, strain_dof, N_modes)
    return plastic_modes


def create_dummy_plastic_strain_localization(strain_localization, N_modes):
    E_eps = strain_localization[:, :, :6]
    E_xi = np.random.rand(*strain_localization.shape[:2], N_modes)
    E_theta = np.expand_dims(strain_localization[:, :, -1], axis=2)
    strain_localization = np.concatenate([E_eps, E_xi, E_theta], axis=2)
    return strain_localization
