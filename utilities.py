import contextlib
import timeit

import h5py
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.sparse
from sympy import symbols, lambdify, Array
from numba import jit, njit, prange, vectorize, prange

plt.rcParams.update({
    'font.size': 8,
    'lines.linewidth': 0.8,
    'lines.markersize': 4,
    'markers.fillstyle': 'none',
    'lines.markeredgewidth': 0.5,
    'text.usetex': True,
    'text.latex.preamble': r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{upgreek} \usepackage{helvet}"
    # \usepackage{sansmath} \sansmath
})

cm = 1 / 2.54  # centimeters in inches

def plot_and_save(xlabel, ylabel, fig_name, xlim=None, ylim=None, loc='best'):
    gca = plt.gca()
    gca.set_xlim(xlim)
    gca.set_ylim(ylim)
    gca.grid(ls='--', color='gray', linewidth=0.5)
    gca.set_xlabel(rf'{xlabel}')
    gca.set_ylabel(rf'{ylabel}')
    gca.legend(loc=loc, facecolor=(0.8, 0.8, 0.8, 0.6), edgecolor='black')
    plt.tight_layout(pad=0.025)
    plt.savefig(f'output/{fig_name}.png')
    plt.show()

def ecdf(x):
    """empirical cumulative distribution function"""
    # plt.step(*ecdf(np.asarray([1, 1, 2, 3, 4])), where='post')
    n = len(x)
    return [np.hstack((np.min(x), np.sort(np.squeeze(np.asarray(x))))), np.hstack((0, np.linspace(1 / n, 1, n)))]

def voxel_quadrature(discretisation, strain_dof=6, nodal_dof=3):
    """
    Voxel formulation with full integration, works with structured grids only
    refer to VTK_VOXEL=11 in https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Testing/Baseline/Cxx/GeometricObjects/TestLinearCellDemo.png 
    :param strain_dof: strain degrees of freedom
    :param nodal_dof: nodal degrees of freedom
    :param discretisation: Number of elements in each direction
    :return: 8 gradient_operators and 8 integration_weights
    """
    xi, eta, zeta = symbols('xi,eta,zeta')

    ref_coordinates = np.asarray([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1], [-1, -1, 1], [1, -1, 1], [-1, 1, 1],
                                  [1, 1, 1]])
    n_nodes = ref_coordinates.shape[0]

    shape_functions = lambda xi, eta, zeta: Array(np.prod(1 + ref_coordinates * np.asarray([xi, eta, zeta]), axis=1) / n_nodes)
    dN_dxi = lambdify((xi, eta, zeta), shape_functions(xi, eta, zeta).diff(xi))
    dN_deta = lambdify((xi, eta, zeta), shape_functions(xi, eta, zeta).diff(eta))
    dN_dzeta = lambdify((xi, eta, zeta), shape_functions(xi, eta, zeta).diff(zeta))

    quadrature_coordinates = ref_coordinates / np.sqrt(3)

    n_gauss = quadrature_coordinates.shape[0]

    gradient_operators = np.empty((n_gauss, strain_dof, nodal_dof * n_nodes))
    integration_weights = np.empty(n_gauss)

    for idx in range(n_gauss):
        shape_function_derivatives = np.asarray(
            [dN_dxi(*quadrature_coordinates[idx]),
             dN_deta(*quadrature_coordinates[idx]),
             dN_dzeta(*quadrature_coordinates[idx])]) * 2 * np.asarray(discretisation)[:, None]
        # l_ref=2 / (1/L_physical = 1/N) -> 2 * N

        # [xx,yy,zz,sqrt(2)*xy,sqrt(2)*xz,sqrt(2)*yz]
        position_x = [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
        position_y = [[0, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]]
        position_z = [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
        gradient_operator = np.asarray(
            ((np.kron(shape_function_derivatives[0, :], position_x) + np.kron(shape_function_derivatives[1, :], position_y) +
              np.kron(shape_function_derivatives[2, :], position_z))) /
            np.asarray([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)]).reshape(-1, 1))
        # Mandel notation [1 / sqrt(2) = 1/2 * sqrt(2)] # https://en.wikipedia.org/wiki/Deformation_(mechanics) 6 components x (8 nodes x 3 dof)

        gp_weight = 1 / (n_gauss * np.prod(discretisation))

        gradient_operators[idx] = gradient_operator
        integration_weights[idx] = gp_weight

    return gradient_operators, integration_weights

def read_h5(file_name, data_path, temperatures, get_mesh=True):
    """
    Read an H5 file that contains responses of simulated microstructures
    :param file_name: e.g. "input/simple_3d_rve_combo.h5"
    :param data_path: the path to the simulation results within the h5 file, e.g. '/ms_1p/dset0_sim'
    :param temperatures: all responses corresponding to these temperatures will be read 
    :return:
        mesh: dictionary that contains microstructural details such as volume fraction, voxel type, ...
        samples: list of simulation results at each temperature
    """
    axis_order = [0, 2, 1]  # n_gauss x strain_dof x 7 (7=6 mechanical + 1 thermal expansion)

    samples = []
    with h5py.File(file_name, 'r') as file:
        mesh = {'vox_type': file[f'{data_path}'].attrs['vox_type'][0]}
        for temperature in temperatures:
            sample = {}
            samples.append(sample)
            sample['temperature'] = temperature
            sample['eff_heat_capacity'] = file[f'{data_path}/eff_heat_capacity_{temperature:07.2f}'][:]
            sample['eff_thermal_strain'] = file[f'{data_path}/eff_thermal_strain_{temperature:07.2f}'][:]
            # eff_stiffness is transposed because it's coming from matlab to h5 file to python
            sample['eff_stiffness'] = file[f'{data_path}/eff_stiffness_{temperature:07.2f}'][:].T
            sample['eff_conductivity'] = file[f'{data_path}/eff_conductivity_{temperature:07.2f}'][:].T

            sample['input_conductivity'] = file[f'{data_path}/material_{temperature:07.2f}'].attrs['conductivity']
            sample['input_heat_capacity'] = file[f'{data_path}/material_{temperature:07.2f}'].attrs['heat_capacity']
            sample['input_elastic_modulus'] = file[f'{data_path}/material_{temperature:07.2f}'].attrs['elastic_modulus']
            sample['input_thermal_strain'] = file[f'{data_path}/material_{temperature:07.2f}'].attrs['thermal_strain']
            sample['input_poisson_ratio'] = file[f'{data_path}/material_{temperature:07.2f}'].attrs['poisson_ratio']  # constant

            sample['normalization_factor_mech'] = file[f'{data_path}'].attrs['normalization_factor_mech'][0]
            sample['normalization_factor_therm'] = file[f'{data_path}'].attrs['normalization_factor_therm'][0]

            sample['mat_stiffness'] = file[f'{data_path}/localization_mat_stiffness_{temperature:07.2f}'][:]

            sample['mat_thermal_strain'] = file[f'{data_path}/localization_mat_thermal_strain_{temperature:07.2f}'][:][..., None]

            sample['combo_strain_loc0'] = None
            sample['combo_strain_loc1'] = None

            with contextlib.suppress(Exception):
                sample['strain_localization'] = file[f'{data_path}/localization_strain_{temperature:07.2f}'][:].transpose(
                    axis_order)

                if mesh['vox_type'] == 'combo':
                    sample['combo_strain_loc0'] = \
                        file[f'{data_path}/localization_strain0_{temperature:07.2f}'][:].transpose(axis_order)

                    sample['combo_strain_loc1'] = \
                        file[f'{data_path}/localization_strain1_{temperature:07.2f}'][:].transpose(axis_order)
        if get_mesh:
            mesh['volume_fraction'] = file[f'{data_path}'].attrs['combo_volume_fraction']
            mesh['combo_discretisation'] = np.int64(file[f'{data_path}'].attrs['combo_discretisation'])
            mesh['n_gauss'] = np.int64(file[f'{data_path}'].attrs['element_integration_points'])
            mesh['element_formulation'] = file[f'{data_path}'].attrs['element_formulation'][0]
            mesh['convergence_tolerance'] = file[f'{data_path}'].attrs['convergence tolerance'][0]
            mesh['assembly_idx'] = np.int64(file[f'{data_path}/assembly_idx'][:] - 1)  # shift to zero based indexing
            mesh['n_elements'] = np.prod(mesh['combo_discretisation'])
            mesh['n_nodes'] = mesh['n_elements']  # due to periodic boundary conditions
            mesh['strain_dof'] = 6  # TODO these values should be stored and read from h5
            mesh['nodal_dof'] = 3
            mesh['element_nodes'] = 8
            mesh['dof'] = mesh['n_nodes'] * mesh['nodal_dof']
            mesh['element_dof'] = mesh['element_nodes'] * mesh['nodal_dof']
            mesh['element_strain_dof'] = mesh['n_gauss'] * mesh['strain_dof']
            mesh['n_integration_points'] = mesh['n_gauss'] * mesh['n_elements']

            mesh['mat_id'] = np.int64(file[f'{data_path}/mat_id'][0])
            if mesh['vox_type'] == 'combo':
                mesh['combo_idx'] = tuple(np.int64(file[f'{data_path}/combo_idx'][0]) - 1)  # shift to zero based indexing
                mesh['combo_vol_frac0'] = file[f'{data_path}/combo_vol_frac0'][0]
                mesh['mat_id'][mesh['mat_id'] == 2] = np.arange(len(mesh['combo_idx'])) + 2
                # for element_id, phase_id in enumerate(mesh['mat_id']):
                #     if phase_id == 2:
                #         mesh['mat_id'][element_id] = mesh['combo_idx'].index(element_id) + 2

            assert (mesh['element_formulation'] == 'hex8'
                    and mesh['n_gauss'] == 8), NotImplementedError('Only fully integrated voxel element is implemented')

            gradient_operators, integration_weights = voxel_quadrature(mesh['combo_discretisation'], mesh['strain_dof'],
                                                                       mesh['nodal_dof'])

            gradient_operators_times_w = gradient_operators * integration_weights[:, None, None]
            mesh['gradient_operators_times_w'] = gradient_operators_times_w

            # costly assembly of the global gradient matrix, replaced by an efficient implementation below
            # t_start = timeit.default_timer()
            # global_gradient = scipy.sparse.lil_matrix((mesh['n_elements'] * mesh['element_strain_dof'], mesh['dof']))
            # for element_id in range(mesh['n_elements']):
            #     for local_gp_id in range(mesh['n_gauss']):
            #         idx = element_id * mesh['element_strain_dof'] + local_gp_id * mesh['strain_dof']
            #         global_gradient[idx:idx + mesh['strain_dof'],
            #                         mesh['assembly_idx'][element_id]] += gradient_operators_times_w[local_gp_id]
            # t_stop = timeit.default_timer()
            # print(" Time0: ", t_stop - t_start)

            cols = np.tile(np.tile(mesh['assembly_idx'], mesh['strain_dof']), mesh['n_gauss']).flatten()
            rows = np.repeat(np.arange(mesh['n_elements'] * mesh['element_strain_dof']), mesh['element_dof'])
            data = np.tile(gradient_operators_times_w.flatten(), mesh['n_elements'])
            global_gradient = scipy.sparse.coo_matrix((data, (rows, cols)),
                                                      (mesh['n_elements'] * mesh['element_strain_dof'], mesh['dof']))

            mesh['global_gradient'] = global_gradient.tocsr()
            print(f'global gradient: {(mesh["global_gradient"].data.nbytes) / 1024 ** 2} MB')

    return mesh, samples

def verify_data(mesh, sample):
    """
    Sanity check to see if it is possible to reconstruct stress, strain fields and effective properties
    :param mesh: from read_h5()
    :param sample: from read_h5()
    :param gradient_operators: from voxel_quadrature()
    :return: Nothing, will do assertion withing this function
    """
    convergence_tolerance, strain_localization = mesh['convergence_tolerance'], sample['strain_localization']
    eff_stiffness, eff_thermal_strain = sample['eff_stiffness'], sample['eff_thermal_strain']
    mat_thermal_strain, mat_stiffness = sample['mat_thermal_strain'], sample['mat_stiffness']
    combo_strain_loc0, combo_strain_loc1 = sample['combo_strain_loc0'], sample['combo_strain_loc1']

    macro_strain = np.asarray([3, .7, 1.5, 0.5, 2, 1])
    zeta = np.hstack((macro_strain, 1))  # 1 accounts for thermoelastic strain, more details in the paper cited in readme.md

    strain = macro_strain + np.einsum('ijk,k', strain_localization, zeta, optimize='optimal')

    stress_localization = construct_stress_localization(strain_localization, mat_stiffness, mat_thermal_strain, mesh['mat_id'],
                                                        mesh['n_gauss'], mesh['strain_dof'])
    eff_stiffness_from_localization = volume_average(stress_localization)

    stress = np.einsum('ijk,k', stress_localization, zeta, optimize='optimal')
    residual = compute_residual(stress, mesh['dof'], mesh['n_elements'], mesh['element_dof'], mesh['n_gauss'],
                                mesh['assembly_idx'], mesh['gradient_operators_times_w'])

    abs_err = eff_stiffness_from_localization - np.hstack((eff_stiffness, -np.reshape(eff_stiffness @ eff_thermal_strain,
                                                                                      (-1, 1))))
    err = lambda x, y: np.mean(la.norm(x - y) / la.norm(y))

    assert err(eff_stiffness_from_localization,
               np.vstack((eff_stiffness, -eff_stiffness @ eff_thermal_strain)).T) < convergence_tolerance, \
        'incompatibility between stress_localization and effective quantities'

    with np.printoptions(precision=4, suppress=True, formatter={'float': '{:>2.2e}'.format}, linewidth=100):
        print('\n', abs_err)

    assert la.norm(residual, np.inf) / sample['normalization_factor_mech'] < 10 * convergence_tolerance, \
        'stress field is not statically admissible'

    stress_localization0, stress_localization1, combo_stress_loc0, combo_stress_loc1 = construct_stress_localization_phases(
        strain_localization, mat_stiffness, mat_thermal_strain, combo_strain_loc0, combo_strain_loc1, mesh)

    combo_stress0 = np.einsum('ijk,k', combo_stress_loc0, zeta, optimize='optimal') if combo_stress_loc0 is not None else None
    combo_stress1 = np.einsum('ijk,k', combo_stress_loc1, zeta, optimize='optimal') if combo_stress_loc1 is not None else None

    stress0 = np.einsum('ijk,k', stress_localization0, zeta, optimize='optimal')
    stress1 = np.einsum('ijk,k', stress_localization1, zeta, optimize='optimal')
    average_stress = volume_average(stress)
    average_stress_0, average_stress_1 = volume_average_phases(stress0, stress1, combo_stress0, combo_stress1, mesh)

    vol_frac0 = mesh['volume_fraction'][0]
    vol_frac1 = mesh['volume_fraction'][1]

    assert err(average_stress, \
               vol_frac0 * average_stress_0 + vol_frac1 * average_stress_1) < convergence_tolerance, \
        'phasewise volume average is not admissible'

def compute_residual(stress, dof, n_elements, element_dof, n_gauss, assembly_idx, gradient_operators_times_w):
    """
    Compute FEM residual given a stress field
    :param stress: stress field or a list of stress fields
    :param mesh: from read_h5()
    :param gradient_operators: from voxel_quadrature()
    :return: residual vector
    """
    stresses = stress
    if not isinstance(stress, list):
        stresses = [stress]

    residuals = np.zeros((dof, len(stresses)))

    for element_id in range(n_elements):
        for idx, stress in enumerate(stresses):
            elmental_force = np.zeros(element_dof)
            for local_gp_id in range(n_gauss):
                gp_id = element_id * n_gauss + local_gp_id
                elmental_force += gradient_operators_times_w[local_gp_id].T @ stress[gp_id]
            residuals[assembly_idx[element_id], idx] += elmental_force
    return residuals

def compute_residual_efficient(stress, global_gradient):
    stresses = stress
    if not isinstance(stress, list):
        stresses = [stress]

    return (np.vstack([x.flatten() for x in stresses]) @ global_gradient).T

@jit(nopython=True, cache=True, parallel=True, nogil=True)
def compute_err_indicator(stress_loc, gradient_operators_times_w, dof, n_gauss, assembly_idx):
    """
    Compute an error indicator that is independent of the loading
    :param stress_loc: stress localization 3D array
    :param mesh: from read_h5()
    :param gradient_operators: from voxel_quadrature()
    :return: error indicator matrix
    """
    err_indicator = np.zeros((dof, stress_loc.shape[-1]))
    for gp_id in range(stress_loc.shape[0]):
        element_id = gp_id // n_gauss
        local_gp_id = gp_id % n_gauss
        err_indicator[assembly_idx[element_id]] += \
            gradient_operators_times_w[local_gp_id].T @ stress_loc[gp_id]
    # err_indicator = np.zeros((dof))
    # for element_id in range(n_elements):
    #     err_indicator[assembly_idx[element_id]] += np.einsum('ijk,ijl->k', gradient_operators_times_w,
    #                                                          stress_loc[element_id * n_gauss:element_id * n_gauss + n_gauss],
    #                                                          optimize='optimal')
    # return la.norm(err_indicator)
    return err_indicator

def compute_err_indicator_efficient(stress_loc, global_gradient):
    return global_gradient.T @ stress_loc.reshape(global_gradient.shape[0], -1)

# @jit(nopython=True, cache=True, parallel=True, nogil=True)
def cheap_err_indicator(stress_loc, global_gradient):
    return la.norm(global_gradient.T @ np.sum(stress_loc, -1).flatten())

@jit(nopython=True, cache=True, parallel=True, nogil=True)
def construct_stress_localization(strain_localization, mat_stiffness, mat_thermal_strain, mat_id, n_gauss, strain_dof):
    """
    Construct stress localization operator out of the strain localization one.
    :param strain_localization: strain localization 3D array
    :param mat_stiffness: material stiffness of each material phase
    :param mat_thermal_strain: thermal strain of each material phase
    :note the structure of mat_stiffness and mat_thermal_strain follows: [QoI_phase0,QoI_phase1,QoI_comb_voxel0,QoI_comb_voxel1,...]
    :param mat_id: material ID, from read_h5()
    :return: stress localization 3D array
    """
    stress_localization = np.empty_like(strain_localization)
    I = np.eye(strain_dof)
    for gp_id in prange(strain_localization.shape[0]):
        phase_id = mat_id[gp_id // n_gauss]
        P = np.hstack((-I, mat_thermal_strain[phase_id]))
        stress_localization[gp_id] = mat_stiffness[phase_id] @ (strain_localization[gp_id] - P)
    return stress_localization

def construct_stress_localization_phases(strain_localization, mat_stiffness, mat_thermal_strain, combo_strain_loc0,
                                         combo_strain_loc1, mesh):
    """
    Same as construct_stress_localization() but it returns stress localization operators for each material phase
    :param strain_localization: strain localization 3D array
    :param mat_stiffness: material stiffness of each material phase
    :param mat_thermal_strain: thermal strain of each material phase
    :param combo_strain_loc0: strain localization array of phase0 of all combo_voxels
    :param combo_strain_loc1: strain localization array of phase1 of all combo_voxels
    :param mesh: from read_h5()
    :return:
        stress_localization0: stress localization of pure phase0 voxels
        stress_localization1: stress localization of pure phase1 voxels
        combo_stress_loc0: stress localization of phase0 of all combo_voxels
        combo_stress_loc1: stress localization of phase1 of all combo_voxels
    """
    n_gauss = mesh['n_gauss']
    idx0 = np.repeat(mesh['mat_id'] == 0, n_gauss)
    idx1 = np.repeat(mesh['mat_id'] == 1, n_gauss)
    stress_localization = construct_stress_localization(strain_localization, mat_stiffness, mat_thermal_strain, mesh['mat_id'],
                                                        mesh['n_gauss'], mesh['strain_dof'])
    stress_localization0 = stress_localization[idx0] if np.any(idx0) else np.zeros((1, *stress_localization.shape[1:]))
    stress_localization1 = stress_localization[idx1] if np.any(idx1) else np.zeros((1, *stress_localization.shape[1:]))
    combo_stress_loc0 = None
    combo_stress_loc1 = None

    if mesh['vox_type'] == 'combo':
        combo_stress_loc0 = np.empty_like(combo_strain_loc0)
        combo_stress_loc1 = np.empty_like(combo_strain_loc1)
        I = np.eye(mesh['strain_dof'])
        for idx in range(len(mesh['combo_idx'])):
            stiffness_idx = idx + 2
            P = np.hstack([-I, mat_thermal_strain[stiffness_idx]])
            combo_stress_loc0[idx] = mat_stiffness[0] @ combo_strain_loc0[idx] - mat_stiffness[stiffness_idx] @ P
            combo_stress_loc1[idx] = mat_stiffness[1] @ combo_strain_loc1[idx] - mat_stiffness[stiffness_idx] @ P
    return stress_localization0, stress_localization1, combo_stress_loc0, combo_stress_loc1

def volume_average_phases(stress0, stress1, combo_stress0, combo_stress1, mesh):
    """
    Volume average of stress field in each phase
    :param stress0: stress field corresponding to pure phase0
    :param stress1: stress field corresponding to pure phase1
    :param combo_stress0: stress field of phase0 of all combo_voxels
    :param combo_stress1: stress field of phase1 of all combo_voxels
    :param mesh: from read_h5()
    :return:
        average_stress_0: stress volume average in phase0
        average_stress_1: stress volume average in phase1
    """
    average_stress_0 = np.mean(stress0, axis=0)
    average_stress_1 = np.mean(stress1, axis=0)

    if mesh['vox_type'] == 'combo':
        n_gauss = mesh['n_gauss']
        combo_vol_frac0 = mesh['combo_vol_frac0']
        n_voxels0 = np.sum(mesh['mat_id'] == 0) * n_gauss
        n_voxels1 = np.sum(mesh['mat_id'] == 1) * n_gauss

        average_stress_0 = ((np.sum(stress0, axis=0) + np.sum(n_gauss * combo_stress0 * combo_vol_frac0[:, None], axis=0)) /
                            (n_voxels0 + n_gauss * np.sum(combo_vol_frac0)))
        average_stress_1 = ((np.sum(stress1, axis=0) + np.sum(n_gauss * combo_stress1 * (1 - combo_vol_frac0[:, None]), axis=0)) /
                            (n_voxels1 + n_gauss * np.sum(1 - combo_vol_frac0)))

    return average_stress_0, average_stress_1

def volume_average(field):
    """ Volume average of a given field in case of identical weights that sum to one """
    return np.mean(field, axis=0)
