import contextlib
import re
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.sparse
from numba import jit, prange
from sympy import symbols, lambdify, Array
from operator import itemgetter
from optimize_alpha import opt4
from interpolate_fluctuation_modes import interpolate_fluctuation_modes
from microstructures import *
from material_parameters import *

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


def read_h5(file_name, data_path, temperatures, get_mesh=True, dummy_plastic_data=True):
    """
    Read an H5 file that contains responses of simulated microstructures
    :param file_name: e.g. "input/simple_3d_rve_combo.h5"
    :param data_path: the path to the simulation results within the h5 file, e.g. '/ms_1p/dset0_sim'
    :param temperatures: all responses corresponding to these temperatures will be read 
    :return:
        mesh: dictionary that contains microstructural details such as volume fraction, voxel type, ...
        samples: list of simulation results at each temperature
    """
    axis_order = [0, 2, 1]  # n_gauss x strain_dof x (n_modes + 7) (n_modes + 7 = 6 mechanical + 1 thermal expansion + n_modes plastic)

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

            with contextlib.suppress(Exception):  # used because some strain localizations are deleted to make h5 files smaller
                sample['strain_localization'] = file[f'{data_path}/localization_strain_{temperature:07.2f}'][:].transpose(
                    axis_order)

                if mesh['vox_type'] == 'combo':
                    sample['combo_strain_loc0'] = \
                        file[f'{data_path}/localization_strain0_{temperature:07.2f}'][:].transpose(axis_order)

                    sample['combo_strain_loc1'] = \
                        file[f'{data_path}/localization_strain1_{temperature:07.2f}'][:].transpose(axis_order)

                if 'plastic_modes' in file[f'{data_path}'].keys():
                    sample['plastic_modes'] = file[f'{data_path}/plastic_modes'][:].transpose(axis_order)
                else:
                    sample['plastic_modes'] = np.zeros((*sample['strain_localization'].shape[:2], 0))
                    # sample['plastic_modes'] = create_dummy_plastic_modes(*sample['strain_localization'].shape[:2], N_modes=13)
                    # sample['strain_localization'] = create_dummy_plastic_strain_localization(sample['strain_localization'], N_modes=13)
                    # if mesh['vox_type'] == 'combo':
                    #     sample['combo_strain_loc0'] = create_dummy_plastic_strain_localization(sample['combo_strain_loc0'], N_modes=13)
                    #     sample['combo_strain_loc1'] = create_dummy_plastic_strain_localization(sample['combo_strain_loc1'], N_modes=13)

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
    plastic_modes = sample['plastic_modes']

    macro_strain = np.asarray([3, .7, 1.5, 0.5, 2, 1])
    xi = np.ones(plastic_modes.shape[-1])
    # 1 accounts for thermoelastic strain, more details in the paper cited in readme.md
    # xi accounts for plastic mode activations
    zeta = np.hstack((macro_strain, 1, xi))

    strain = macro_strain + np.einsum('ijk,k', strain_localization, zeta, optimize='optimal')

    stress_localization = construct_stress_localization(strain_localization, mat_stiffness, mat_thermal_strain, plastic_modes,
                                                        mesh['mat_id'], mesh['n_gauss'], mesh['strain_dof'])
    eff_stiffness_from_localization = volume_average(stress_localization)

    stress = np.einsum('ijk,k', stress_localization, zeta, optimize='optimal')
    residual = compute_residual(stress, mesh['dof'], mesh['n_elements'], mesh['element_dof'], mesh['n_gauss'],
                                mesh['assembly_idx'], mesh['gradient_operators_times_w'])

    eff_thermoelastic_stiffness = eff_stiffness_from_localization[:, :7]
    abs_err = eff_thermoelastic_stiffness - np.hstack((eff_stiffness, -np.reshape(eff_stiffness @ eff_thermal_strain,
                                                                                  (-1, 1))))
    # - C @ eff_plastic_strain # eff_plastic_strain is not stored because it depends on the macroscopic strain
    err = lambda x, y: np.mean(la.norm(x - y) / la.norm(y))

    assert err(eff_thermoelastic_stiffness,
               np.vstack((eff_stiffness, -eff_stiffness @ eff_thermal_strain)).T) < convergence_tolerance, \
        'incompatibility between stress_localization and effective quantities'

    with np.printoptions(precision=4, suppress=True, formatter={'float': '{:>2.2e}'.format}, linewidth=100):
        print('\n', abs_err)

    assert la.norm(residual, np.inf) / sample['normalization_factor_mech'] < 10 * convergence_tolerance, \
        'stress field is not statically admissible'

    stress_localization0, stress_localization1, combo_stress_loc0, combo_stress_loc1 = construct_stress_localization_phases(
        strain_localization, mat_stiffness, mat_thermal_strain, plastic_modes, combo_strain_loc0, combo_strain_loc1, mesh)

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

    plastic_modes = sample['plastic_modes']
    n_modes = plastic_modes.shape[-1]
    gramian = volume_average(np.einsum('ijk,ijl->ikl', plastic_modes, plastic_modes))
    assert np.allclose(gramian, np.diag(np.diag(gramian))), 'plastic modes are not orthogonal'
    assert np.allclose([volume_average(norm_2(plastic_modes[:,:,i])) for i in range(n_modes)], vol_frac0), 'plastic modes are not normalized correctly'


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


@jit(nopython=True, cache=True, parallel=True, nogil=True)
def cheap_err_indicator(stress_loc, global_gradient):
    return la.norm(global_gradient.T @ np.sum(stress_loc, -1).flatten())


@jit(nopython=True, cache=True, parallel=True, nogil=True)
def construct_stress_localization(strain_localization, mat_stiffness, mat_thermal_strain, plastic_modes, mat_id, n_gauss,
                                  strain_dof):
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
        P = np.hstack((-I, mat_thermal_strain[phase_id], plastic_modes[gp_id]))
        stress_localization[gp_id] = mat_stiffness[phase_id] @ (strain_localization[gp_id] - P)
    return stress_localization


def construct_stress_localization_phases(strain_localization, mat_stiffness, mat_thermal_strain, plastic_modes, combo_strain_loc0,
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
    stress_localization = construct_stress_localization(strain_localization, mat_stiffness, mat_thermal_strain, plastic_modes,
                                                        mesh['mat_id'],
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
            # TODO: do we need separate plastic modes for each material phase?
            P = np.hstack((-I, mat_thermal_strain[stiffness_idx], plastic_modes[stiffness_idx // 2]))
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
    """
    Volume average of a given field in case of identical weights that sum to one
    :param field: array with shape (n_integration_points, ...)
    """
    return np.mean(field, axis=0)


def inner_product(a, b):
    """
    Compute inner product between tensor fields a and b in case of identical weights that sum to one
    :param a: array with shape (n_integration_points, ...)
    :param b: array with same shape as b
    """
    assert a.shape == b.shape
    summation_axes = tuple(ax for ax in range(1, a.ndim))
    return np.sum(a * b, axis=summation_axes)


def norm_2(a):
    """
    Compute euclidean norm of tensor field a in case of identical weights that sum to one
    :param a: array with shape (n_integration_points, ...)
    :param b: array with same shape as b
    """
    return np.sqrt(inner_product(a, a))


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


def compute_ntfa_matrices(strain_localization, stress_localization, plastic_modes, thermal_strain, mesh):
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
        D_theta with shape (strain_dof, n_modes)
    """
    strain_dof = mesh['strain_dof']
    mat_id = mesh['mat_id']
    n_modes = plastic_modes.shape[2]
    
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
    tau_xi = volume_average(np.einsum('ni,nij->nj', E_theta - thermal_strain, S_xi))

    # Compute D_xi via < (E_xi - P_xi).T @ S_xi >
    D_xi = volume_average((E_xi - plastic_modes).transpose((0, 2, 1)) @ S_xi)

    # Compute D_theta via < (E_theta - P_theta).T @ S_theta >
    D_theta = volume_average(np.einsum('ni,ni->n', E_theta - thermal_strain, S_theta))

    return C_bar, tau_theta, A_bar, tau_xi, D_xi, D_theta


def compute_tabular_data_for_ms(ms_id, temperatures):
    """
    Perform `compute_tabular_data` for microstructure with id `ms_id`
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
    """
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
    # interpolate_temp = lambda x1, x2, alpha: x1 + alpha * (x2 - x1)
    # dns_temperatures = interpolate_temp(temp1, temp2, sample_alphas)
    sample_temperatures = np.array([sample['temperature'] for sample in samples])
    temp1, temp2 = min(sample_temperatures), max(sample_temperatures)
    sample_alphas = (sample_temperatures - temp1) / (temp2 - temp1)
    plastic_modes = samples[0]['plastic_modes']
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
        C_bar[:, :, idx], tau_theta[:, idx], A_bar[:, :, idx], tau_xi[:, idx], D_xi[:, :, idx], D_theta[idx] = \
            compute_ntfa_matrices(E, S, plastic_modes, eps[0,:,0], mesh)
    return C_bar, tau_theta, A_bar, tau_xi, D_xi, D_theta


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


def save_tabular_data(file_name, data_path, temperatures, C_bar, tau_theta, A_bar, tau_xi, D_xi, D_theta):
    """
    Save tabular data
    :param file_name: e.g. "input/simple_3d_rve_combo.h5"
    :param data_path:
    :param temperatures:
    :param C_bar: tabular data for C_bar with shape (strain_dof, strain_dof, n_temp)
    :param tau_theta: tabular data for tau_theta with shape (strain_dof, n_temp)
    :param A_bar: tabular data for A_bar with shape (strain_dof, strain_dof, n_temp)
    :param tau_xi: tabular data for tau_xi with shape (n_modes, n_temp)
    :param D_xi: tabular data for D_xi with shape (n_modes, n_modes, n_temp)
    :param D_xi: tabular data for D_theta with shape (n_temp)
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
