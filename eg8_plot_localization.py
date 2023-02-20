#%%
from operator import itemgetter

import numpy.linalg as la
import matplotlib.pyplot as plt
from interpolate_fluctuation_modes import interpolate_fluctuation_modes
from microstructures import *
from optimize_alpha import opt1, opt2, opt4, naive
from utilities import read_h5, construct_stress_localization, volume_average, compute_residual_efficient

np.random.seed(0)
file_name, data_path, temp1, temp2, n_tests, sampling_alphas = itemgetter('file_name', 'data_path', 'temp1', 'temp2', 'n_tests',
                                                                          'sampling_alphas')(microstructures[7])
print(file_name, '\t', data_path)

test_temperatures = np.linspace(temp1, temp2, num=n_tests)
test_alphas = np.linspace(0, 1, num=n_tests)

mesh, ref = read_h5(file_name, data_path, test_temperatures)
mat_id = mesh['mat_id']
n_gauss = mesh['n_gauss']
strain_dof = mesh['strain_dof']
global_gradient = mesh['global_gradient']
n_gp = mesh['n_integration_points']
n_modes = ref[0]['strain_localization'].shape[-1]

temp0 = ref[0]['temperature']
E0 = ref[0]['strain_localization']
C0 = ref[0]['mat_stiffness']
eps0 = ref[0]['mat_thermal_strain']
S0 = construct_stress_localization(E0, C0, eps0, mat_id, n_gauss, strain_dof)

alpha = sampling_alphas[1][1]
a = int(alpha * n_tests)
tempa = ref[a]['temperature']
Ea = ref[a]['strain_localization']
Ca = ref[a]['mat_stiffness']
epsa = ref[a]['mat_thermal_strain']
Sa = construct_stress_localization(Ea, Ca, epsa, mat_id, n_gauss, strain_dof)

temp1 = ref[-1]['temperature']
E1 = ref[-1]['strain_localization']
C1 = ref[-1]['mat_stiffness']
eps1 = ref[-1]['mat_thermal_strain']
S1 = construct_stress_localization(E1, C1, eps1, mat_id, n_gauss, strain_dof)

# %%


def plot_localization(ax, mesh, E, i=0, j=0):
    discr = mesh['combo_discretisation']
    n_gauss = mesh['n_gauss']
    assert E.ndim == 3
    assert E.shape[0] == n_gauss * np.prod(discr)
    assert E.shape[1] == 6
    assert E.shape[2] == 7
    E_r = E.reshape(*discr, n_gauss, 6, 7)  # resize
    E_ra = np.mean(E_r, axis=3)  # average over gauss points
    E_rai = np.linalg.norm(E_ra[:, :, :, i, :], axis=-1)  # compute norm of i-th row
    ax.imshow(E_rai[0, :, :], interpolation='spline16')  # plot y-z-cross section at x=0


fig, ax = plt.subplots(1, 3)
plot_localization(ax[0], mesh, E0, i=0)
ax[0].set_title(r'$\underline{\underline{E}}\;\mathrm{at}\;\theta=' +
                f'{temp0:.2f}' + r'\mathrm{K}$')
plot_localization(ax[1], mesh, Ea, i=0)
ax[1].set_title(r'$\underline{\underline{E}}\;\mathrm{at}\;\theta=' +
                f'{tempa:.2f}' + r'\mathrm{K}$')
#plot_localization(ax[1], mesh, 0.5*(E0+E1), i=0)  # compare with naive interpolation
plot_localization(ax[2], mesh, E1, i=0)
ax[2].set_title(r'$\underline{\underline{E}}\;\mathrm{at}\;\theta=' +
                f'{temp1:.2f}' + r'\mathrm{K}$')
plt.savefig('E.pgf', dpi=300)
plt.show()

fig, ax = plt.subplots(1, 3)
plot_localization(ax[0], mesh, S0, i=0)
ax[0].set_title(r'$\underline{\underline{S}}\;\mathrm{at}\;\theta=' +
                f'{temp0:.2f}' + r'\mathrm{K}$')
plot_localization(ax[1], mesh, Sa, i=0)
ax[1].set_title(r'$\underline{\underline{S}}\;\mathrm{at}\;\theta=' +
                f'{tempa:.2f}' + r'\mathrm{K}$')
#plot_localization(ax[2], mesh, 0.5*(S0+S1), i=0)  # compare with naive interpolation
plot_localization(ax[2], mesh, S1, i=0)
ax[2].set_title(r'$\underline{\underline{S}}\;\mathrm{at}\;\theta=' +
                f'{temp1:.2f}' + r'\mathrm{K}$')
plt.savefig('S.pgf', dpi=300)
plt.show()
