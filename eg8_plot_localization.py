"""
Plot the strain localization operator E and stress localization operator S at different temperatures
"""
#%%
from operator import itemgetter

import numpy.linalg as la
import matplotlib.pyplot as plt
from microstructures import *
from utilities import read_h5, construct_stress_localization

################



    stress_localization = np.empty_like(strain_localization)
    strain_localization_transp = ...
    I = np.eye(strain_dof)
    for gp_id in prange(strain_localization.shape[0]):
        phase_id = mat_id[gp_id // n_gauss]
        stress_localization[gp_id] = strain_localization_transp[gp_id] @ mat_stiffness[phase_id] @ (plastic_modes - eigen_strains)
    A = volume_average(stress_localization)
    
    D0 = volume_average(inner_product((plastic_modes - eigen_strains), eigen_strains))

    K0 = -volume_average(inner_product(plastic_modes, K @ eigen_strains))
    K = k * K0

    D = D0 + K
    
    R = volume_average(thermal_stresses @ (plastic_modes - eigen_strains)) / delta_theta


###########

np.random.seed(0)
file_name, data_path, temp1, temp2, n_tests, sampling_alphas = itemgetter(
    "file_name", "data_path", "temp1", "temp2", "n_tests", "sampling_alphas"
)(microstructures[7])
print(file_name, "\t", data_path)

test_temperatures = np.linspace(temp1, temp2, num=n_tests)
test_alphas = np.linspace(0, 1, num=n_tests)

mesh, ref = read_h5(file_name, data_path, test_temperatures)
mat_id = mesh["mat_id"]
n_gauss = mesh["n_gauss"]
strain_dof = mesh["strain_dof"]
nodal_dof = mesh["nodal_dof"]
n_elements = mesh["n_elements"]
global_gradient = mesh["global_gradient"]
n_gp = mesh["n_integration_points"]
n_modes = ref[0]["strain_localization"].shape[-1]
disc = mesh["combo_discretisation"]

# Lowest temperature
temp0 = ref[0]["temperature"]
E0 = ref[0]["strain_localization"]
C0 = ref[0]["mat_stiffness"]
eps0 = ref[0]["mat_thermal_strain"]
S0 = construct_stress_localization(E0, C0, eps0, mat_id, n_gauss, strain_dof)

# First enrichment temperature
a = len(ref) // 2
if sampling_alphas is not None:
    alpha = sampling_alphas[1][1]
    a = int(alpha * n_tests)
tempa = ref[a]["temperature"]
Ea = ref[a]["strain_localization"]
Ca = ref[a]["mat_stiffness"]
epsa = ref[a]["mat_thermal_strain"]
Sa = construct_stress_localization(Ea, Ca, epsa, mat_id, n_gauss, strain_dof)

# Highest temperature
temp1 = ref[-1]["temperature"]
E1 = ref[-1]["strain_localization"]
C1 = ref[-1]["mat_stiffness"]
eps1 = ref[-1]["mat_thermal_strain"]
S1 = construct_stress_localization(E1, C1, eps1, mat_id, n_gauss, strain_dof)

# %%


def plot_localization(ax, E, idx=0):
    """Plots the effective total strain/stress norm (not the deviatoric part)
    of a given localization operator `E` on the y-z-cross section at x=idx.

    Args:
        ax: matplotlib axis
        E (np.ndarray): localization operator with shape (nx*ny*nz*ngauss, 6, 7)
        idx (int, optional): y-z-cross section index. Defaults to 0.
    """
    assert E.ndim == nodal_dof
    assert E.shape[0] == n_gp
    assert E.shape[1] == strain_dof
    # assert E.shape[2] == n_modes
    E_r = E.reshape(*disc, n_gauss, strain_dof, n_modes)
    E_ra = np.mean(E_r, axis=3)  # average over gauss points
    # compute the effective total strain norm (not the deviatoric part);
    # account for Mandel notation, i.e. activation strain with all components being 1.0
    E_rai = np.einsum('ijklm,m', E_ra, np.asarray([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2), 1]))
    effective_strain = np.sqrt(2/3) * la.norm(E_rai, axis=-1)
    # plot y-z-cross section at x=idx
    ax.imshow(effective_strain[idx, :, :], interpolation="gaussian")


# Plot strain localization operator E at different temperatures
fig, ax = plt.subplots(1, 3)
plot_localization(ax[0], E0, idx=0)
ax[0].set_title(
    r"$\underline{\underline{E}}\;\mathrm{at}\;\theta="
    + f"{temp0:.2f}"
    + r"\mathrm{K}$"
)
plot_localization(ax[1], Ea, idx=0)
ax[1].set_title(
    r"$\underline{\underline{E}}\;\mathrm{at}\;\theta="
    + f"{tempa:.2f}"
    + r"\mathrm{K}$"
)
plot_localization(ax[2], E1, idx=0)
ax[2].set_title(
    r"$\underline{\underline{E}}\;\mathrm{at}\;\theta="
    + f"{temp1:.2f}"
    + r"\mathrm{K}$"
)
plt.savefig("output/E.png", dpi=300)
plt.show()

# Plot stress localization operator S at different temperatures
fig, ax = plt.subplots(1, 3)
plot_localization(ax[0], S0, idx=0)
ax[0].set_title(
    r"$\underline{\underline{S}}\;\mathrm{at}\;\theta="
    + f"{temp0:.2f}"
    + r"\mathrm{K}$"
)
plot_localization(ax[1], Sa, idx=0)
ax[1].set_title(
    r"$\underline{\underline{S}}\;\mathrm{at}\;\theta="
    + f"{tempa:.2f}"
    + r"\mathrm{K}$"
)
plot_localization(ax[2], S1, idx=0)
ax[2].set_title(
    r"$\underline{\underline{S}}\;\mathrm{at}\;\theta="
    + f"{temp1:.2f}"
    + r"\mathrm{K}$"
)
plt.savefig("output/S.png", dpi=300)
plt.show()

# %%
