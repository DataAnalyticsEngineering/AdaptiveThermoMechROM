"""
Temperature dependent material parameters of Copper(Cu) and Fused Tungsten Carbide (FTC)
The units here differ from the paper cited in readme.md by converting meter to millimeter
The example after `if __name__ == "__main__"` uses meter again and it matches the paper
"""

import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
from utilities import cm

I2 = np.asarray([1., 1., 1., 0, 0, 0])
I4 = np.eye(6)
IxI = np.outer(I2, I2)
P1 = IxI / 3.0
P2 = I4 - P1

min_temperature = 293.00
max_temperature = 1300

poisson_ratio_cu = lambda x: 3.40000e-01 * x**0
conductivity_cu = lambda x: 4.20749e+05 * x**0 + -6.84915e+01 * x**1
heat_capacity_cu = lambda x: 2.94929e+03 * x**0 + 2.30217e+00 * x**1 + -2.95302e-03 * x**2 + 1.47057e-06 * x**3
cte_cu = lambda x: 1.28170e-05 * x**0 + 8.23091e-09 * x**1
elastic_modulus_cu = lambda x: 1.35742e+08 * x**0 + 5.85757e+03 * x**1 + -8.16134e+01 * x**2

thermal_strain_cu = lambda x: integrate.quad(cte_cu, min_temperature, x)[0] * I2
shear_modulus_cu = lambda x: elastic_modulus_cu(x) / (2. * (1. + poisson_ratio_cu(x)))
bulk_modulus_cu = lambda x: elastic_modulus_cu(x) / (3. * (1. - 2. * poisson_ratio_cu(x)))
stiffness_cu = lambda x: bulk_modulus_cu(x) * IxI + 2. * shear_modulus_cu(x) * P2

poisson_ratio_wsc = lambda x: 2.80000e-01 * x**0
conductivity_wsc = lambda x: 2.19308e+05 * x**0 + -1.87425e+02 * x**1 + 1.05157e-01 * x**2 + -2.01180e-05 * x**3
heat_capacity_wsc = lambda x: 2.39247e+03 * x**0 + 6.62775e-01 * x**1 + -2.80323e-04 * x**2 + 6.39511e-08 * x**3
cte_wsc = lambda x: 5.07893e-06 * x**0 + 5.67524e-10 * x**1
elastic_modulus_wsc = lambda x: 4.13295e+08 * x**0 + -7.83159e+03 * x**1 + -3.65909e+01 * x**2 + 5.48782e-03 * x**3

thermal_strain_wsc = lambda x: integrate.quad(cte_wsc, min_temperature, x)[0] * I2
shear_modulus_wsc = lambda x: elastic_modulus_wsc(x) / (2. * (1. + poisson_ratio_wsc(x)))
bulk_modulus_wsc = lambda x: elastic_modulus_wsc(x) / (3. * (1. - 2. * poisson_ratio_wsc(x)))
stiffness_wsc = lambda x: bulk_modulus_wsc(x) * IxI + 2. * shear_modulus_wsc(x) * P2

if __name__ == "__main__":
    temp1 = 300
    temp2 = 1300
    n_tests = 25
    test_temperatures = np.linspace(temp1, temp2, num=n_tests)

    temp = temp1
    print(f'phase contrast at {temp:6}K: {elastic_modulus_wsc(temp) / elastic_modulus_cu(temp):.2f}')
    temp = temp2
    print(f'phase contrast at {temp:6}K: {elastic_modulus_wsc(temp) / elastic_modulus_cu(temp):.2f}')

    parameters = np.zeros((n_tests, 8))
    for idx, temperature in enumerate(test_temperatures):
        # print(f'{temperature = :.2f}')

        parameters[idx, 0] = elastic_modulus_cu(temperature) / 1e6
        parameters[idx, 1] = elastic_modulus_wsc(temperature) / 1e6

        parameters[idx, 2] = thermal_strain_cu(temperature)[0]
        parameters[idx, 3] = thermal_strain_wsc(temperature)[0]

        parameters[idx, 4] = conductivity_cu(temperature) / 1e3
        parameters[idx, 5] = conductivity_wsc(temperature) / 1e3

        parameters[idx, 6] = heat_capacity_cu(temperature) * 1e3
        parameters[idx, 7] = heat_capacity_wsc(temperature) * 1e3

    labels = [
        r'E\textsuperscript{Cu}', r'E\textsuperscript{FTC}', r'${\varepsilon}_{\uptheta}^{\text{Cu}}$',
        r'${\varepsilon}_{\uptheta}^{\text{FTC}}$', r'$\kappa$\textsuperscript{Cu}', r'$\kappa$\textsuperscript{FTC}',
        r'c\textsuperscript{Cu}', r'c\textsuperscript{FTC}'
    ]
    markers = ['s', 'd', '+', 'x', 'o']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    fig_name = 'eg1_mat_parameters1'
    xlabel = 'Temperature [K]'
    ylabel = 'Elastic modulus [GPa]'
    plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
    plt.plot(test_temperatures, parameters[:, 0], label=labels[0], marker=markers[0], color=colors[0], markevery=3)
    plt.plot(test_temperatures, parameters[:, 1], label=labels[1], marker=markers[1], color=colors[1], markevery=3)
    gca1 = plt.gca()
    gca2 = gca1.twinx()
    gca2.plot(test_temperatures, parameters[:, 2], label=labels[2], marker=markers[2], color=colors[2], markevery=3)
    gca2.plot(test_temperatures, parameters[:, 3], label=labels[3], marker=markers[3], color=colors[3], markevery=3)
    gca1.set_xlim([temp1, temp2])
    gca1.set_xlabel(rf'{xlabel}')
    gca1.set_ylabel(rf'{ylabel}')
    gca1.grid(ls='--', color='gray', linewidth=0.5)
    ylabel = 'Thermal dilation [-]'
    gca2.set_ylabel(rf'{ylabel}')
    gca2.ticklabel_format(axis='y', scilimits=[0, 2])
    gca1.legend(loc='upper left', facecolor=(0.9, 0.9, 0.9, 0.6), edgecolor='black')
    gca2.legend(loc='center right', facecolor=(0.9, 0.9, 0.9, 0.6), edgecolor='black')
    plt.tight_layout(pad=0.025)
    plt.savefig(f'output/{fig_name}.png')
    plt.show()

    fig_name = 'eg1_mat_parameters2'
    xlabel = 'Temperature [K]'
    ylabel = 'Conductivity [W/(m K)]'
    plt.figure(figsize=(6 * cm, 6 * cm), dpi=600)
    plt.plot(test_temperatures, parameters[:, 0 + 4], label=labels[0 + 4], marker=markers[0], color=colors[0], markevery=3)
    plt.plot(test_temperatures, parameters[:, 1 + 4], label=labels[1 + 4], marker=markers[1], color=colors[1], markevery=3)
    gca1 = plt.gca()
    gca2 = gca1.twinx()
    gca2.plot(test_temperatures, parameters[:, 2 + 4], label=labels[2 + 4], marker=markers[2], color=colors[2], markevery=3)
    gca2.plot(test_temperatures, parameters[:, 3 + 4], label=labels[3 + 4], marker=markers[3], color=colors[3], markevery=3)
    gca1.set_xlim([temp1, temp2])
    gca1.set_xlabel(rf'{xlabel}')
    gca1.set_ylabel(rf'{ylabel}')
    gca1.grid(ls='--', color='gray', linewidth=0.5)
    ylabel = r'Heat capacity [J/(m$^3$ K)]'
    gca2.set_ylabel(rf'{ylabel}')
    gca1.legend(loc='upper left', facecolor=(0.9, 0.9, 0.9, 0.6), edgecolor='black')
    gca2.legend(loc='center right', facecolor=(0.9, 0.9, 0.9, 0.6), edgecolor='black')
    plt.tight_layout(pad=0.025)
    plt.savefig(f'output/{fig_name}.png')
    plt.show()
