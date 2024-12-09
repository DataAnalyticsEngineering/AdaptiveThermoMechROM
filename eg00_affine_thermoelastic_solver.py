"""
Load all microstructures and try to reconstruct stress and strain fields and effective properties
"""

from operator import itemgetter

from microstructures import *
from utilities import verify_data, read_h5

for microstructure in microstructures[-9:]:

    file_name, data_path, temp1, temp2 = itemgetter('file_name', 'data_path', 'temp1', 'temp2')(microstructure)

    print(file_name, '\t', data_path)

    temperatures = np.linspace(temp1, temp2, 2)

    mesh, samples = read_h5(file_name, data_path, temperatures)

    # print(mesh.keys())
    # print(samples[0].keys())
    print('strain localication shape:', samples[0]['strain_localization'].shape)
    print('material stiffness shape:', samples[0]['mat_stiffness'].shape)
    print('plastic modes shape:', samples[0]['plastic_modes'].shape)

    for sample in samples:
        verify_data(mesh, sample)

print(f'{"done":-^50}')
