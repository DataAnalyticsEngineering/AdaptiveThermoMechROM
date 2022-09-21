import numpy as np
from microstructures import *
from operator import itemgetter
from utilities import verify_data, read_h5

def test_read_simple_microstructure():
    """
    Load a microstructure and try to reconstruct stress and strain fields and effective properties
    """    

    file_name, data_path, temp1, temp2 = itemgetter('file_name', 'data_path', 'temp1', 'temp2')(microstructures[0])

    print(file_name, '\t', data_path)

    temperatures = np.linspace(temp1, temp2, 2)

    mesh, samples = read_h5(file_name, data_path, temperatures)

    for sample in samples:
        verify_data(mesh, sample)

    print(f'{"done":-^50}')
    assert True
