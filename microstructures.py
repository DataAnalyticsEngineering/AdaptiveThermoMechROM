from pathlib import Path as path

import numpy as np

microstructures = [
{
    'data_path': '/ms_9p/dset0_sim',
    'file_name': path("input/rve_thermoplastic_6loadings_10samples.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 10,
    'sampling_alphas': None
},
{
    'data_path': '/ms_9p/dset0_sim',
    'file_name': path("input/simple_3d_rve_B1-B6_16x16x16_100samples_fix.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 100,
    'sampling_alphas': None
},
]
