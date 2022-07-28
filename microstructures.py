import numpy as np
from pathlib import Path as path

microstructures = [{
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/striped_normal_4x4x4.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 100,
    'sampling_alphas': None
}, {
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/sphere_normal_16x16x16_10samples.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 10,
    'sampling_alphas': None
}, {
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/sphere_normal_32x32x32_10samples.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 10,
    'sampling_alphas': None
}, {
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/sphere_combo_16x16x16_10samples.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 10,
    'sampling_alphas': None
}, {
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/octahedron_normal_16x16x16_10samples.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 10,
    'sampling_alphas': None
}, {
    'data_path': '/ms_1p/dset0_sim',
    'file_name': path("input/octahedron_combo_16x16x16_10samples.h5"),
    'temp1': 300,
    'temp2': 1300,
    'n_tests': 10,
    'sampling_alphas': None
}, {
    'data_path':
    '/ms_1p/dset0_sim',
    'file_name':
    path("input/octahedron_combo_32x32x32.h5"),
    'temp1':
    300,
    'temp2':
    1300,
    'n_tests':
    100,
    'sampling_alphas':
    np.asarray([
        np.asarray([0., 1.]),
        np.asarray([0., 0.82828283, 1.]),
        np.asarray([0., 0.82828283, 0.93939394, 1.]),
        np.asarray([0., 0.60606061, 0.82828283, 0.93939394, 1.]),
        np.asarray([0., 0.60606061, 0.82828283, 0.93939394, 0.97979798, 1.])
    ], dtype=object)
}, {
    'data_path':
    '/image_data/dset_0_sim',
    'file_name':
    path("input/random_rve_vol20.h5"),
    'temp1':
    300,
    'temp2':
    1300,
    'n_tests':
    100,
    'sampling_alphas':
    np.asarray([
        np.asarray([0., 1.]),
        np.asarray([0., 0.85858586, 1.]),
        np.asarray([0., 0.85858586, 0.94949495, 1.]),
        np.asarray([0., 0.66666667, 0.85858586, 0.94949495, 1.]),
        np.asarray([0., 0.66666667, 0.85858586, 0.94949495, 0.97979798, 1.])
    ], dtype=object)
}, {
    'data_path':
    '/image_data/dset_0_sim',
    'file_name':
    path("input/random_rve_vol40.h5"),
    'temp1':
    300,
    'temp2':
    1300,
    'n_tests':
    100,
    'sampling_alphas':
    np.asarray([
        np.asarray([0., 1.]),
        np.asarray([0., 0.8989899, 1.]),
        np.asarray([0., 0.8989899, 0.96969697, 1.]),
        np.asarray([0., 0.71717172, 0.8989899, 0.96969697, 1.]),
        np.asarray([0., 0.51515152, 0.71717172, 0.8989899, 0.96969697, 1.])
    ], dtype=object)
}, {
    'data_path':
    '/image_data/dset_0_sim',
    'file_name':
    path('input/random_rve_vol60.h5'),
    'temp1':
    300,
    'temp2':
    1300,
    'n_tests':
    100,
    'sampling_alphas':
    np.asarray([
        np.asarray([0., 1.]),
        np.asarray([0., 0.8989899, 1.]),
        np.asarray([0., 0.72727273, 0.8989899, 1.]),
        np.asarray([0., 0.72727273, 0.8989899, 0.96969697, 1.]),
        np.asarray([0., 0.52525253, 0.72727273, 0.8989899, 0.96969697, 1.])
    ], dtype=object)
}]
