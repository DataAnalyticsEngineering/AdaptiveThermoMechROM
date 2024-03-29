<!--
https://docs.github.com/en/actions/quickstart
https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python
 -->
[![Build](https://github.com/DataAnalyticsEngineering/AdaptiveThermoMechROM/actions/workflows/github-actions.yml/badge.svg?branch=main)](https://github.com/DataAnalyticsEngineering/AdaptiveThermoMechROM/actions/workflows/github-actions.yml)

# [AdaptiveThermoMechROM](https://github.com/DataAnalyticsEngineering/AdaptiveThermoMechROM)

An adaptive approach for strongly temperature-dependent thermoelastic homogenization. Using direct numerical simulations at few
discrete temperatures, an energy optimal basis is constructed to be used at any intermediate temperature in real-time.

Not only the effective behavior, i.e. the effective stiffness and the effective thermal expansion, of the microscopic reference
volume element (RVE) are predicted but also accurate full-field reconstructions of all mechanical fields on the RVE scale.

We show that the proposed method referred to as optimal field interpolation is on par with linear interpolation in terms of its
numerical cost but with an accuracy that matches DNS in many cases, i.e. very accurate real-time predictions are anticipated with
minimal DNS inputs that range from two to six simulations. Further, we pick up black box machine learning models as an alternative
route and show their limitations in view of both accuracy and the amount of required training data.

## Requirements

- Python 3.9 or later
- Input
  dataset: [![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--2822-d45815.svg)](https://doi.org/10.18419/darus-2822)

All necessary data can be downloaded from [DaRUS](https://darus.uni-stuttgart.de/) using the script [`download_data.sh`](download_data.sh).

## How to use?

The provided code is independent of direct numerical simulators, i.e. it expects DNS results to be stored in a HDF5 file with a
structure that follows [`input/h5_data_structure.pdf`](input/h5_data_structure.pdf). It is assumed that DNSs are coming from voxel-based thermomechanical solvers
with a voxel following the node numbering as in [`input/vtk_node_numbering.png`](input/vtk_node_numbering.png) (VTK_VOXEL=11).

Note that extensions to directly calling a direct numerical simulator or use a different element type require a slight
modification from interested users. This was not already included here to ensure having a standalone code that is able to
reproduce results from the publication cited below.

For details about the setup of the following examples, please refer to the cited publication.

- [`eg0_affine_thermoelastic_solver.py`](eg0_affine_thermoelastic_solver.py):
  This module goes throw all microstructure files, given in [`microstructures.py`](microstructures.py) then it tries to load all available data for all
  given temperatures and does some sanity checks to ensure that the provided results are consistent.

- [`eg1_approximation_of_mat_properties.py`](eg1_approximation_of_mat_properties.py):
  Approximate copper and tungsten temperature-dependent material properties given in [`material_parameters.py`](material_parameters.py) using various
  approaches.

- [`eg2_compare_approximations.py`](eg2_compare_approximations.py):
  Given DNSs at only two temperatures, compare the different interpolation schemes.

- [`eg3_hierarchical_sampling.py`](eg3_hierarchical_sampling.py):
  Build a hierarchy of samples such that approximation error is reduced.

- [`eg4_hierarchical_sampling_efficient.py`](eg4_hierarchical_sampling_efficient.py):
  Same as [`eg3_hierarchical_sampling.py`](eg3_hierarchical_sampling.py) but more efficient due to exploitation of affine structure of the proposed interpolation
  scheme.

- [`eg5_FFANN.py`](eg5_FFANN.py):
  Train feed-forward artificial neural networks to approximate the homogenized response with different number of samples.

- [`eg6_post_process_ann_vs_proposed.py`](eg6_post_process_ann_vs_proposed.py):
  Compare the results using the trained ANNs with our proposed interpolation scheme.

- [`eg7_staggered_model_interpolation.py`](eg7_staggered_model_interpolation.py):
  Interpolate the homogenized response at arbitrary temperatures based on the approximations in [`eg4_hierarchical_sampling_efficient.py`](eg4_hierarchical_sampling_efficient.py)

<!-- https://mybinder.readthedocs.io/en/latest/using/config_files.html -->
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DataAnalyticsEngineering/AdaptiveThermoMechROM/HEAD)

## Manuscript

["Reduced order homogenization of thermoelastic materials with strong temperature-dependence and comparison to a machine-learned model"](https://doi.org/10.1007/s00419-023-02411-6)

by Shadi Sharba, Julius Herb and Felix Fritzen. Published in *Archive of Applied Mechanics*, DOI: [10.1007/s00419-023-02411-6](https://doi.org/10.1007/s00419-023-02411-6).

Affiliation: [Data Analytics in Engineering, University of Stuttgart](http://www.mib.uni-stuttgart.de/dae)

## Acknowledgments

- The IGF-Project with the IGF-No.: 21079 N / DVS-No.: 06.3341 of the “Forschungsvereinigung Schweißen und verwandte Verfahren e.
  V.” of the German Welding Society (DVS), Aachener Str. 172, 40223 Düsseldorf was funded by the Federal Ministry for Economic
  Affairs and Climate Action (BMWK) via the German Federation of Industrial Research Associations (AiF) in accordance with the
  policy to support the Industrial Collective Research (IGF) on the basis of a decision by the German Bundestag. Furthermore, the
  authors gratefully acknowledge the collaboration with the members of the project affiliated committee regarding the support of
  knowledge, material and equipment over the course of the research.

- Contributions by Felix Fritzen are partially funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under
  Germany’s Excellence Strategy - EXC 2075 – 390740016. Felix Fritzen is funded by Deutsche Forschungsgemeinschaft (DFG, German
  Research Foundation) within the Heisenberg program DFG-FR2702/8 - 406068690 and DFG-FR2702/10 - 517847245.

- The authors acknowledge the support by the Stuttgart Center for Simulation Science (SimTech).
