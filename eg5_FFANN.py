# %%
# # JEL_thermo-el-ROM: Machine Learned Model
# ### Imports:
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from operator import itemgetter
from microstructures import *
from utilities import read_h5
from utilities_ann import cholesky, reverse_cholesky, bisection_sampling, hierarchical_sampling, RectFFModule, model_training, \
    plot_training_history, mech_loss

torch.set_default_dtype(torch.float32)
# ### Load DNS data from HDF5 file:
for ms_id in [7, 8, 9]:
    file_name, data_path, temp1, temp2, n_tests, sampling_alphas = itemgetter('file_name', 'data_path', 'temp1', 'temp2',
                                                                              'n_tests',
                                                                              'sampling_alphas')(microstructures[ms_id])
    # debuging options
    # sampling_alphas = [sampling_alphas[-1]]

    print(file_name, '\t', data_path)
    out_file = path(f'output/ann_{file_name.name}')
    test_temperatures = np.linspace(temp1, temp2, num=n_tests)
    test_alphas = np.linspace(0, 1, num=n_tests)
    _, refs = read_h5(file_name, data_path, test_temperatures, get_mesh=False)
    x = test_temperatures[:, None]
    y = np.stack([np.hstack((cholesky(ref['eff_stiffness']), ref['eff_thermal_strain'])) for ref in refs])

    # ### Data scaling:
    # As sole input feature the temperature $\theta$ is used.
    # The output features $y = [\mathrm{vec}(L), E] \in \mathbb{R}^{27}$ are given by the Cholesky decomposition $L$ of the Mandel notation $C = L L^\intercal$ of the fourth-order effective stiffness tensor $\mathbb{C}$ and the Mandel notation $E \in \mathbb{R}^6$ of the thermal strain tensor $\varepsilon_\theta$ that are obtained by the DNS.
    # Since the different components of $y$ differ greatly in magnitude, each output feature is normalized with its absolute maximum value.
    # Scaling
    x = torch.FloatTensor(x / np.max(x, axis=0))
    y_normalization_factor = np.max(np.abs(y), axis=0)
    y = torch.FloatTensor(y / y_normalization_factor)
    dset = TensorDataset(x, y)
    n_out_features = y.size()[1]
    n_in_features = x.size()[1]

    file = h5py.File(out_file, 'w')
    for level, sampling_alpha in enumerate(sampling_alphas):
        # ### Data sampling:
        # #### Bisection sampling:
        # reproducibility
        torch.manual_seed(1)
        torch.use_deterministic_algorithms(True)
        # Split in test and validation data
        # train_data, val_data = bisection_sampling(dset, levels=3, samples_per_pos=1, validation=True)
        # sampling_alphas = np.asarray([0, 0.41, 0.74, 0.9, 0.95, 1])
        train_data, val_data = hierarchical_sampling(dset, x[:, 0], sampling_alpha)

        # Create dataloaders
        batch_size = len(train_data)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data))
        # Create a PyTorch model for a Feedforward Artificial Neural Network (FFANN) with 3 hidden layers and 64 neurons per layer.
        # `Tanh` is used as activation function in the hidden layers and the identity as activation function in the output layer.
        model = RectFFModule(n_in_features, 64, 3, nn.Tanh(), nn.Identity(), n_out_features)
        # print(model)

        # The MSE loss function is used for training. Using the "mechanical loss function", which is defined as
        # $$\frac{||L_{pred} - L||_F}{||L||_F} + \frac{||\varepsilon_{\theta,pred} - \varepsilon_{\theta}||_F}{||\varepsilon_\theta||_F}$$
        # Here, using the Adam as optimizer leads to faster convergence than using Stochastic Gradient Descent (SGD).
        loss_fn = nn.MSELoss(reduction='mean')
        # loss_fn = mech_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        train_losses, val_losses, best_epoch = model_training(model, loss_fn, optimizer, train_loader, val_loader,
                                                              epochs=4000 * len(train_data), verbose=False)

        # The training history of the ANN is plotted in the figure below.
        fig, ax = plt.subplots()
        plot_training_history(ax, train_losses, val_losses, best_epoch)
        y_pred = model(x)

        # The predictions of the ANN are compared to the ground truth in the figure below.
        fig, ax = plt.subplots(1, 1, figsize=[4, 4])
        colors = colormap.rainbow(np.linspace(0, 1, n_out_features))
        for i in range(n_out_features):
            ax.plot(test_temperatures, y[:, i], '-', color=colors[i], lw=1)
            ax.plot(test_temperatures, y_pred[:, i].detach().numpy(), '--', color=colors[i], lw=2)
            ax.set_title('normalized features')
            ax.set_xlabel(r'Temperature $\theta$ [K]')
            ax.set_ylabel(r'Scaled output feature [-]')
        plt.grid('on')
        plt.tight_layout()
        plt.show(block=False)

        # Plot of the error
        norm_error = np.linalg.norm(y - y_pred.detach(), axis=1) / np.linalg.norm(y, axis=1) * 100
        fig, ax = plt.subplots(1, 1, figsize=[4, 4])
        ax.plot(test_temperatures, norm_error, 'b-', label='0 levels')
        ax.set_xlabel(r'Temperature [K]')
        ax.set_ylabel(r'Relative error [\%]')
        plt.grid('on')
        plt.tight_layout()
        plt.show(block=False)
        print(f'{np.argmax(norm_error)/100 = :.2f}')
        print(f'{len(train_data)} samples')
        print(f'{len(val_data)} validations')
        print(f'{np.max(norm_error) = :.2f} %')

        # %%
        y_approx = y_pred.detach().numpy() * y_normalization_factor
        errC = np.zeros(n_tests)
        erreps = np.zeros(n_tests)

        group = file.require_group(f'{data_path}_level{level}')
        # group.attrs['sampling_strategy'] = "model description"
        for i, ref in enumerate(refs):
            dset_stiffness = group.require_dataset(f'eff_stiffness_{test_temperatures[i]:07.2f}', (6, 6), dtype='f')
            dset_thermal_strain = group.require_dataset(f'eff_thermal_strain_{test_temperatures[i]:07.2f}', (6), dtype='f')

            C = reverse_cholesky(y_approx[i, :21])
            eps = y_approx[i, 21:]

            dset_stiffness[:] = C.T
            dset_thermal_strain[:] = eps

            Cref = np.asarray(ref['eff_stiffness'], dtype=float)
            epsref = np.asarray(ref['eff_thermal_strain'], dtype=float)
            invL = np.linalg.inv(np.linalg.cholesky(Cref))
            errC[i] = np.linalg.norm(invL @ C @ invL.T - np.eye(6)) / np.linalg.norm(np.eye(6)) * 100
            erreps[i] = np.linalg.norm(epsref - eps) / np.linalg.norm(epsref) * 100
            # erreps[i] = (epsref @ Cref @ epsref - eps @ Cref @ eps) / (epsref @ Cref @ epsref) * 100

        ylabels = [
            'Relative error $e_{\overline{\mathbb{C}}}$ [\%]',
            r'Relative error $e_{\overline{\boldmath{\varepsilon}}_{\uptheta}}$ [\%]'
        ]
        for idx, err in enumerate([errC, erreps]):
            fig, ax = plt.subplots(1, 1, figsize=[4, 4])
            ax.plot(test_temperatures, err, 'b-')
            ax.set_xlabel(r'Temperature [K]')
            ax.set_ylabel(ylabels[idx])
            plt.grid('on')
            plt.tight_layout()
            plt.show(block=False)
            print(f'{np.max(err) = :.2f} %')
        plt.close('all')
    file.close()
