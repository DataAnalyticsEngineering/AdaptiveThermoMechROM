import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import copy

class BaseModule(nn.Module):
    """Represents a `Base Module` that contains the basic functionality of an artificial neural network (ANN).

    All modules should inherit from :class:`mlprum.models.BaseModule` and override :meth:`mlprum.models.BaseModule.forward`.
    The Base Module itself inherits from :class:`torch.nn.Module`. See the `PyTorch` documentation for further information.
    """
    def __init__(self):
        """Constructor of the class. Initialize the Base Module.

        Should be called at the beginning of the constructor of a subclass.
        The class :class:`mlprum.models.BaseModule` should not be instantiated itself, but only its subclasses.
        """
        super().__init__()
        self.device = 'cpu'

    def forward(*args):
        """Forward propagation of the ANN. Subclasses must override this method.

        :raises NotImplementedError: If the method is not overriden by a subclass.
        """
        raise NotImplementedError('subclasses must override forward()!')

    def training_step(self, dataloader, loss_fn, optimizer):
        """Single training step that performs the forward propagation of an entire batch,
        the training loss calculation and a subsequent optimization step.

        A training epoch must contain one call to this method.

        Example:
            >>> train_loss = module.training_step(train_loader, loss_fn, optimizer)

        :param dataloader: Dataloader with training data
        :type dataloader: :class:`torch.utils.data.Dataloader`
        :param loss_fn: Loss function for the model training
        :type loss_fn: method
        :param optimizer: Optimizer for model training
        :type optimizer: :class:`torch.optim.Optimizer`
        :return: Training loss
        :rtype: float
        """
        self.train()  # enable training mode
        cumulative_loss = 0
        samples = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            # Loss calculation
            y_pred = self(x)
            loss = loss_fn(y_pred, y)
            cumulative_loss += loss.item() * x.size(0)
            samples += x.size(0)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = cumulative_loss / samples
        return average_loss

    def loss_calculation(self, dataloader, loss_fns):
        """Perform the forward propagation of an entire batch from a given `dataloader`
        and the subsequent loss calculation for one or multiple loss functions in `loss_fns`.

        Example:
            >>> val_loss = module.loss_calculation(val_loader, loss_fn)

        :param dataloader: Dataloader with validation data
        :type dataloader: :class:`torch.utils.data.Dataloader`
        :param loss_fn: Loss function for model training
        :type loss_fn: method or list of methods
        :return: Validation loss
        :rtype: float
        """
        self.eval()  # disable training mode
        if not isinstance(loss_fns, list):
            loss_fns = [loss_fns]
        cumulative_loss = torch.zeros(len(loss_fns))
        samples = 0
        with torch.no_grad():  # disable gradient calculation
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self(x)
                samples += x.size(0)
                for i, loss_fn in enumerate(loss_fns):
                    loss = loss_fn(y_pred, y)
                    cumulative_loss[i] += loss.item() * x.size(0)
        average_loss = cumulative_loss / samples
        if torch.numel(average_loss) == 1:
            average_loss = average_loss[0]
        return average_loss

    def parameter_count(self):
        """Get the number of learnable parameters, that are contained in the model.

        :return: Number of learnable parameters
        :rtype: int
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, device, *args, **kwargs):
        """Transfers a model to another device, e.g. to a GPU.

        This method overrides the PyTorch built-in method :code:`model.to(...)`.

        Example:
            >>> # Transfer the model to a GPU
            >>> module.to('cuda:0')

        :param device: Identifier of the device, e.g. :code:`'cpu'`, :code:`'cuda:0'`, :code:`'cuda:1'`, ...
        :type device: str
        :return: The model itself
        :rtype: :class:`mlprum.models.BaseModule`
        """
        self.device = device
        return super().to(device, *args, **kwargs)

class FFModule(BaseModule):
    """General feedforward neural network.

    It consists of an input layer with :code:`in_dim` neurons, multiple hidden layers with the
    neuron counts from the list :code:`neurons`, activation functions from the list :code:`activations`
    and an output layer with :code:`out_dim` neurons with the last element from :code:`activations` as activation function.

    Example:
        >>> # Create FF neural network with 2 hidden layers
        >>> model = FFModule(8, [50, 40], 35, [nn.SELU(), nn.SELU(), nn.Softplus()])
        >>> print(model) # print model summary
        ...
        >>> # model training based on data (x,y)...
        >>> y_pred = model(x) # model prediction

    """
    def __init__(self, in_dim, neurons, activations, out_activation, out_dim, prepare_fn=None):
        """Constructor of the class. Initialize a feedforward neural network with given neuron counts and activation functions.

        Example:
        >>> # Create FF neural network with 2 hidden layers
        >>> model = FFModule(8, [50, 40], [nn.SELU(), nn.SELU()], nn.Softplus(), 35)

        :param in_dim: Number of neurons in the input layer, i.e. number of input features
        :type in_dim: int
        :param neurons: List of neuron counts in the hidden layers
        :type neurons: list
        :param out_dim: Number of neurons in the output layer, i.e. number of output features
        :type out_dim: int
        :param activations: List of activation functions for the hidden layers and the output layer
        :type activations: list
        :param prepare_fn: Method for preprocessing that is called before the input layer, defaults to None
        :type prepare_fn: method, optional
        """
        super().__init__()
        self.neurons = [in_dim, *neurons]
        self.activations = activations
        self.hidden_layers = []
        self.prepare_fn = prepare_fn

        # Create the hidden layers
        for i, activation in enumerate(self.activations):
            self.hidden_layers.append(nn.Linear(self.neurons[i], self.neurons[i + 1]))
            self.hidden_layers.append(activation)
        self.hidden = nn.Sequential(*self.hidden_layers)

        # Create the output layer
        self.output = nn.Sequential(nn.Linear(self.neurons[-1], out_dim), out_activation)

        # Initialization for SELU activation function
        """
        for param in self.parameters():
            # biases zero
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            # others using lecun-normal initialization
            else:
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')
        """

    def forward(self, x):
        """Forward propagation of the deterministic model.

        Do not call the method `forward` itself, but call the module directly.

        Example:
            >>> y_pred = model(x) # forward propagation

        :param x: input data
        :type x: Tensor
        :return: predicted data
        :rtype: Tensor
        """
        z = self.intermediate(x)
        return self.output(z)

    def intermediate(self, x):
        if self.prepare_fn is not None:
            x = self.prepare_fn(x)
        return self.hidden(x)

class RectFFModule(FFModule):
    """General feedforward neural network with a rectangular shape for the hidden layers.

    Example:
        >>> # Create a feedforward ANN with 4 hidden layers (each 50 neurons)
        >>> model = RectFFModel(8, 50, 4, nn.SELU(), nn.Softplus(), 35) # create module
        >>> print(model) # print model summary
        ...
        >>> # model training based on data (x,y)...
        >>> y_pred = model(x) # model prediction

    """
    def __init__(self, in_dim, neuron_count, layer_count, activation, out_activation, out_dim, prepare_fn=None):
        """Constructor of the class. Initialize the neural network with given neuron count, layer count and activation functions.

        Example:
        >>> # Create feedforward ANN with 4 hidden layers (50 neurons each)
        >>> module = RectFFModel(8, 50, 4, nn.SELU(), nn.Softplus(), 35)

        :param in_dim: Number of neurons in the input layer, i.e. number of input features
        :type in_dim: int
        :param neuron_count: Number of neurons in each hidden layer
        :type neuron_count: int
        :param layer_count: Number of hidden layers
        :type layer_count: int
        :param activation: Activation function for the hidden layers
        :type activation: method
        :param out_activation: Activation function for the output layer
        :type out_activation: method
        :param out_dim: Number of neurons in the input layer, i.e. number of input features
        :type out_dim: int
        :param prepare_fn: Method for preprocessing that is called before the input layer, defaults to None
        :type prepare_fn: method, optional
        """
        neurons, activations = [], []
        for _ in range(layer_count):
            neurons.append(neuron_count)
            activations.append(copy.deepcopy(activation))
        super().__init__(in_dim, neurons, activations, out_activation, out_dim, prepare_fn)

def cholesky(array):
    return np.linalg.cholesky(array)[np.tril_indices(array.shape[0])]

def reverse_cholesky(array):
    """Reconstruct matrices from their Cholesky decompositions (https://en.wikipedia.org/wiki/Cholesky_decomposition),
    whose elements are given in flattened form by `array`.

    The inverse function is given by :meth:`mlprum.data.Dataset.cholesky`.

    :param array: `NumPy` array with shape :code:`(M)` or :code:`(N, M)`, where M is the m-th triangular number
    :type array: :class:`numpy.ndarray`
    :raises ValueError: if array shape is not correct
    :return: `Numpy` array with shape :code:`(m, m)` or :code:`(N, m, m)`
    :rtype: :class:`numpy.ndarray`
    """
    if array.ndim == 1:
        array = np.expand_dims(array, axis=0)
    N, M = array.shape
    m = np.sqrt(2 * M + 0.25) - 0.5
    if m == np.floor(m):  # check if M is triangular number
        m = int(m)
    else:
        raise ValueError
    C_tril = np.zeros((N, m, m))
    idx0 = np.arange(N)
    idx1, idx2 = np.tril_indices(m)
    idx0, idx1, idx2 = np.repeat(idx0, array.shape[1]), np.tile(idx1, N), np.tile(idx2, N)
    C_tril[idx0, idx1, idx2] = array.flatten()
    C = np.einsum('nij,nkj->nik', C_tril, C_tril)  # Einstein summation of C_tril @ C_tril.T

    if C.shape[0] == 1:
        C = np.squeeze(C, axis=0)
    return C

def inv_cholesky(array):
    L_tril = tril_cholesky(array)
    return torch.linalg.inv(L_tril)

def unsqueeze(output, target):
    if output.dim() == 1:
        output = torch.unsqueeze(output, 0)
    if target.dim() == 1:
        target = torch.unsqueeze(target, 0)
    return output, target

def stiffness_loss(output, target, reduction='mean'):
    output, target = unsqueeze(output, target)
    L, L_pred = (target, output[:,:21]) if target.size(1) == 21 else (target[:,:21], output[:,:21])
    C, C_pred = reverse_cholesky(L), reverse_cholesky(L_pred)
    #assert torch.allclose(torch.linalg.cholesky(C), tril_cholesky(L))
    #loss = (torch.linalg.norm(L - L_pred, dim=1) / torch.linalg.norm(L, dim=1))**2
    invL = inv_cholesky(L)
    C_star = torch.einsum('nij,njk,nlk->nil', invL, C_pred, invL)  # Einstein summation of invL @ C_pred @ invL.T
    #L_tril = tril_cholesky(L)
    #C_star2 = torch.linalg.solve_triangular(L_tril, torch.linalg.solve_triangular(L_tril.transpose(1,2), C_pred, left=False, upper=True), upper=False)
    I = torch.eye(6).repeat(target.size(0), 1, 1)
    loss = torch.linalg.norm(C_star - I, dim=[1,2]) / torch.linalg.norm(I, dim=[1,2])
    #loss2 = torch.linalg.norm(C_star2 - I, dim=[1,2]) / torch.linalg.norm(I, dim=[1,2])
    #assert torch.allclose(loss1, loss2)
    return torch.mean(loss)

def thermal_strain_loss(output, target, reduction='mean'):
    output, target = unsqueeze(output, target)
    eps, eps_pred = (target, output[:,:6]) if target.size(1) == 6 else (target[:,21:27], output[:,21:27])
    loss = torch.linalg.norm(eps - eps_pred, dim=1) / torch.linalg.norm(eps, dim=1)
    return torch.mean(loss)
    
def mech_loss(output, target, reduction='mean'):
    output, target = unsqueeze(output, target)
    L, L_pred = target[:,:21], output[:,:21]
    eps, eps_pred = target[:,21:27], output[:,21:27]
    loss1 = stiffness_loss(L_pred, L, reduction)
    loss2 = thermal_strain_loss(eps_pred, eps, reduction)
    #print(loss1, loss2)
    return loss1 + loss2

def get_data(data_loader, device='cpu'):
    x_list, y_list = [], []
    for x_batch, y_batch in list(data_loader):
        x_list.append(x_batch)
        y_list.append(y_batch)
    x = torch.cat(x_list)
    y = torch.cat(y_list)
    return x.to(device), y.to(device)

def model_training(model, loss_fn, optimizer, train_loader, val_loader, epochs, verbose=False):
    early_stop_patience = 1
    early_stop_counter = early_stop_patience
    epoch_list = []
    train_losses = []
    val_losses = []
    best_epoch = 0
    best_loss = float('inf')
    best_parameters = model.state_dict()
    for t in range(epochs):
        epoch_list.append(t + 1)
        # training step:
        model.training_step(train_loader, loss_fn, optimizer)
        train_loss = model.loss_calculation(train_loader, loss_fn)
        train_losses.append(train_loss)
        if np.isnan(train_loss):
            raise Exception('training loss is not a number')
        # validation step:
        val_loss = model.loss_calculation(val_loader, loss_fn)
        val_losses.append(val_loss)
        # early stopping:
        if t > int(0.1 * epochs) and val_loss < best_loss:
            if early_stop_counter < early_stop_patience:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
                best_epoch, best_loss = t, val_loss
                best_parameters = model.state_dict()
        # status update:
        if verbose and ((t + 1) % 1000 == 0):
            print(f"Epoch {t + 1}: training loss {train_loss:>8f}, validation loss {val_loss:>8f}")
    model.load_state_dict(best_parameters)
    return train_losses, val_losses, best_epoch

def plot_training_history(ax, train_losses, val_losses, best_epoch):
    epoch_list = torch.arange(len(train_losses)) + 1
    if min(train_losses) < 0:
        # probabilistic loss
        ax.plot(epoch_list, train_losses, linestyle='solid', alpha=1, label='prob. training loss')
        ax.plot(epoch_list, val_losses, linestyle='solid', alpha=0.7, label='prob. validation loss')
        # plt.ylim(top=0, bottom=-60)
        print(
            f'Best epoch ({best_epoch}): prob. training loss {train_losses[best_epoch]}, prob. validation loss {val_losses[best_epoch]}'
        )
    else:
        # deterministic loss
        ax.semilogy(epoch_list, train_losses, linestyle='solid', alpha=1, label='training loss')
        ax.semilogy(epoch_list, val_losses, linestyle='solid', alpha=0.7, label='validation loss')
        print(
            f'Best epoch ({best_epoch}): training loss {train_losses[best_epoch]:e}, validation loss {val_losses[best_epoch]:e}')
    ax.axvline(x=best_epoch, color='k', linestyle='dashed', label='best epoch')
    ax.legend()
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    plt.show()

def get_stats(df, key):
    data = [np.array(df_s[key]) for df_s in df]
    d_med = np.array([np.median(d) for d in data])
    d_extrema = [d_med - np.array([np.quantile(d, 0.25) for d in data]), np.array([np.quantile(d, 0.75) for d in data]) - d_med]
    # d_extrema = [d_med - np.array([np.min(d) for d in data]), np.array([np.min(d) for d in data]) - d_med]
    # d_mean = np.mean(data, axis=1)
    # d_std = np.std(data, axis=1)
    return d_med, d_extrema
    # return d_mean, [d_std, d_std]

def plot_model(df, ax, xaxis, prob=False, train_loss=True):
    df_samples = [y for x, y in df.groupby(xaxis, as_index=False)]
    if prob:
        train_med, train_err = get_stats(df_samples, 'prob_train_loss')
        val_med, val_err = get_stats(df_samples, 'prob_val_loss')
    else:
        train_med, train_err = get_stats(df_samples, 'train_loss')
        val_med, val_err = get_stats(df_samples, 'val_loss')
    train_samples = df[xaxis].unique()
    if train_loss:
        ax.errorbar(train_samples, train_med, yerr=train_err, fmt='--o', label='training loss', capsize=8)
    ax.errorbar(train_samples, val_med, yerr=val_err, fmt='--o', label='validation loss', capsize=8)
    # axs[1,0].set_ylim([1e-7, 1e-4])
    ##axs[1,1].set_ylim([1e-4, 1e-1])
    # ax.set_yscale('log', nonpositive='clip')
    ax.set_xlabel(xaxis)
    if not prob:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.legend()

def bisection_sampling(dset, levels=0, samples_per_pos=1, validation=False):
    """Samples training and validation data from the PyTorch Dataset `dset`
    using a bisection sampling strategy with a given number of bisection `levels`.
    At each point that is obtained by bisecting the data a number of `samples_per_pos`
    are sampled around this position. If `val` is set to `True` (default: `False`),
    then extra validation data is sampled from the points in between the training data.
    Otherwise the returned validation data is the same as the training data.

    :param dset: dataset to be sampled from
    :type dset: PyTorch Dataset
    :param levels: number of bisection levels, defaults to 0 (i.e. only two positions)
    :type levels: int, optional
    :param samples_per_pos: number of samples around each position, defaults to 1
    :type samples_per_pos: int, optional
    :param validation: use extra validation data if True, defaults to False (i.e. validation is the same as training data)
    :type validation: bool, optional
    :return: training dataset, validation dataset
    :rtype: tuple of PyTorch Datasets
    """
    max_idx = len(dset) - samples_per_pos
    idx = torch.linspace(0, max_idx, 1 + 2**(levels + 1), dtype=int)
    train_idx = idx[::2]
    val_idx = idx[1::2] if validation else idx[::2]
    train_idx_all = torch.cat([train_idx + i for i in range(samples_per_pos)])
    val_idx_all = torch.cat([val_idx + i for i in range(samples_per_pos)])
    train_data = TensorDataset(*dset[train_idx_all])
    val_data = TensorDataset(*dset[val_idx_all])
    return train_data, val_data

def hierarchical_sampling(dset, x, sampling_points, validation=False):
    """Select the data points that are closest to a given list of `sampling_points`
    :param dset: _description_
    :type dset: _type_
    :param sampling_points: _description_
    :type sampling_points: _type_
    :return: _description_
    :rtype: _type_
    """
    train_idx = [int(torch.argmin(torch.abs(x - t))) for t in sampling_points]
    train_data = TensorDataset(*dset[train_idx])
    val_idx = set(np.random.randint(low=1, high=len(dset), size=len(train_idx) - 1)).difference(train_idx)
    val_data = TensorDataset(*dset[val_idx]) if validation else train_data
    return train_data, val_data
