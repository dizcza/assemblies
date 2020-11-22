r"""

PyTorch implementation of `project` and `associate` operations [1]_.


The forward pass is defined in two steps:

.. math::
    \begin{cases}
    \bold{y} = \sum_l W^{input}_l \bold{x}_l +
               \alpha W^{recurrent}\bold{y}^{latent}
    \\
    \bold{y} = \text{kWTA}(\bold{y}, k)
    \end{cases}
    :label: forward


Assembly areas
--------------
.. autosummary::
    :toctree: toctree/nn

    AreaRNNHebb
    AreaRNNWillshaw
    AreaStack
    AreasSequential


Activation function
-------------------
.. autosummary::
    :toctree: toctree/nn

    KWinnersTakeAll

References
----------
.. [1] Papadimitriou, C. H., Vempala, S. S., Mitropolsky, D., Collins, M., &
   Maass, W. (2020). Brain computation by assemblies of neurons. Proceedings of
   the National Academy of Sciences.

"""

from collections import OrderedDict, defaultdict

import math
import numpy as np
import time
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm

from mighty.monitor.batch_timer import timer
from mighty.monitor.viz import VisdomMighty
from mighty.utils.common import find_layers, find_named_layers

N_NEURONS = 1000
K_ACTIVE = 50


def expected_random_overlap(n, k):
    p_overlap = [math.comb(k, x) * math.comb(n - k, k - x) / math.comb(n, k)
                 for x in range(k + 1)]
    overlap_expected = np.multiply(p_overlap, range(k + 1)).sum()
    return overlap_expected


def sample_bernoulli(*size, proba: float):
    proba = torch.full(size, fill_value=proba)
    return torch.bernoulli(proba)


def sample_uniform_masked(*size, proba: float):
    mask = sample_bernoulli(*size, proba=proba)
    x = torch.rand(*size)
    return x * mask


def sample_k_active(n=N_NEURONS, k=K_ACTIVE):
    k_indices = torch.randperm(n)[:k]
    x = torch.zeros(n)
    x[k_indices] = 1
    return x


class KWinnersTakeAll(nn.Module):
    """
    K-winners-take-all activation function.

    Parameters
    ----------
    k_active : int, optional
        `k`, the number of active (winner) neurons within an output layer.
        Default: 50
    """
    def __init__(self, k_active=K_ACTIVE):
        super().__init__()
        self.k_active = k_active

    def forward(self, x):
        """
        The forward pass of kWTA.

        Parameters
        ----------
        x : (N,) torch.Tensor
            An input vector.

        Returns
        -------
        y : (N,) torch.Tensor
            The output vector ``y = kwta(x)`` with exactly :attr:`k` active
            neurons.

        """
        winners = x.topk(k=self.k_active, sorted=False).indices
        y = torch.zeros_like(x)
        y[winners] = 1
        return y

    def extra_repr(self):
        return f"k_active={self.k_active}"


class AreaInterface(nn.Module, ABC):
    def recall(self, xs_stim):
        """
        A forward pass without latent activations.

        Parameters
        ----------
        xs_stim : torch.Tensor or tuple of torch.Tensor
            Input vectors from the incoming areas.

        Returns
        -------
        y_out : torch.Tensor
            The output vector.
        """
        mode = self.training
        self.eval()
        y_out = self(xs_stim)
        self.train(mode)
        return y_out

    def complete_from_input(self, xs_partial, y_latent=None):
        """
        Complete the pattern from the partial input.

        Nothing more than a simple forward pass without updating the weights.

        Parameters
        ----------
        xs_partial : torch.Tensor or tuple of torch.Tensor
            Input vectors from the incoming areas.
        y_latent : torch.Tensor or None, optional
            The stored latent (hidden activations) vector from the previous
            step.
            Default: None

        Returns
        -------
        y_out : torch.Tensor
            The output vector.
        """
        mode = self.training
        self.eval()
        y_out = self(xs_partial, y_latent=y_latent)
        self.train(mode)
        return y_out

    def memory_used(self):
        r"""
        Computes the used memory bits as
        :math:`\frac{||W||_0}{\text{size}(W)}`

        Returns
        -------
        dict
            A dictionary with used memory for each parameter (weight matrix).
        """
        memory_used = {}
        for name, param in self.named_parameters():
            memory_used[name] = param.norm(p=0) / param.nelement()
        return memory_used

    def normalize_weights(self):
        """
        Normalize the pre-synaptic weights sum to ``1.0``.
        """
        for module in find_layers(self, layer_class=AreaRNN):
            for w_in in module.parameters(recurse=False):
                # input and recurrent weights
                module.normalize_weight(w_in)


class AreaRNN(AreaInterface, ABC):
    def __init__(self, *in_features: int, out_features, p_synapse=0.01,
                 recurrent_coef=1., sampler=sample_bernoulli):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.recurrent_coef = recurrent_coef
        self.weights_input = []
        for parent_id, neurons_in in enumerate(in_features):
            weight_in = nn.Parameter(
                sampler(out_features, neurons_in, proba=p_synapse),
                requires_grad=False)
            self.register_parameter(name=f"weight_input{parent_id}",
                                    param=weight_in)
            self.weights_input.append(weight_in)
        self.weight_recurrent = nn.Parameter(
            sampler(out_features, out_features, proba=p_synapse),
            requires_grad=False)
        self.kwta = KWinnersTakeAll()
        self.normalize_weights()

    def forward(self, xs_stim, y_latent=None):
        """
        The forward pass :eq:`forward`.

        Parameters
        ----------
        xs_stim : torch.Tensor or tuple of torch.Tensor
            Input vectors from the incoming areas.
        y_latent : torch.Tensor or None, optional
            The stored latent (hidden activations) vector from the previous
            step.
            Default: None

        Returns
        -------
        y_out : torch.Tensor
            The output vector.

        """
        if isinstance(xs_stim, torch.Tensor):
            xs_stim = [xs_stim]
        assert len(xs_stim) == len(self.weights_input)
        y_out = torch.zeros(self.out_features)
        for x, w_in in zip(xs_stim, self.weights_input):
            y_out += w_in.matmul(x)
        if y_latent is not None:
            # y_out += alpha * W_rec @ y_latent
            y_out.addmv_(mat=self.weight_recurrent, vec=y_latent,
                         alpha=self.recurrent_coef)
        y_out = self.kwta(y_out)
        if self.training:
            for x, w_in in zip(xs_stim, self.weights_input):
                self.update_weight(w_in, x=x, y=y_out)
            if y_latent is not None:
                self.update_weight(self.weight_recurrent, x=y_latent, y=y_out)
        return y_out

    def recall(self, xs_stim: torch.Tensor):
        mode = self.training
        self.eval()
        y_out = self(xs_stim)
        self.train(mode)
        return y_out

    @abstractmethod
    def update_weight(self, weight, x, y):
        pass

    @abstractmethod
    def normalize_weight(self, weight):
        """
        Normalize the pre-synaptic weights sum to ``1.0``.

        Parameters
        ----------
        weight : torch.Tensor
            A weight matrix.
        """
        pass

    def complete_pattern(self, y_partial: torch.Tensor) -> torch.Tensor:
        y = self.weight_recurrent.matmul(y_partial)
        y = self.kwta(y)
        return y

    def extra_repr(self):
        return f"in_features: {self.in_features}, " \
               f"out_features: {self.out_features}, " \
               f"recurrent_coef={self.recurrent_coef}"


class AreaRNNHebb(AreaRNN):
    r"""
    A Hebbian-learning recurrent neural network with one or more incoming input
    layers and only one output layer.

    The update rule, if :math:`x_j` and :math:`y_i` neurons fired:

    .. math::
        W_{ij} = W_{ij} + \beta
        :label: update-hebb

    After each epoch, many repetitions of the same input trial, the weights
    are normalized to have ``1.0`` in its pre-synaptic sum for each neuron.

    Parameters
    ----------
    *in_features
        The sizes of input vectors from incoming areas.
    out_features : int
        The size of the output layer.
    p_synapse : float, optional
        The initial probability of recurrent and afferent synaptic
        connectivity.
        Default: 0.01
    recurrent_coef : float, optional
        The recurrent coefficient :math:`\alpha` in :eq:`forward`.
        Default: 1
    learning_rate : float, optional
        The plasticity coefficient :math:`\beta` in :eq:`update-hebb`.
        Default: 0.1
    sampler : {sample_bernoulli, sample_uniform_masked}, optional
        Weights initialization function to call: either Bernoulli or uniform.
        Default: sample_bernoulli

    """
    def __init__(self, *in_features: int, out_features, p_synapse=0.01,
                 recurrent_coef=1., learning_rate=0.1,
                 sampler=sample_bernoulli):
        super().__init__(*in_features, out_features=out_features,
                         p_synapse=p_synapse, recurrent_coef=recurrent_coef,
                         sampler=sampler)
        self.learning_rate = learning_rate

    def update_weight(self, weight, x, y):
        # w_ij = w_ij + learning_rate, if x_j and y_i fired:
        # w_ij = w_ij + learning_rate * x_j * y_i
        weight.addr_(y, x, alpha=self.learning_rate)

    def update_weight_multiplicative(self, weight, x, y):
        # w_ij = w_ij * (1 + learning_rate), if x_j and y_i fired:
        # w_ij = w_ij * (1 + learning_rate * x_j * y_i)
        weight.mul_(1 + self.learning_rate * y.unsqueeze(1) * x.unsqueeze(0))

    def normalize_weight(self, weight):
        weight /= weight.sum(dim=1, keepdim=True)


class AreaRNNWillshaw(AreaRNN):
    r"""
    Non-Holographic Associative Memory Area [1]_: a recurrent neural network
    with one or more incoming input layers and only one output layer. The
    weights are sparse and binary.

    The update rule, if :math:`x_j` and :math:`y_i` neurons fired:

    .. math::
        W_{ij} = 1
        :label: update-will

    Parameters
    ----------
    *in_features
        The sizes of input vectors from incoming areas.
    out_features : int
        The size of the output layer.
    p_synapse : float, optional
        The initial probability of recurrent and afferent synaptic
        connectivity.
        Default: 0.01
    recurrent_coef : float, optional
        The recurrent coefficient :math:`\alpha` in :eq:`forward`.
        Default: 1.0

    References
    ----------
    .. [1] Willshaw, D. J., Buneman, O. P., & Longuet-Higgins, H. C. (1969).
       Non-holographic associative memory. Nature, 222(5197), 960-962.

    """

    def __init__(self, *in_features: int, out_features, p_synapse=0.01,
                 recurrent_coef=1):
        super().__init__(*in_features,
                         out_features=out_features,
                         p_synapse=p_synapse,
                         recurrent_coef=recurrent_coef,
                         sampler=sample_bernoulli)

    def update_weight(self, weight, x, y):
        # w_ij = 1, if x_j and y_i fired, and 0 otherwise
        weight.addr_(y, x)
        weight.clamp_max_(1)

    def normalize_weight(self, weight):
        # the weights are already binary at the update stage
        pass


class AreaStack(nn.Sequential, AreaInterface):

    def __init__(self, *areas: AreaRNN):
        areas_named = OrderedDict({
            f"vertical-{idx}": area for idx, area in enumerate(areas)
        })
        nn.Sequential.__init__(self, areas_named)

    def forward(self, xs_stim, y_latent=None):
        assert len(xs_stim) == len(self)
        if y_latent is None:
            y_latent = [None] * len(xs_stim)
        y_out = [area(x, y_latent=yl) for area, x, yl in
                 zip(self, xs_stim, y_latent)]
        return y_out


class AreasSequential(nn.Sequential, AreaInterface):

    def forward(self, xs_stim, y_latent=None):
        assert len(xs_stim) == len(self)
        y_out = xs_stim
        if y_latent is None:
            y_latent = [None] * len(self)
        y_intermediate = []  # hidden activations of the intermediate layers
        for module, yl in zip(self, y_latent):
            y_out = module(y_out, y_latent=yl)
            y_intermediate.append(y_out)
        return y_out, y_intermediate

    def recall(self, xs_stim):
        y_out, y_intermediate = super().recall(xs_stim)
        return y_out


def pairwise_similarity(tensors):
    """

    Parameters
    ----------
    tensors : list of torch.Tensor
        A list of input vectors to recall. Each entry can be either a
        single vector tensor (one incoming area) or a tuple of tensors
        (multiple incoming areas).

    Returns
    -------

    """
    if not isinstance(tensors, torch.Tensor):
        if not isinstance(tensors[0], torch.Tensor):
            # multiple incoming areas
            sim_areas = list(map(pairwise_similarity, zip(*tensors)))
            sim_areas = np.mean(sim_areas)
            return sim_areas
        else:
            tensors = torch.stack(tensors)
    similarity = tensors.matmul(tensors.t())
    n_elements = len(tensors)
    ii, jj = torch.triu_indices(row=n_elements, col=n_elements, offset=1)
    similarity = similarity[ii, jj].mean()
    similarity /= K_ACTIVE
    return similarity


def associate_example(n_samples=10, epoch_size=10):
    na, nb, nc, n_out = N_NEURONS, 2 * N_NEURONS, N_NEURONS // 2, N_NEURONS
    area_A = AreaRNNHebb(na, out_features=nc)
    area_B = AreaRNNHebb(nb, out_features=nc)
    area_C = AreaRNNHebb(nc, nc, out_features=n_out)
    area_AB = AreaStack(area_A, area_B)
    brain = AreasSequential(area_AB, area_C)
    print(brain)
    xa_samples = [sample_k_active(n=na, k=K_ACTIVE) for _ in range(n_samples)]
    xb_samples = [sample_k_active(n=nb, k=K_ACTIVE) for _ in range(n_samples)]
    x_pairs = list(zip(xa_samples, xb_samples))
    simulate(area=brain, x_samples=x_pairs)
    # for xa, xb in zip(xa_samples, xb_samples):
    #     y_latent = None
    #     for step in range(epoch_size):
    #         y_out, y_latent = brain(xa, xb, y_latent=y_latent)


class Monitor:

    def __init__(self, area: AreaInterface):
        self.area = area
        env_name = f"{time.strftime('%Y.%m.%d')} {area.__class__.__name__}"
        self.viz = VisdomMighty(env=env_name)
        self.viz.close()  # clear previous plots
        self.handles = []
        self.module_name = dict()
        for name, layer in find_named_layers(
                area, layer_class=AreaRNN,
                name_prefix=area.__class__.__name__):
            self.module_name[layer] = name
            handle = layer.register_forward_hook(self._forward_hook)
            self.handles.append(handle)
        self.ys_output = dict()
        self.ys_previous = None
        self.ys_learned = defaultdict(list)

    def _forward_hook(self, module, input, output):
        name = self.module_name[module]
        self.ys_output[name] = output

    def remove_handles(self):
        """
        Remove the hooks that has been used to track intermediate layers
        output.
        """
        for handle in self.handles:
            handle.remove()

    def _update_convergence(self):
        if self.ys_previous is None:
            return
        overlaps = {'k-active': K_ACTIVE}
        for name in self.ys_output.keys():
            overlaps[name] = self.ys_output[name].matmul(
                self.ys_previous[name]).item()
        for name, overlap in overlaps.items():
            self.viz.line_update(y=overlap, opts=dict(
                xlabel='Epoch',
                ylabel='overlap',
                title='convergence (y, y_prev)'
            ), name=name)

    def _update_recall(self, x_samples_learned):
        assert len(self.ys_output) == 0
        ys_learned = {}
        for name in self.ys_previous.keys():
            ys_learned[name] = self.ys_learned[name] + [self.ys_previous[name]]
            assert len(ys_learned[name]) == len(x_samples_learned)
        recall = defaultdict(float)
        recall['k-active'] = K_ACTIVE
        for i, x in enumerate(x_samples_learned):
            # ys will be populated via the forward hook
            y_ignored = self.area.recall(x)
            for name, y_predicted in self.ys_output.items():
                y_learned = ys_learned[name][i]
                n_total = len(ys_learned[name])
                recall[name] += (y_predicted * y_learned).sum() / n_total
            self.ys_output.clear()
        for name in recall.keys():
            self.viz.line_update(y=recall[name], opts=dict(
                xlabel='Epoch',
                ylabel='overlap',
                title='recall (y_pred, y_learned)'
            ), name=name)

    def _update_memory_used(self):
        for name, memory_used in self.area.memory_used().items():
            self.viz.line_update(y=memory_used, opts=dict(
                xlabel='Epoch',
                title=r"Memory used (L0 norm)"
            ), name=name)

    def assembly_similarity(self):
        """
        Computes the similarity of learned assemblies.

        Returns
        -------
        similarity : dict
            A dict with pairwise assembly similarity.
        """
        similarity = dict()
        for name, y_learned in self.ys_learned.items():
            similarity[name] = pairwise_similarity(y_learned)
        return similarity

    def trial_finished(self, x_samples_learned):
        """
        A sample is being learned callback.

        Parameters
        ----------
        x_samples_learned : list
            A list of learned input vectors to recall. Each entry can be either
            a single vector tensor (one incoming area) or a tuple of tensors
            (multiple incoming areas).
        """
        self._update_convergence()
        self.ys_previous = self.ys_output.copy()
        self.ys_output.clear()
        self._update_recall(x_samples_learned)
        self._update_memory_used()

    def epoch_finished(self):
        """
        A sample is learned callback.
        """
        # ys_output is already cleared up
        for name, y_final in self.ys_previous.items():
            self.ys_learned[name].append(y_final)
        self.ys_previous = None

    def log_assembly_similarity(self, input_similarity=None):
        assembly_similarity = self.assembly_similarity()
        if input_similarity:
            assembly_similarity['input'] = input_similarity
        lines = ["Learned assemblies similarity:"]
        for name, similarity in assembly_similarity.items():
            lines.append(f"--{name}: {similarity:.3f}")
        text = '<br>'.join(lines)
        self.viz.log(text=text, timestamp=False)


def simulate(area, x_samples, epoch_size=10):
    timer.init(batches_in_epoch=epoch_size)
    monitor = Monitor(area=area)

    for sample_count, x in enumerate(tqdm(x_samples, desc="Projecting"),
                                     start=1):
        y_prev = None  # inhibit the area
        for step in range(epoch_size):
            y = area(x, y_latent=y_prev)
            if isinstance(area, AreasSequential):
                y, y_prev = y
            else:
                y_prev = y
            timer.tick()
            monitor.trial_finished(x_samples[:sample_count])
        area.normalize_weights()
        monitor.epoch_finished()

    monitor.log_assembly_similarity(
        input_similarity=pairwise_similarity(x_samples))
    monitor.viz.log(f"Expected random overlap: "
                    f"{expected_random_overlap(n=N_NEURONS, k=K_ACTIVE):.3f}",
                    timestamp=False)

    monitor.remove_handles()


def simulate_example(n_samples=10):
    area = AreaRNNHebb(N_NEURONS, out_features=N_NEURONS // 2)
    xs = [sample_k_active(n=N_NEURONS, k=K_ACTIVE) for _ in range(n_samples)]
    simulate(area=area, x_samples=xs)


if __name__ == '__main__':
    torch.manual_seed(19)
    associate_example()
    # simulate_example()
