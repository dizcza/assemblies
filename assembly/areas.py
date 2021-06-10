from collections import OrderedDict
from string import ascii_uppercase

import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from assembly.constants import K_ACTIVE
from assembly.samplers import sample_bernoulli
from mighty.utils.common import find_layers

__all__ = [
    "KWinnersTakeAll",
    "AreaRNNHebb",
    "AreaRNNWillshaw",
    "AreaStack",
    "AreaSequential"
]


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

    def complete_from_input(self, xs_partial, y_partial=None):
        """
        Complete the pattern from the partial input.

        Nothing more than a simple forward pass without updating the weights.

        Parameters
        ----------
        xs_partial : torch.Tensor or tuple of torch.Tensor
            Partially active input vectors from the incoming areas.
        y_partial : torch.Tensor or None, optional
            The stored latent (hidden activations) vector from the previous
            step with partial activations.
            Default: None

        Returns
        -------
        y_out : torch.Tensor
            The output vector.
        """
        mode = self.training
        self.eval()
        y_out = self(xs_partial, y_latent=y_partial)
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

        Without the normalization, all inputs converge to the same output
        vector determined by the lateral weights because the sum
        ``w_xy @ x + w_lat @ y`` favors the second element. Normalization of
        the feedforward and lateral weights makes ``w_xy @ x`` and
        ``w_lat @ y`` of the same magnitude.
        """
        for module in find_layers(self, layer_class=AreaRNN):
            for weight in module.parameters(recurse=False):
                # input and recurrent weights
                module._normalize_weight(weight)
                assert torch.isfinite(weight).all()


class AreaRNN(AreaInterface, ABC):
    def __init__(self, *in_features: int, out_features, p_synapse=0.05,
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
        if xs_stim is None or all(x is None for x in xs_stim):
            return None
        assert len(xs_stim) == len(self.weights_input)
        y_out = torch.zeros(self.out_features)
        for x, w_in in zip(xs_stim, self.weights_input):
            if x is not None:
                y_out += w_in.matmul(x)
        if y_latent is not None:
            # y_out += alpha * W_rec @ y_latent
            y_out.addmv_(mat=self.weight_recurrent, vec=y_latent,
                         alpha=self.recurrent_coef)
        y_out = self.kwta(y_out)
        if self.training:
            for x, w_in in zip(xs_stim, self.weights_input):
                if x is not None:
                    self.update_weight(w_in, x=x, y=y_out)
            if y_latent is not None:
                self.update_weight(self.weight_recurrent, x=y_latent, y=y_out)
        return y_out

    def update_weight(self, weight, x, y):
        """
        Update the weight, given the activations.

        Parameters
        ----------
        weight : torch.Tensor
            The weight to update.
        x, y : torch.Tensor
            Input and output vectors.
        """
        pass

    @abstractmethod
    def _normalize_weight(self, weight):
        """
        Normalize the pre-synaptic weight sum to ``1.0``.

        Parameters
        ----------
        weight : torch.Tensor
            A weight matrix.
        """
        pass

    def complete_pattern(self, y_partial):
        """
        Complete the pattern using the recurrent connections only.

        Parameters
        ----------
        y_partial : torch.Tensor
            A partially activated latent vector.

        Returns
        -------
        y : torch.Tensor
            The reconstructed vector `y`.
        """
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

    * additive:

        .. math::
            W_{ij} = W_{ij} + \beta
            :label: update-additive

    * multiplicative:

        .. math::
            W_{ij} = W_{ij} * (1 + \beta)
            :label: update-multiplicative

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
        Default: 0.05
    recurrent_coef : float, optional
        The recurrent coefficient :math:`\alpha` in :eq:`forward`.
        Default: 1
    learning_rate : float, optional
        The plasticity coefficient :math:`\beta` in :eq:`update-additive` and
        :eq:`update-multiplicative`.
        Default: 0.1
    sampler : {sample_bernoulli, sample_uniform_masked}, optional
        Weights initialization function to call: either Bernoulli or uniform.
        Default: sample_bernoulli
    update : {'additive', 'multiplicative'}, optional
        The weight update learning rule.
        Default: 'multiplicative'

    Notes
    -----
    `'additive'` update learning rule allows new weights to grow, as opposed
    to `'multiplicative'`.

    """
    def __init__(self, *in_features: int, out_features, p_synapse=0.05,
                 recurrent_coef=1., learning_rate=0.1,
                 sampler=sample_bernoulli, update='multiplicative'):
        super().__init__(*in_features, out_features=out_features,
                         p_synapse=p_synapse, recurrent_coef=recurrent_coef,
                         sampler=sampler)
        self.learning_rate = learning_rate
        if update == 'additive':
            self.update_weight = self.update_weight_additive
        elif update == 'multiplicative':
            self.update_weight = self.update_weight_multiplicative
        else:
            raise ValueError(f"Invalid update rule: '{update}'")

    def update_weight_additive(self, weight, x, y):
        # w_ij = w_ij + learning_rate, if x_j and y_i fired:
        # w_ij = w_ij + learning_rate * x_j * y_i
        weight.addr_(y, x, alpha=self.learning_rate)

    def update_weight_multiplicative(self, weight, x, y):
        # w_ij = w_ij * (1 + learning_rate), if x_j and y_i fired:
        # w_ij = w_ij * (1 + learning_rate * x_j * y_i)
        weight.mul_(1 + self.learning_rate * y.unsqueeze(1) * x.unsqueeze(0))

    def _normalize_weight(self, weight):
        presum = weight.sum(dim=1, keepdim=True)
        presum[presum == 0] = 1  # all elements in a row are zeros
        weight /= presum

    def extra_repr(self):
        update = self.update_weight.__name__.lstrip('update_weight_')
        return f"{super().extra_repr()}, update='{update}'"


class AreaRNNWillshaw(AreaRNN):
    r"""
    Non-Holographic Associative Memory Area [1]_: a recurrent neural network
    with one or more incoming input layers and only one output layer. The
    weights are sparse and binary.

    The update rule, if :math:`x_j` and :math:`y_i` neurons fired:

    .. math::
        W_{ij} = 1
        :label: update-will

    This update rule is the simplest possible update rule that requires
    neither the learning rate nor the weight normalization, compared to
    :class:`AreaRNNHebb`.

    Parameters
    ----------
    *in_features
        The sizes of input vectors from incoming areas.
    out_features : int
        The size of the output layer.
    p_synapse : float, optional
        The initial probability of recurrent and afferent synaptic
        connectivity.
        Default: 0.05
    recurrent_coef : float, optional
        The recurrent coefficient :math:`\alpha` in :eq:`forward`.
        Default: 1.0

    References
    ----------
    .. [1] Willshaw, D. J., Buneman, O. P., & Longuet-Higgins, H. C. (1969).
       Non-holographic associative memory. Nature, 222(5197), 960-962.

    """

    def __init__(self, *in_features: int, out_features, p_synapse=0.05,
                 recurrent_coef=1, **ignored):
        super().__init__(*in_features,
                         out_features=out_features,
                         p_synapse=p_synapse,
                         recurrent_coef=recurrent_coef,
                         sampler=sample_bernoulli)

    def update_weight(self, weight, x, y):
        # w_ij = 1, if x_j and y_i fired, and 0 otherwise
        weight.addr_(y, x)
        weight.clamp_max_(1)

    def _normalize_weight(self, weight):
        # the weights are already binary at the update stage
        pass


class AreaStack(nn.Sequential, AreaInterface):
    """
    Vertically stacked areas. The output activations will be linearly summed.

    Parameters
    ----------
    *areas
        Vertically stacked :class:`AreaRNN`.
    """

    def __init__(self, *areas: AreaRNN):
        areas_named = OrderedDict({
            f"{letter}": area for letter, area in zip(ascii_uppercase, areas)
        })
        nn.Sequential.__init__(self, areas_named)

    def forward(self, xs_stim, y_latent=None):
        if xs_stim is None:
            xs_stim = [None] * len(self)
        assert len(xs_stim) == len(self)
        if y_latent is None:
            y_latent = [None] * len(xs_stim)
        y_out = [area(x, y_latent=yl) for area, x, yl in
                 zip(self, xs_stim, y_latent)]
        return y_out


class AreaSequential(nn.Sequential, AreaInterface):
    """
    A sequence of areas. The output of one area is fed into the next area.
    """

    def forward(self, xs_stim, y_latent=None):
        if xs_stim is None:
            xs_stim = [None] * len(self)
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
