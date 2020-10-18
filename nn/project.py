import math
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

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
    def __init__(self, k_active=K_ACTIVE):
        super().__init__()
        self.k_active = k_active

    def forward(self, x: torch.Tensor):
        winners = x.topk(k=self.k_active, sorted=False).indices
        y = torch.zeros_like(x)
        y[winners] = 1
        return y

    def extra_repr(self):
        return f"k_active={self.k_active}"


class AreaInterface(nn.Module, ABC):
    def recall(self, *xs_stim: torch.Tensor) -> torch.Tensor:
        mode = self.training
        self.eval()
        y_out = self(*xs_stim)
        self.train(mode)
        return y_out

    def complete_from_input(self, *xs_partial: torch.Tensor,
                            y_latent=None) -> torch.Tensor:
        mode = self.training
        self.eval()
        y_out = self(*xs_partial, y_latent=y_latent)
        self.train(mode)
        return y_out

    def memory_used(self):
        mem_used = {}
        for name, param in self.named_parameters():
            mem_used[name] = param.norm(p=0) / param.nelement()
        return mem_used


class AreaRNN(AreaInterface, ABC):
    def __init__(self, *in_features: int, out_features: int, p_synapse=0.01,
                 recurrent_coef=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.recurrent_coef = recurrent_coef
        self.weights_input = []
        for parent_id, neurons_in in enumerate(in_features):
            weight_in = nn.Parameter(
                sample_bernoulli(out_features, neurons_in, proba=p_synapse),
                requires_grad=False)
            self.register_parameter(name=f"weight_input{parent_id}",
                                    param=weight_in)
            self.weights_input.append(weight_in)
        self.weight_recurrent = nn.Parameter(
            sample_bernoulli(out_features, out_features, proba=p_synapse),
            requires_grad=False)
        self.kwta = KWinnersTakeAll()
        self.normalize_weights()

    def forward(self, *xs_stim: torch.Tensor, y_latent=None):
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

    def recall(self, *xs_stim: torch.Tensor):
        mode = self.training
        self.eval()
        y_out = self(*xs_stim)
        self.train(mode)
        return y_out

    @abstractmethod
    def update_weight(self, weight, x, y):
        pass

    def normalize_weights(self):
        for w_in in self.parameters():
            # input and recurrent weights
            self.normalize_weight(w_in)

    @abstractmethod
    def normalize_weight(self, weight):
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
    def __init__(self, *in_features: int, out_features: int, p_synapse=0.01,
                 recurrent_coef=1., learning_rate=0.1):
        super().__init__(*in_features, out_features=out_features,
                         p_synapse=p_synapse, recurrent_coef=recurrent_coef)
        self.learning_rate = learning_rate

    def update_weight(self, weight, x, y):
        weight.addr_(y, x, alpha=self.learning_rate)

    def normalize_weight(self, weight):
        weight /= weight.sum(dim=1, keepdim=True)


class AreaRNNWillshaw(AreaRNN):
    """
    Non-Holographic Associative Memory Area by Willshaw.
    """

    def update_weight(self, weight, x, y):
        weight.addr_(y, x)
        weight.clamp_max_(1)

    def normalize_weight(self, weight):
        # the weights are already binary at the update stage
        pass


class AreaStack(nn.Sequential, AreaInterface):

    def __init__(self, *areas: AreaRNN):
        areas_named = OrderedDict({
            f"vertical{idx}": area for idx, area in enumerate(areas)
        })
        nn.Sequential.__init__(self, areas_named)

    def forward(self, *xs_stim: torch.Tensor, y_latent=None):
        assert len(xs_stim) == len(self)
        if y_latent is None:
            y_latent = [None] * len(xs_stim)
        y_out = [area(x, y_latent=yl) for area, x, yl in
                 zip(self, xs_stim, y_latent)]
        return y_out


class AreasSequential(nn.Sequential, AreaInterface):

    def forward(self, *xs_stim, y_latent=None):
        assert len(xs_stim) == len(self)
        y_out = xs_stim
        if y_latent is None:
            y_latent = [None] * len(self)
        y_intermediate = []  # hidden activations of the intermediate layers
        for module, yl in zip(self, y_latent):
            if isinstance(y_out, torch.Tensor):
                y_out = [y_out]
            y_out = module(*y_out, y_latent=yl)
            y_intermediate.append(y_out)
        return y_out, y_intermediate

    def recall(self, *xs_stim: torch.Tensor):
        mode = self.training
        self.eval()
        y_out, y_intermediate = self(*xs_stim)
        self.train(mode)
        return y_out


def recall_overlap(xs, area: AreaRNN, ys_learned):
    y_predicted = torch.stack([area.recall(x) for x in xs])
    recall = (y_predicted * torch.stack(ys_learned)).sum(dim=1).mean()
    return recall


def pairwise_similarity(tensors):
    if not isinstance(tensors, torch.Tensor):
        tensors = torch.stack(tensors)
    similarity = tensors[1:].matmul(tensors[:-1].t())
    similarity = similarity.mean() / K_ACTIVE
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
    for xa, xb in zip(xa_samples, xb_samples):
        y_latent = None
        for step in range(epoch_size):
            y_out, y_latent = brain(xa, xb, y_latent=y_latent)


def simulate(n_samples=10, epoch_size=10):
    area = AreaRNNHebb(N_NEURONS, out_features=N_NEURONS // 2)
    print(area)
    xs = [sample_k_active(n=N_NEURONS, k=K_ACTIVE) for _ in range(n_samples)]
    ys_learned = []
    overlaps_convergence = []
    overlaps_learned = []  # recall
    memory_used = defaultdict(list)
    for sample_count, x in enumerate(tqdm(xs, desc="Projecting"), start=1):
        y_prev = None  # inhibit the area
        for step in range(epoch_size):
            y = area(x, y_latent=y_prev)
            converged = False
            if y_prev is None:
                overlaps_convergence.append(np.nan)
            else:
                overlap = y.matmul(y_prev).item()
                overlaps_convergence.append(overlap)
                converged = overlap == K_ACTIVE
            overlap_recall = recall_overlap(
                xs[:sample_count],
                area=area,
                ys_learned=ys_learned + [y]
            )
            overlaps_learned.append(overlap_recall)
            mem_used_dict = area.memory_used()
            memory_used['input'].append(mem_used_dict['weight_input0'])
            memory_used['recurrent'].append(mem_used_dict['weight_recurrent'])
            y_prev = y
            if converged:
                break
        ys_learned.append(y_prev)
        area.normalize_weights()

    print(f"Expected random overlap: "
          f"{expected_random_overlap(n=N_NEURONS, k=K_ACTIVE):.3f}")
    print(f"Learned assemblies similarity: "
          f"{pairwise_similarity(ys_learned):.3f}, "
          f"input: {pairwise_similarity(xs):.3f}")
    fig, axes = plt.subplots(nrows=2, ncols=1)

    iterations = np.arange(len(overlaps_convergence))
    axes[0].plot(iterations, overlaps_convergence,
                 label='convergence (y, y_prev)', marker='o')
    axes[0].plot(iterations, overlaps_learned,
                 label='recall (y_pred, y_learned)')
    axes[0].set_ylabel("overlap")
    axes[0].set_xticks([])
    xmin, xmax = axes[0].get_xlim()
    axes[0].axhline(y=K_ACTIVE, xmin=xmin, xmax=xmax, ls='--',
                    color='black',
                    label='k active', alpha=0.5)
    axes[0].legend()

    for key in memory_used.keys():
        axes[1].plot(iterations, memory_used[key], label=key)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"Memory used ($L_0$ norm)")
    axes[1].legend()
    plt.suptitle(f"{n_samples} learned samples")
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(19)
    # associate_example()
    simulate()
