import math
from abc import ABC, abstractmethod

import numpy as np
import torch
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


def k_winners_take_all(x: torch.Tensor, k=K_ACTIVE):
    winners = x.topk(k=k, sorted=False).indices
    y = torch.zeros_like(x)
    y[winners] = 1
    return y


class Area(ABC):
    def __init__(self, in_features: int, out_features: int, p=0.01,
                 recurrence_coef=1):
        self.weight_input = sample_bernoulli(out_features, in_features,
                                             proba=p)
        self.weight_recurrent = sample_bernoulli(out_features, out_features,
                                                 proba=p)
        self.recurrence_coef = recurrence_coef
        self.normalize_weights()

    def memory_used(self):
        wi = self.weight_input.norm(p=0) / self.weight_input.nelement()
        wr = self.weight_recurrent.norm(p=0) / self.weight_recurrent.nelement()
        return wi, wr

    @abstractmethod
    def normalize_weights(self):
        pass

    @abstractmethod
    def update_weights(self, x: torch.Tensor, y: torch.Tensor,
                       y_latent: torch.Tensor = None):
        pass

    def project(self, x: torch.Tensor, y_latent: torch.Tensor = None,
                train=True) -> torch.Tensor:
        y_out = x.matmul(self.weight_input.t())
        if y_latent is not None:
            y_out += self.recurrence_coef * \
                     y_latent.matmul(self.weight_recurrent.t())
        y_out = k_winners_take_all(y_out, k=K_ACTIVE)
        if train:
            self.update_weights(x=x, y=y_out, y_latent=y_latent)
        return y_out

    def recall(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return self.project(x=x, y_latent=None, train=False)
        ys = [self.project(x=xi, y_latent=None, train=False) for xi in x]
        return torch.stack(ys)

    def complete_pattern(self, y_partial: torch.Tensor) -> torch.Tensor:
        y = y_partial.matmul(self.weight_recurrent.t())
        y = k_winners_take_all(y, k=K_ACTIVE)
        return y

    def complete_from_input(self, x_partial: torch.Tensor,
                            y_latent: torch.Tensor = None) -> torch.Tensor:
        y_out = x_partial.matmul(self.weight_input.t())
        if y_latent is not None:
            y_out += self.recurrence_coef * \
                     y_latent.matmul(self.weight_recurrent.t())
        y_out = k_winners_take_all(y_out, k=K_ACTIVE)
        return y_out


class AreaHebb(Area):
    def __init__(self, in_features: int, out_features: int, p=0.01,
                 recurrence_coef=1, learning_rate=0.1):
        super().__init__(in_features=in_features, out_features=out_features,
                         p=p, recurrence_coef=recurrence_coef)
        self.learning_rate = learning_rate

    def normalize_weights(self):
        self.weight_input /= self.weight_input.sum(dim=1, keepdim=True)
        self.weight_recurrent /= self.weight_recurrent.sum(dim=1, keepdim=True)

    def update_weights_additive(self, x, y, y_latent=None):
        # w_ij = w_ij + learning_rate if x_ij and y_ij fired:
        # w_ij = w_ij + learning_rate * x_ij * y_ij
        def update(weight, _x):
            # add the outer-product of vectors _y and _x
            weight.addr_(y, _x, alpha=self.learning_rate)

        update(self.weight_input, x)
        if y_latent is not None:
            update(self.weight_recurrent, y_latent)

    def update_weights_multiplicative(self, x, y, y_latent=None):
        # w_ij = w_ij * (1 + learning_rate) if x_ij and y_ij fired:
        # w_ij = w_ij * (1 + learning_rate * x_ij * y_ij)
        def update(weight, _x):
            weight.mul_(1 + self.learning_rate *
                        y.unsqueeze(1) *
                        _x.unsqueeze(0))

        update(self.weight_input, x)
        if y_latent is not None:
            update(self.weight_recurrent, y_latent)

    def update_weights(self, x: torch.Tensor, y: torch.Tensor,
                       y_latent: torch.Tensor = None):
        self.update_weights_additive(x=x, y=y, y_latent=y_latent)


class AreaWillshaw(AreaHebb):
    """
    Non-Holographic Associative Memory Area by Willshaw.
    """

    def update_weights(self, x: torch.Tensor, y: torch.Tensor,
                       y_latent: torch.Tensor = None):
        def update(weight, _x):
            # add the outer-product of vectors _y and _x
            weight.addr_(y, _x)
            weight.clamp_max_(1)

        update(self.weight_input, x)
        if y_latent is not None:
            update(self.weight_recurrent, y_latent)

    def normalize_weights(self):
        return


def project(x: torch.Tensor, area_B: AreaHebb, y: torch.Tensor = None):
    return area_B.project(x=x, y_latent=y)


def recall_overlap(xs: torch.Tensor, area: AreaHebb, ys_learned: torch.Tensor):
    y_predicted = area.recall(xs)
    recall = (y_predicted * ys_learned).sum(dim=1)
    return recall


def pairwise_similarity(tensors):
    if not isinstance(tensors, torch.Tensor):
        tensors = torch.stack(tensors)
    similarity = tensors[1:].matmul(tensors[:-1].t())
    similarity = similarity.mean() / K_ACTIVE
    return similarity


def simulate(n_samples=10, epoch_size=10):
    area = AreaWillshaw(in_features=N_NEURONS, out_features=N_NEURONS // 2)
    xs = [sample_k_active(n=N_NEURONS, k=K_ACTIVE) for _ in range(n_samples)]
    xs = torch.stack(xs)
    pairwise_similarity(xs)
    ys_learned = []
    overlaps_convergence = []
    overlaps_learned = []  # recall
    memory_used = []
    for sample_count, x in enumerate(tqdm(xs, desc="Projecting"), start=1):
        y_prev = None  # inhibit the area
        for step in range(epoch_size):
            y = area.project(x=x, y_latent=y_prev)
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
                ys_learned=torch.stack(ys_learned + [y])
            )
            overlaps_learned.append(overlap_recall.mean())
            memory_used.append(area.memory_used())
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

    wi, wr = zip(*memory_used)
    axes[1].plot(iterations, wi, label='Input')
    axes[1].plot(iterations, wr, label='Recurrent')
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"Memory used ($L_0$ norm)")
    axes[1].legend()
    plt.suptitle(f"{n_samples} learned samples")
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(19)
    simulate()
