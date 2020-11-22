import torch

from nn.constants import K_ACTIVE, N_NEURONS

__all__ = [
    "sample_bernoulli",
    "sample_uniform_masked",
    "sample_k_active"
]


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
