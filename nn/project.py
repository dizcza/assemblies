import torch
from tqdm import trange
from matplotlib import pyplot as plt

N_NEURONS = 1000
K_ACTIVE = 50


def sample_bernoulli(*size, proba: float):
    proba = torch.full(size, fill_value=proba)
    return torch.bernoulli(proba)


def sample_uniform_masked(*size, proba: float):
    mask = sample_bernoulli(*size, proba=proba)
    x = torch.rand(*size)
    return x * mask


def sample_k(n=N_NEURONS, k=K_ACTIVE):
    k_indices = torch.randperm(n)[:k]
    x = torch.ones(n)
    x[k_indices] = 1
    return x


def k_winners_take_all(x: torch.Tensor, k=K_ACTIVE):
    winners = x.topk(k=k, sorted=False).indices
    y = torch.zeros_like(x)
    y[winners] = 1
    return y


class Area:
    def __init__(self, in_features: int, out_features: int, p=0.01,
                 recurrence=1, learning_rate=0.01):
        self.weight_input = sample_bernoulli(out_features, in_features,
                                             proba=p)
        self.weight_recurrent = sample_bernoulli(out_features, out_features,
                                                 proba=p)
        self.recurrence = recurrence
        self.learning_rate = learning_rate
        self.normalize_weights()

    def memory_used(self):
        wi = self.weight_input.norm(p=0) / self.weight_input.nelement()
        wr = self.weight_recurrent.norm(p=0) / self.weight_recurrent.nelement()
        return wi, wr

    def normalize_weights(self):
        self.weight_input /= self.weight_input.sum(dim=1, keepdim=True)
        self.weight_recurrent /= self.weight_recurrent.sum(dim=1, keepdim=True)

    def update_weights_additive(self, x, y_out, y):
        # w_ij = w_ij + learning_rate if x_ij and y_ij fired:
        # w_ij = w_ij + learning_rate * x_ij * y_ij
        self.weight_input.add_(
            self.learning_rate * y_out.unsqueeze(1) * x.unsqueeze(0))
        if y is not None:
            self.weight_recurrent.add_(
                self.learning_rate * y_out.unsqueeze(1) * y.unsqueeze(0))

    def update_weights_multiplicative(self, x, y_out, y):
        # w_ij = w_ij * (1 + learning_rate) if x_ij and y_ij fired:
        # w_ij = w_ij * (1 + learning_rate * x_ij * y_ij)
        self.weight_input.mul_(
            1 + self.learning_rate * y_out.unsqueeze(1) * x.unsqueeze(0))
        if y is not None:
            self.weight_recurrent.mul_(
                1 + self.learning_rate * y_out.unsqueeze(1) * y.unsqueeze(0))

    def project(self, x: torch.Tensor, y: torch.Tensor = None, train=True):
        y_out = x.matmul(self.weight_input.t())
        if y is not None:
            y_out += self.recurrence * y.matmul(self.weight_recurrent.t())
        y_out = k_winners_take_all(y_out, k=K_ACTIVE)
        if train:
            self.update_weights_additive(x=x, y_out=y_out, y=y)
        return y_out

    def complete_pattern(self, y_partial: torch.Tensor):
        y = y_partial.matmul(self.weight_recurrent.t())
        y = k_winners_take_all(y, k=K_ACTIVE)
        return y

    def complete_from_input(self, x_partial: torch.Tensor,
                            y: torch.Tensor = None):
        y_out = x_partial.matmul(self.weight_input.t())
        if y is not None:
            y_out += self.recurrence * y.matmul(self.weight_recurrent.t())
        y_out = k_winners_take_all(y_out, k=K_ACTIVE)
        return y_out


def project(x: torch.Tensor, area_B: Area, y: torch.Tensor = None):
    return area_B.project(x=x, y=y)


def simulate(steps=100, epoch=10):
    area = Area(in_features=N_NEURONS, out_features=N_NEURONS // 2)
    x = sample_k(n=N_NEURONS, k=K_ACTIVE)
    y_prev = None
    # y_prev = torch.ones(area.weight_recurrent.shape[0])
    overlaps = []
    memory_used = []
    for t in trange(steps):
        if t % epoch == 0:
            area.normalize_weights()
        y = area.project(x=x, y=y_prev)
        if y_prev is not None:
            overlap = y.matmul(y_prev)
            # print(overlap)
            overlaps.append(overlap)
        y_prev = y
        memory_used.append(area.memory_used())

    fig, axes = plt.subplots(nrows=2, ncols=1)
    axes[0].plot(overlaps, label='overlap')
    axes[0].set_title("overlap(y, y_prev)")
    axes[0].set_xticks([])
    xmin, xmax = axes[0].get_xlim()
    axes[0].axhline(y=K_ACTIVE, xmin=xmin, xmax=xmax, ls='--', color='black',
                    label='k active')
    axes[0].legend()

    wi, wr = zip(*memory_used)
    axes[1].plot(wi, label='Input')
    axes[1].plot(wr, label='Recurrent')
    axes[1].set_title("Memory used")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$||w||_0$ / n_elements(w)")
    axes[1].legend()
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(19)
    simulate()
