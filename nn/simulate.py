import torch
from tqdm import tqdm
from functools import partial

from mighty.monitor.batch_timer import timer
from nn.areas import *
from nn.constants import K_ACTIVE, N_NEURONS
from nn.monitor import Monitor, expected_random_overlap, pairwise_similarity
from nn.samplers import *


class Simulator:
    """
    Train and simulate Computation with Assemblies.

    Parameters
    ----------
    model : AreaInterface
        A NN model, consisting of one or more areas.
    epoch_size : int, optional
        Defines the number of simulations to run for each trial sample.
        Each trial sample represent a complete epoch.
        Default: 10
    """
    def __init__(self, model, epoch_size=10):
        self.model = model
        self.monitor = Monitor(model)
        self.epoch_size = epoch_size
        timer.init(epoch_size)

    def simulate(self, x_samples):
        """
        Train and simulate the :attr:`model` with the input `x_samples` data.

        Parameters
        ----------
        x_samples : list of torch.Tensor
            The input stimuli samples list.
        """
        self.monitor.reset()
        self.monitor.viz.log(f"Simulate epoch_start={timer.epoch}")
        for sample_count, x in enumerate(tqdm(x_samples, desc="Projecting"),
                                         start=1):
            y_prev = None  # inhibit the area
            for step in range(self.epoch_size):
                y = self.model(x, y_latent=y_prev)
                if isinstance(self.model, AreaSequential):
                    y, y_prev = y
                else:
                    y_prev = y
                timer.tick()
                self.monitor.trial_finished(x_samples[:sample_count])
            self.model.normalize_weights()
            self.monitor.epoch_finished()

        if len(x_samples) > 1:
            self.monitor.log_assembly_similarity(
                input_similarity=pairwise_similarity(x_samples))

    def associate_benchmark(self, x_samples):
        assert isinstance(self.model, AreaSequential)
        self.monitor.reset()
        mode_saved = self.model.training
        self.model.eval()
        n_parents = len(x_samples[0])
        ys_all = []
        for i in range(n_parents):
            ys = []
            for x_pair in x_samples:
                assert isinstance(x_pair, (list, tuple))
                x_active = [None] * len(x_pair)
                x_active[i] = x_pair[i]
                y, y_latent = self.model(x_active)
                ys.append(y)
            ys = torch.stack(ys)  # (n_samples, n_neurons)
            ys_all.append(ys)
        prod = (ys_all[0] * ys_all[1]).mean(dim=1).mean()
        print(f"Learned assemblies intra-similarity: {prod}")
        self.model.train(mode_saved)


def associate_example(n_samples=1, area_type=AreaRNNHebb):
    n_stim_a, n_stim_b = N_NEURONS, N_NEURONS // 2
    na, nb, nc = N_NEURONS * 2, int(N_NEURONS * 1.5), N_NEURONS
    area_type = partial(area_type, p_synapse=0.05, update='multiplicative',
                        sampler=sample_uniform_masked)
    area_A = area_type(N_NEURONS, out_features=na)
    area_B = area_type(N_NEURONS // 2, out_features=nb)
    area_C = area_type(na, nb, out_features=nc)
    area_AB = AreaStack(area_A, area_B)
    brain = AreaSequential(area_AB, area_C)
    print(brain)
    xa_samples = [sample_k_active(n=n_stim_a, k=K_ACTIVE) for _ in range(n_samples)]
    xb_samples = [sample_k_active(n=n_stim_b, k=K_ACTIVE) for _ in range(n_samples)]
    x_pairs = list(zip(xa_samples, xb_samples))
    simulator = Simulator(model=brain, epoch_size=10)
    simulator.simulate(x_samples=list(zip(xa_samples, [None] * n_samples)))
    simulator.simulate(x_samples=list(zip([None] * n_samples, xb_samples)))
    simulator.simulate(x_samples=x_pairs)
    simulator.associate_benchmark(x_samples=x_pairs)


def simulate_example(n_samples=10):
    area = AreaRNNHebb(N_NEURONS, out_features=N_NEURONS // 2)
    xs = [sample_k_active(n=N_NEURONS, k=K_ACTIVE) for _ in range(n_samples)]
    Simulator(model=area, epoch_size=10).simulate(x_samples=xs)


if __name__ == '__main__':
    torch.manual_seed(19)
    associate_example()
    # simulate_example()
