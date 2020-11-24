import torch
from tqdm import tqdm

from mighty.monitor.batch_timer import timer
from nn.areas import AreaRNNHebb, AreaStack, AreaSequential, AreaInterface
from nn.constants import K_ACTIVE, N_NEURONS
from nn.monitor import Monitor, expected_random_overlap, pairwise_similarity
from nn.samplers import *


class Simulator:
    """
    Train and simulate Computation with Assemblies.

    Parameters
    ----------
    area : AreaInterface
        The recurrent NN area to train/simulate.
    epoch_size : int, optional
        Defines the number of simulations to run for each trial sample.
        Each trial sample represent a complete epoch.
        Default: 10
    """
    def __init__(self, area, epoch_size=10):
        self.area = area
        self.monitor = Monitor(area)
        self.epoch_size = epoch_size
        timer.init(epoch_size)

    def simulate(self, x_samples):
        """
        Train and simulate the :attr:`area` with the input `x_samples` data.

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
                y = self.area(x, y_latent=y_prev)
                if isinstance(self.area, AreaSequential):
                    y, y_prev = y
                else:
                    y_prev = y
                timer.tick()
                self.monitor.trial_finished(x_samples[:sample_count])
            self.area.normalize_weights()
            self.monitor.epoch_finished()

        self.monitor.log_assembly_similarity(
            input_similarity=pairwise_similarity(x_samples))
        self.monitor.viz.log(
            f"Expected random overlap: "
            f"{expected_random_overlap(n=N_NEURONS, k=K_ACTIVE):.3f}",
            timestamp=False)


def associate_example(n_samples=10):
    na, nb, nc, n_out = N_NEURONS, 2 * N_NEURONS, N_NEURONS // 2, N_NEURONS
    area_A = AreaRNNHebb(na, out_features=nc)
    area_B = AreaRNNHebb(nb, out_features=nc)
    area_C = AreaRNNHebb(nc, nc, out_features=n_out)
    area_AB = AreaStack(area_A, area_B)
    brain = AreaSequential(area_AB, area_C)
    print(brain)
    xa_samples = [sample_k_active(n=na, k=K_ACTIVE) for _ in range(n_samples)]
    xb_samples = [sample_k_active(n=nb, k=K_ACTIVE) for _ in range(n_samples)]
    x_pairs = list(zip(xa_samples, xb_samples))
    simulator = Simulator(area=brain, epoch_size=10)
    simulator.simulate(x_samples=list(zip(xa_samples, [None] * n_samples)))
    simulator.simulate(x_samples=list(zip([None] * n_samples, xb_samples)))
    simulator.simulate(x_samples=x_pairs)
    # for xa, xb in zip(xa_samples, xb_samples):
    #     y_latent = None
    #     for step in range(epoch_size):
    #         y_out, y_latent = brain(xa, xb, y_latent=y_latent)


def simulate_example(n_samples=10):
    area = AreaRNNHebb(N_NEURONS, out_features=N_NEURONS // 2)
    xs = [sample_k_active(n=N_NEURONS, k=K_ACTIVE) for _ in range(n_samples)]
    Simulator(area=area).simulate(x_samples=xs)


if __name__ == '__main__':
    torch.manual_seed(19)
    associate_example()
    # simulate_example()
