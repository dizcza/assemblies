import torch
from tqdm import tqdm

from mighty.monitor.batch_timer import timer
from nn.areas import AreaRNNHebb, AreaStack, AreasSequential
from nn.constants import K_ACTIVE, N_NEURONS
from nn.monitor import Monitor, expected_random_overlap, pairwise_similarity
from nn.samplers import *


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
