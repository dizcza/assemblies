import torch
from functools import partial
from tqdm import tqdm

from assembly.areas import *
from assembly.constants import K_ACTIVE, N_NEURONS
from assembly.monitor import Monitor, pairwise_similarity
from assembly.samplers import *
from mighty.monitor.batch_timer import timer


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
    def __init__(self, model, epoch_size=10, env_suffix=''):
        self.model = model
        self.monitor = Monitor(model, env_suffix=env_suffix)
        self.epoch_size = epoch_size
        timer.init(epoch_size)

    def simulate(self, x_samples):
        """
        Train and simulate the :attr:`model` with the input `x_samples` data.

        Parameters
        ----------
        x_samples : list
            The input stimuli samples list. Each entry can be either a vector
            tensor or, in case of :class:`AreaStack`, a pair of tensors.
        """
        self.monitor.reset()
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
                self.monitor.trial_finished(x_samples[:sample_count],
                                            step=step)
            self.model.normalize_weights()
            self.monitor.epoch_finished()

        if len(x_samples) > 1:
            self.monitor.update_assembly_similarity(
                input_similarity=pairwise_similarity(x_samples))

    def associate_benchmark(self, x_samples, learned=False):
        """
        Measure `associate` operation overlap between projected assemblies
        from two (or more) parent areas. Each assembly is projected
        individually.

        Parameters
        ----------
        x_samples : list of tuple of torch.Tensor
            The input stimuli samples list. In this case, each entry must be
            a pair of vectors.
        learned : bool, optional
            Have the model areas been associated (True) or not? This flag
            is used in the plots title only.
            Default: False
        """
        assert isinstance(self.model, AreaSequential)
        mode_saved = self.model.training
        self.model.eval()
        n_parents = len(x_samples[0])
        ys_traces = []  # individual traces
        for parent_active in range(n_parents):
            ys = []
            for x_pair in x_samples:
                assert isinstance(x_pair, (list, tuple))
                x_active = [None] * len(x_pair)
                x_active[parent_active] = x_pair[parent_active]
                y, y_latent = self.model(x_active)
                ys.append(y)
            self.monitor.reset()  # we don't need the history
            ys = torch.stack(ys)  # (n_samples, n_neurons)
            ys_traces.append(ys)
        ys_traces = torch.stack(ys_traces, dim=1)  # (S, P, N)

        ys_all_active = torch.stack([self.model(x_pair)[0]
                                     for x_pair in x_samples])
        self.monitor.plot_associated_activations(ys_traces=ys_traces,
                                                 ys_all_active=ys_all_active,
                                                 learned=learned)

        pairwise = ys_traces.bmm(ys_traces.transpose(1, 2))  # (S, P, P)
        ii, jj = torch.triu_indices(row=n_parents, col=n_parents, offset=1)
        similarity = pairwise[:, ii, jj].mean()
        similarity /= K_ACTIVE

        learned_str = "after" if learned else "before"
        self.monitor.viz.log(f"Assemblies inter-similarity {learned_str} "
                             f"learning: {similarity:.3f}")
        if learned:
            self.monitor.viz.scatter(X=[[timer.epoch + 1, similarity]], Y=[1],
                                     win='similarity', name='A-B via C',
                                     opts=self.monitor.viz.opts['similarity'],
                                     update='append')
        self.model.train(mode_saved)


def associate_example(n_samples=5, area_type=AreaRNNHebb):
    n_stim_a, n_stim_b = N_NEURONS, N_NEURONS // 2
    na, nb, nc = N_NEURONS * 2, int(N_NEURONS * 1.5), N_NEURONS
    area_type = partial(area_type, p_synapse=0.05, update='multiplicative',
                        learning_rate=0.1)
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
    simulator.associate_benchmark(x_samples=x_pairs, learned=False)
    simulator.simulate(x_samples=x_pairs)
    simulator.associate_benchmark(x_samples=x_pairs, learned=True)


def simulate_example(n_samples=10):
    area = AreaRNNHebb(N_NEURONS, out_features=N_NEURONS // 2)
    xs = [sample_k_active(n=N_NEURONS, k=K_ACTIVE) for _ in range(n_samples)]
    Simulator(model=area, epoch_size=10).simulate(x_samples=xs)


if __name__ == '__main__':
    torch.manual_seed(19)
    associate_example()
    # simulate_example()
