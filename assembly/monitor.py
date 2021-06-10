import math
import time
from collections import defaultdict

import numpy as np
import torch
from mighty.monitor.batch_timer import timer
from mighty.monitor.viz import VisdomMighty
from mighty.utils.common import find_named_layers
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from assembly.areas import AreaInterface, AreaRNN
from assembly.constants import K_ACTIVE, N_NEURONS
from assembly.graph import GraphArea, graphviz_notify_if_not_installed
from assembly.utils import factors_root


def expected_random_overlap(n, k):
    """
    Computes the expected overlap of binomial sampled random vectors.

    Parameters
    ----------
    n : int
        The total number of neurons.
    k : int
        The number of active neurons.

    Returns
    -------
    float
        The expected random overlap.
    """
    p_overlap = [math.comb(k, x) * math.comb(n - k, k - x) / math.comb(n, k)
                 for x in range(k + 1)]
    overlap_expected = np.multiply(p_overlap, range(k + 1)).sum()
    return overlap_expected


def pairwise_similarity(tensors):
    """
    Computes the pairwise similarity overlap of the tensors.

    Parameters
    ----------
    tensors : list of torch.Tensor
        A list of binary vectors. Each entry can be either a single vector
        tensor (one incoming area) or a tuple of tensors (multiple incoming
        areas).

    Returns
    -------
    similarity : float
        The pairwise :math:`L_{0/1}` similarity from 0 to 1.

    """
    tensors = [t for t in tensors if t is not None]
    if len(tensors) <= 1:
        return np.nan
    if not isinstance(tensors[0], torch.Tensor):
        # multiple incoming areas
        sim_areas = list(map(pairwise_similarity, zip(*tensors)))
        sim_areas = np.nanmean(sim_areas)
        return sim_areas
    else:
        tensors = torch.stack(tensors)
    similarity = tensors.matmul(tensors.t())
    n_elements = len(tensors)
    ii, jj = torch.triu_indices(row=n_elements, col=n_elements, offset=1)
    similarity = similarity[ii, jj].mean()
    similarity /= K_ACTIVE
    return similarity


class VisdomBuffered(VisdomMighty):
    def __init__(self, legend_labels, env="main"):
        super().__init__(env=env)
        self.close()  # clear previous plots
        self._legend_labels = tuple(legend_labels)
        legend_labels = self.legend_labels('k-active')
        self.opts = {'recall': dict(
            xlabel='Epoch',
            ylabel='overlap',
            legend=list(legend_labels),
            title='recall (y_pred, y_learned)',
        ), 'convergence': dict(
            xlabel='Epoch',
            ylabel='overlap',
            legend=list(legend_labels),
            title='convergence (y, y_prev)',
        ), 'support': dict(
            xlabel='Epoch',
            ylabel='support',
            legend=list(legend_labels),
            title='support size across epoch trials'
        ), 'similarity': dict(
            xlabel='Epoch',
            ylabel='similarity',
            legend=list(self._legend_labels),
            title="Learned assemblies pairwise similarity overlap",
            markers=True,
            markersize=8,
        )}
        self.data_epoch = defaultdict(list)

    def legend_labels(self, *prepend):
        return prepend + self._legend_labels

    def send_buffered(self):
        legend_labels = self.legend_labels('k-active')
        for win in self.data_epoch.keys():
            data_epoch_finished = {name: np.nan for name in legend_labels}
            self.buffer(data=data_epoch_finished, win=win)
            n_trials = len(self.data_epoch[win])
            y = np.full((n_trials, len(legend_labels)), fill_value=np.nan)
            times, data = zip(*self.data_epoch[win])
            for data_dict, yi in zip(data, y):
                for label, val in data_dict.items():
                    yi[legend_labels.index(label)] = val
            y[:, 0] = K_ACTIVE
            times = np.tile(times, reps=(len(legend_labels), 1)).T
            self.line(Y=y, X=times, win=win, opts=self.opts[win],
                      update='append')
        self.data_epoch.clear()

    def buffer(self, data, win):
        self.data_epoch[win].append((timer.epoch_progress(), data))


class Monitor:
    """
    Monitor the training progress.

    Parameters
    ----------
    model : AreaInterface
        A NN model, consisting of one or more areas.
    """

    def __init__(self, model, env_suffix=''):
        self.model = model

        self.ys_output = dict()
        self.ys_previous = None
        self.ys_learned = defaultdict(list)
        self.support = {}

        self.handles = []
        self.module_name = dict()
        for name, layer in find_named_layers(model, layer_class=AreaRNN):
            self.module_name[layer] = f"{name}-{layer.__class__.__name__}" \
                .lstrip('-')
            handle = layer.register_forward_hook(self._forward_hook)
            self.handles.append(handle)

        env_name = f"{time.strftime('%Y.%m.%d')} " \
                   f"{model.__class__.__name__} assemblies"
        if env_suffix:
            env_name = f"{env_name} {env_suffix}"
        self.viz = VisdomBuffered(legend_labels=self.module_name.values(),
                                  env=env_name)
        self.log_expected_random_overlap()
        self.log_model()
        self.draw_model()
        self.reset()

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

    def reset(self):
        self.ys_output.clear()
        self.ys_previous = None
        self.ys_learned.clear()
        self.support.clear()

    def _names_active(self, ys_output=True):
        assert sorted(self.ys_previous.keys()) == sorted(
            self.module_name.values())
        if ys_output:
            assert sorted(self.ys_previous.keys()) == sorted(
                self.ys_output.keys())
        names_active = []
        for name in self.module_name.values():
            active = self.ys_previous[name] is not None
            if ys_output:
                active &= self.ys_output[name] is not None
            if active:
                names_active.append(name)
        return tuple(names_active)

    def _update_convergence(self):
        if self.ys_previous is None:
            return
        overlaps = {}
        for name in self._names_active():
            overlaps[name] = self.ys_output[name].matmul(
                self.ys_previous[name]).item()
        self.viz.buffer(data=overlaps, win='convergence')

    def _update_support(self):
        support = {}
        for name, y in self.ys_output.items():
            if y is None:
                continue
            if name not in self.support:
                self.support[name] = y.clone().bool()
            self.support[name] |= y.bool()
            support[name] = self.support[name].sum()
        self.viz.buffer(data=support, win='support')

    def _update_recall(self, x_samples_learned):
        assert len(self.ys_output) == 0
        ys_learned = {}
        names_active = self._names_active(ys_output=False)
        for name in names_active:
            ys_learned[name] = self.ys_learned[name] + [self.ys_previous[name]]
            assert len(ys_learned[name]) == len(x_samples_learned)
        recall = defaultdict(float)
        for i, x in enumerate(x_samples_learned):
            # ys_output will be populated via the forward hook
            y_ignored = self.model.recall(x)
            for name in names_active:
                y_predicted = self.ys_output[name]
                y_learned = ys_learned[name][i]
                n_total = len(ys_learned[name])
                recall[name] += (y_predicted * y_learned).sum() / n_total
            self.ys_output.clear()
        self.viz.buffer(data=recall, win='recall')

    def update_memory_used(self):
        """
        Plot the weights histograms.
        """
        names, values = list(zip(*self.model.memory_used().items()))
        self.viz.line_update(y=values, opts=dict(
            xlabel='Epoch',
            title=r"Memory used (L0 norm)",
            legend=list(names),
        ))

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
            y_learned = [y for y in y_learned if y is not None]
            if len(y_learned) > 0:
                similarity[name] = pairwise_similarity(y_learned)
        return similarity

    def trial_finished(self, x_samples_learned, step):
        """
        A sample is being learned callback.

        Parameters
        ----------
        x_samples_learned : list
            A list of learned input vectors to recall. Each entry can be either
            a single vector tensor (one incoming area) or a tuple of tensors
            (multiple incoming areas).
        step : int
            The batch ID.
        """
        self._update_convergence()
        self._update_support()
        self.ys_previous = self.ys_output.copy()
        self.ys_output.clear()
        if step != 0:
            # skip the first batch since the recall is exactly 'k'
            self._update_recall(x_samples_learned)

    def epoch_finished(self):
        """
        A sample is learned callback.
        """
        # ys_output is already cleared up
        for name, y_final in self.ys_previous.items():
            self.ys_learned[name].append(y_final)
        self.ys_previous = None
        self.support.clear()
        self.update_memory_used()
        self.update_weight_histogram()
        self.viz.send_buffered()

    def log_expected_random_overlap(self, n=N_NEURONS, k=K_ACTIVE):
        r"""
        Log the expected random overlap between samples, drawn from
        `Binomial(n, k)` distribution.

        Parameters
        ----------
        n : int, optional
            The number of neurons in a layer.
            Default: 1000
        k : int, optional
            The number of active neurons in a layer.
            Default: 50

        """
        self.viz.log(f"Expected random overlap (n={n}, k={k}): "
                     f"{expected_random_overlap(n=n, k=k):.3f}")

    def update_assembly_similarity(self, input_similarity=None, log=False):
        r"""
        Plot the :math:`L_{0/1}` similarity of the projected (learned)
        assemblies.

        The similarity of two binary vectors :math:`\bm{x}` and
        :math:`\bm{y}` of size `n` that have `k` active neurons is computed
        as their dot product, divided by `k`:

        .. math::
            \frac{\bm{x} \cdot \bm{y}}{k}

        Parameters
        ----------
        input_similarity : float or None, optional
            If given, plot the input vectors similarity as well.
            Default: None
        log : bool, optional
            If True, log the similarities as text.
            Default: False

        """
        assembly_similarity = self.assembly_similarity()
        if input_similarity is not None:
            assembly_similarity['input'] = input_similarity
        legend = self.viz.opts['similarity']['legend']
        similarity_nans = [assembly_similarity.get(name, np.nan)
                           for name in legend]
        x = np.stack([[timer.epoch] * len(legend), similarity_nans], axis=1)
        self.viz.scatter(X=x, Y=range(1, len(legend) + 1),
                         opts=self.viz.opts['similarity'], win='similarity',
                         update='append')

        if log:
            lines = ["Learned assemblies intra-similarity:"]
            for name, similarity in assembly_similarity.items():
                lines.append(f"--{name}: {similarity:.3f}")
            text = '<br>'.join(lines)
            self.viz.log(text=text)

    def plot_associated_activations(self, ys_traces, ys_all_active,
                                    learned=False):
        """
        Plot the output layer activations for each single trace (active input
        area). Overlapping neurons are shown in green.

        Parameters
        ----------
        ys_traces : (S, P, N) torch.Tensor
            Stacked individual traces (output layer `C` activations). Axes:
                1) `S` - the number of (learned) samples;

                2) `P` - the number of incoming areas (parents);

                3) `N` - the number of output neurons.
        ys_all_active : (S, N) torch.Tensor
            The output activations of layer `C` with all input areas active
            (have input).
        learned : bool, optional
            Have the model areas been associated (True) or not? This flag
            is used in the plots title only.
            Default: False
        """
        learned_str = "after" if learned else "before"
        rgb_green = torch.tensor([0, 1, 0], dtype=torch.float)

        # (S, P+1, N)
        images = torch.cat([ys_traces, ys_all_active.unsqueeze(dim=1)], dim=1)
        row_size = images.shape[1]
        # (S, P+1, 3, N)
        images = images.unsqueeze(dim=2).repeat_interleave(3, dim=2)
        mask_overlap = ys_traces.prod(dim=1)  # (S, N)
        ii_overlap, jj_overlap = mask_overlap.nonzero(as_tuple=True)
        images[ii_overlap, :-1, :, jj_overlap] = rgb_green
        mask_overlap *= ys_all_active
        ii_overlap_all, jj_overlap_all = mask_overlap.nonzero(as_tuple=True)
        images[ii_overlap_all, -1, :, jj_overlap_all] = rgb_green
        # (S * (P+1), 3, H, N/H)
        images = images.view(-1, 3, *factors_root(images.shape[-1]))
        resize = transforms.Resize(size=128,
                                   interpolation=InterpolationMode.NEAREST)
        images = resize(images)  # (S * (P+1), 3, ~128, ~128)
        title = f"{learned_str}: A only | B only | all"
        self.viz.images(images,
                        win=f"activations {learned_str}",
                        nrow=row_size,
                        opts=dict(title=title))

    def log_model(self, space='-'):
        """
        Log the :attr:`model`.

        Parameters
        ----------
        space : str, optional
            A space substitution to correctly parse HTML later on.
            Default: '-'
        """
        lines = ["Area model:"]
        for line in repr(self.model).splitlines():
            n_spaces = len(line) - len(line.lstrip())
            line = space * n_spaces + line
            lines.append(line)
        lines = '<br>'.join(lines)
        self.viz.log(lines)

    @graphviz_notify_if_not_installed
    def draw_model(self, sample=None):
        """
        Draw the model graph.

        Parameters
        ----------
        sample : torch.Tensor or None, optional
            Input sample.
            Default: None
        """
        graph = GraphArea()
        svg = graph.draw_model(self.model, sample=sample)
        self.viz.svg(svgstr=svg, win='graph')

    def update_weight_histogram(self):
        """
        Plot the model weights histogram.
        """
        for name, param in self.model.named_parameters():
            self.viz.histogram(X=param.data.view(-1), win=name, opts=dict(
                xlabel='Weight values',
                ylabel='count',
                title=f"{name} histogram",
                ytype='log',
            ))
