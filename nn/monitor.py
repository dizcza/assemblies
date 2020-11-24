from collections import defaultdict

import math
import numpy as np
import time
import torch

from mighty.monitor.viz import VisdomMighty
from mighty.utils.common import find_named_layers
from nn.areas import AreaInterface, AreaRNN
from nn.constants import K_ACTIVE


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
    Computes the pairwise similarity of the tensors.

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
    if len(tensors) == 0:
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


class Monitor:
    """
    Monitor the training progress.

    Parameters
    ----------
    area : AreaInterface
        The area or a sequence of areas.
    """

    def __init__(self, area):
        self.area = area
        env_name = f"{time.strftime('%Y.%m.%d')} {area.__class__.__name__}"
        self.viz = VisdomMighty(env=env_name)
        self.viz.close()  # clear previous plots
        self.handles = []
        self.module_name = dict()
        for name, layer in find_named_layers(
                area, layer_class=AreaRNN,
                name_prefix=area.__class__.__name__):
            self.module_name[layer] = name
            handle = layer.register_forward_hook(self._forward_hook)
            self.handles.append(handle)
        self.ys_output = dict()
        self.ys_previous = None
        self.ys_learned = defaultdict(list)

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

    def names_active(self, ys_output=True):
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
        overlaps = {'k-active': K_ACTIVE}
        for name in self.names_active():
            overlaps[name] = self.ys_output[name].matmul(
                self.ys_previous[name]).item()
        overlaps = {name: overlaps.get(name, np.nan)
                    for name in self.module_name.values()}
        names, values = list(zip(*overlaps.items()))
        self.viz.line_update(y=values, opts=dict(
            xlabel='Epoch',
            ylabel='overlap',
            title='convergence (y, y_prev)',
            legend=names,
        ))

    def _update_recall(self, x_samples_learned):
        assert len(self.ys_output) == 0
        ys_learned = {}
        names_active = self.names_active(ys_output=False)
        for name in names_active:
            ys_learned[name] = self.ys_learned[name] + [self.ys_previous[name]]
            assert len(ys_learned[name]) == len(x_samples_learned)
        recall = defaultdict(float)
        recall['k-active'] = K_ACTIVE
        for i, x in enumerate(x_samples_learned):
            # ys_output will be populated via the forward hook
            y_ignored = self.area.recall(x)
            for name in names_active:
                y_predicted = self.ys_output[name]
                y_learned = ys_learned[name][i]
                n_total = len(ys_learned[name])
                recall[name] += (y_predicted * y_learned).sum() / n_total
            self.ys_output.clear()
        recall = {name: recall.get(name, np.nan)
                  for name in self.module_name.values()}
        names, values = list(zip(*recall.items()))
        self.viz.line_update(y=values, opts=dict(
            xlabel='Epoch',
            ylabel='overlap',
            title='recall (y_pred, y_learned)',
            legend=names,
        ))

    def _update_memory_used(self):
        names, values = list(zip(*self.area.memory_used().items()))
        self.viz.line_update(y=values, opts=dict(
            xlabel='Epoch',
            title=r"Memory used (L0 norm)",
            legend=names,
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

    def trial_finished(self, x_samples_learned):
        """
        A sample is being learned callback.

        Parameters
        ----------
        x_samples_learned : list
            A list of learned input vectors to recall. Each entry can be either
            a single vector tensor (one incoming area) or a tuple of tensors
            (multiple incoming areas).
        """
        self._update_convergence()
        self.ys_previous = self.ys_output.copy()
        self.ys_output.clear()
        self._update_recall(x_samples_learned)
        self._update_memory_used()

    def epoch_finished(self):
        """
        A sample is learned callback.
        """
        # ys_output is already cleared up
        for name, y_final in self.ys_previous.items():
            self.ys_learned[name].append(y_final)
        self.ys_previous = None

    def log_assembly_similarity(self, input_similarity=None):
        assembly_similarity = self.assembly_similarity()
        if input_similarity:
            assembly_similarity['input'] = input_similarity
        lines = ["Learned assemblies similarity:"]
        for name, similarity in assembly_similarity.items():
            lines.append(f"--{name}: {similarity:.3f}")
        text = '<br>'.join(lines)
        self.viz.log(text=text, timestamp=False)
