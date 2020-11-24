from collections import defaultdict, namedtuple

import graphviz

from mighty.utils.common import find_named_layers
from mighty.utils.hooks import get_layers_ordered
from nn.areas import AreaRNN, AreaStack, AreaSequential, KWinnersTakeAll, \
    AreaInterface


class GraphArea:
    def __init__(self, name=None):
        self.graph = graphviz.Digraph(name=name, format='svg',
                                      graph_attr=dict(rankdir='LR',
                                                      style='invisible'),
                                      node_attr=dict(shape='box'))

    def draw_model(self, model: AreaInterface, sample):
        model_mode = model.training
        model.eval()
        ordered = get_layers_ordered(model, sample,
                                     ignore_layers=(AreaSequential,
                                                    KWinnersTakeAll))
        model.train(model_mode)
        ordered_idx = {}
        for i, layer in enumerate(ordered):
            ordered_idx[layer] = i
        for layer in ordered:
            if isinstance(layer, AreaStack):
                for child in layer.children():
                    if isinstance(child, AreaRNN):
                        ordered_idx[child] = ordered_idx[layer]
        clusters = defaultdict(list)
        NamedLayer = namedtuple("NamedLayer", ("name", "layer"))
        for name, layer in find_named_layers(model, layer_class=AreaRNN):
            name = f"{layer.__class__.__name__} '{name}'"
            nl = NamedLayer(name=name, layer=layer)
            clusters[ordered_idx[layer]].append(nl)
        for idx, named_layers in clusters.items():
            with self.graph.subgraph(name=f"cluster_{idx}") as c:
                for nl in named_layers:
                    c.node(nl.name)
        keys = tuple(clusters.keys())
        for source_id, sink_id in zip(keys[:-1], keys[1:]):
            for tail in clusters[source_id]:
                for head in clusters[sink_id]:
                    self.graph.edge(tail_name=tail.name,
                                    head_name=head.name,
                                    label=str(tail.layer.out_features))
        svg = self.graph.pipe(format='svg').decode('utf-8')
        return svg
