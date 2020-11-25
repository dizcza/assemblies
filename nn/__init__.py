r"""

PyTorch implementation of `project` and `associate` operations [1]_.


The forward pass is defined in two steps:

.. math::
    \begin{cases}
    \bold{y} = \sum_l W^{input}_l \bold{x}_l +
               \alpha W^{recurrent}\bold{y}^{latent}
    \\
    \bold{y} = \text{kWTA}(\bold{y}, k)
    \end{cases}
    :label: forward

where :math:`\bold{x}_l` are input stimuli from an incoming layer :math:`l`
and :math:`\bold{y}^{latent}` are latent activations from a previous step;
:math:`k` is the number of active output neurons (winners).

Assembly areas
--------------
.. autosummary::
    :toctree: toctree/nn

    AreaRNNHebb
    AreaRNNWillshaw
    AreaStack
    AreaSequential


Activation function
-------------------
.. autosummary::
    :toctree: toctree/nn

    KWinnersTakeAll


Train and simulate
------------------
.. autosummary::
    :toctree: toctree/nn

    Simulator


Monitoring
----------
.. autosummary::
    :toctree: toctree/nn

    Monitor
    expected_random_overlap
    pairwise_similarity


References
----------
.. [1] Papadimitriou, C. H., Vempala, S. S., Mitropolsky, D., Collins, M., &
   Maass, W. (2020). Brain computation by assemblies of neurons. Proceedings of
   the National Academy of Sciences.

"""

from .monitor import Monitor, expected_random_overlap, pairwise_similarity
from .areas import AreaRNNHebb, AreaRNNWillshaw, AreaStack, AreaSequential, \
    KWinnersTakeAll
from .simulate import Simulator
