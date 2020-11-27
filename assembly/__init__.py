r"""

The forward pass is defined in two steps:

.. math::
    \begin{cases}
    \bm{y} = \sum_l W^{input}_l \bm{x}_l +
               \alpha W^{recurrent}\bm{y}^{latent}
    \\
    \bm{y} = \text{kWTA}(\bm{y}, k)
    \end{cases}
    :label: forward

where :math:`\bm{x}_l` are input stimuli from an incoming layer :math:`l`
and :math:`\bm{y}^{latent}` are latent activations from a previous step;
:math:`k` is the number of active output neurons (winners).

Assembly areas
--------------
.. autosummary::
    :toctree: toctree/assembly

    AreaRNNHebb
    AreaRNNWillshaw
    AreaStack
    AreaSequential


Activation function
-------------------
.. autosummary::
    :toctree: toctree/assembly

    KWinnersTakeAll


Train and simulate
------------------
.. autosummary::
    :toctree: toctree/assembly

    Simulator


Monitoring
----------
.. autosummary::
    :toctree: toctree/assembly

    Monitor
    expected_random_overlap
    pairwise_similarity

"""

from .monitor import Monitor, expected_random_overlap, pairwise_similarity
from .areas import AreaRNNHebb, AreaRNNWillshaw, AreaStack, AreaSequential, \
    KWinnersTakeAll
from .simulate import Simulator
