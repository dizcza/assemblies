.. _usage:

Usage
=====

.. automodule:: nn


Example
-------

Associate script:

.. code-block:: python

    from nn.areas import *
    from nn.samplers import sample_k_active
    from nn.simulate import Simulator

    N_NEURONS, K_ACTIVE = 1000, 50

    n_stim_a, n_stim_b = N_NEURONS, N_NEURONS // 2
    na, nb, nc = N_NEURONS * 2, int(N_NEURONS * 1.5), N_NEURONS
    area_A = AreaRNNHebb(N_NEURONS, out_features=na)
    area_B = AreaRNNHebb(N_NEURONS // 2, out_features=nb)
    area_C = AreaRNNHebb(na, nb, out_features=nc)
    area_AB = AreaStack(area_A, area_B)
    brain = AreaSequential(area_AB, area_C)
    stim_a = sample_k_active(n=n_stim_a, k=K_ACTIVE)
    stim_b = sample_k_active(n=n_stim_b, k=K_ACTIVE)
    stim_ab = (stim_a, stim_b)
    simulator = Simulator(model=brain, epoch_size=10)
    simulator.simulate(x_samples=[stim_ab])


More examples are in `nn/simulate.py <https://github.com/dizcza/assemblies/blob/master/nn/simulate.py>`_
