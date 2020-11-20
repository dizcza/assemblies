# Assembly Calculus

[![Documentation Status](https://readthedocs.org/projects/assemblies/badge/?version=latest)](https://assemblies.readthedocs.io/en/latest/?badge=latest)


The reproducible version of the Assembly Calculus ([Papadimitrioua et al., 2020](https://www.pnas.org/content/pnas/early/2020/06/08/2001893117.full.pdf)) with the minimum invasion of the [original repository](https://github.com/dmitropolsky/assemblies).

### Installation

```
pip install -r requirements.txt
```

### Reproducing the plots

```
python simulations.py
```

Figures will be saved in the `plots` directory.

If you want to reproduce the figures from scratch, delete the `results` folder
before running the simulations.

### NN module

`nn` module contains PyTorch implementation of _project_ and _associate_ operations.

## Original README

This repository contains code for simulating operations in the assembly model of brain computation.
