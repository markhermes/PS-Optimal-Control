# Pseudospectral Optimal Control

Trajectory optimization using pseudospectral collocation. Discretizes the
state and control on orthogonal-polynomial (e.g. Legendre-Gauss-Lobatto)
nodes and solves the resulting nonlinear program, a standard approach for
aerospace trajectory and reentry guidance problems.

## What's here
<!-- TODO: confirm/adjust to match the actual files -->
- Pseudospectral discretization of the state and control
- Cost and constraint setup for a representative optimal-control problem
- NLP solve and trajectory reconstruction

## Stack
Python <!-- TODO: list key libs, e.g. NumPy, SciPy, CasADi/IPOPT -->

## Run
```bash
# TODO: add the actual entry point, e.g.
python main.py
```

## Example
<!-- TODO: add a plot of the optimized trajectory (states/controls vs time).
     A single figure here does more than any amount of text. -->

## Background
Related to my work in reentry and atmospheric guidance. See my
[profile](https://github.com/markhermes) for context.
