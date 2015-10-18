# spinsys - Spherical midpoint for spin systems

[![Build Status](https://img.shields.io/travis/olivierverdier/spinsys/master.svg?style=flat-square)](https://travis-ci.org/olivierverdier/spinsys)
[![Coverage Status](https://img.shields.io/coveralls/olivierverdier/spinsys/master.svg?style=flat-square)](https://coveralls.io/r/olivierverdier/spinsys?branch=master)
![Python version](https://img.shields.io/badge/python-2.7, 3.4-blue.svg?style=flat-square)

## About

This is a symplectic midpoint solver for *spin systems*. These systems are symplectic differential equations derived from the Lie–Poisson structure of a product of spheres. The papers "[Symplectic integrators for spin systems](http://arxiv.org/abs/1402.4114)", "[A minimal-coordinate symplectic integrator on spheres](http://arxiv.org/abs/1402.3334)" and "[Geometry of discrete-time spin systems](http://arxiv.org/abs/1505.04035)" gives the details of the setting and the method, as well as a theoretical background.

The method and the implementation are a joint work of [R. I. McLachlan](https://www.massey.ac.nz/~rmclachl/), [K. Modin](https://klasmodin.wordpress.com/) and [O. Verdier](https://olivierverdier.com).

## How to use it

### What you need

 * `m0`: An initial condition: an Nx3 matrix where each row is a vector of length one.
 * `gradient`: A gradient which returns the gradient of a Hamiltonian at each such points.

You may also optionally need
 * `strengths`: A N vector of strength which determines the symplectic form on the product of spheres.

If such a `strenghts` vector is not provided, the strengths are assumed to be one for every sphere.

### Run the simulation

The code is then as follows:

```python
import midpoint
import sphere

dt = .01 # time step
nb_steps = 10 # number of steps

vector = sphere.radially_constant(gradient, strengths)
generator = midpoint.run(midpoint.get_increment(vector), m0=m0.ravel(), dt=dt, nb_steps=nb_steps)

# run the solver
ms = [m.reshape(-1,3) for m in generator]
```
Now the solution `ms` is a list of Nx3 matrices which correspond to the solutions at each time step.

## Available systems

There are two common spin systems which are already available in this package.

### Spin chain

The Heisenberg spin chain system for N points is initialized as follows:
```python
from spinsys import spinchain
gradient = spinchain.get_gradient(1/N**2)
```
The strength vector may be omitted in the call of `radially_constant`:
```python
vector = sphere.radially_constant(gradient)
```

### Point vortices

The gradient is obtained from the `strengths` vector:
```python
from spinsys import vortex
import numpy as np

strengths = np.array([1., 1., -1.])
gradient = vortex.get_gradient(strengths)

```
