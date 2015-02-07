# spinsys - Spherical midpoint for spin systems

[![Build Status](https://travis-ci.org/olivierverdier/spinsys.svg?branch=master)](https://travis-ci.org/olivierverdier/spinsys)  [![Coverage Status](https://img.shields.io/coveralls/olivierverdier/spinsys/master.svg)](https://coveralls.io/r/olivierverdier/spinsys?branch=master)

## About

This is a symplectic midpoint solver for *spin systems*. These systems are symplectic differential equations derived from the Lie–Poisson structure of a product of spheres. The paper "[Discrete time Hamiltonian spin systems][1]" gives the details of the setting and the method.

The method and the implementation are a joint work of [R. I. McLachlan][2], [K. Modin][3] and [O. Verdier][4].

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
[1]: http://arxiv.org/abs/1402.3334
[2]: http://www.massey.ac.nz/~rmclachl/
[3]: http://klasmodin.wordpress.com/
[4]: http://olivierverdier.com
