#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

from . import nonlin


def mid_G(dm, m0, h, X, Xprime, idn):
	return dm - h*X(m0 + dm/2)

def mid_Gprime(dm, m0, h, X, Xprime, idn):
	return idn - h/2*Xprime(m0 + dm/2)

def get_increment(X, Xprime=None):
	"""
	General midpoint method.
	X: vector field
	"""
	if Xprime is None:
		Xprime = nonlin.FiniteDifferenceJacobian(X)
	newtonit = nonlin.NewtonIterator(mid_G, mid_Gprime)

	def increment(m, dm, idn, dt):
		dm = newtonit(dm, m, dt, X, Xprime, idn)
		return dm

	return increment

def run(increment, m0, dt, N=1):
	"""
	Run the one-step method `increment`
	starting from an initial condition m0,
	and a time step `dt`.
	"""
	m = m0.copy()
	yield m.copy()
	dm = np.zeros_like(m)
	idn = np.eye(np.shape(m)[0])
	for iteration in range(N):
		dm = increment(m, dm, idn, dt=dt)
		m += dm
		yield m.copy()
