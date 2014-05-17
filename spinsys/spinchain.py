#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

def get_gradient(dssq):
	"""
	Produce an energy gradient from a discrete parameter dssq = ds*ds
	"""
	def gradH(m):
		return (np.roll(m,3) + np.roll(m,-3))/dssq
	return gradH

def get_energy(dssq):
	def energy(w):
		"""
		w is supposed to be a Nx2 matix.
		"""
		return np.dot(np.roll(w, 1, axis=0), np.roll(w, -1, axis=0))/dssq
	return energy


