#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

def initial_curve(t, frequency=3, amplitude=0.3, mean=0.7, perturbation=0.05):
	"""
	A nice chain of points on the sphere.
	Frequency: frequency of the oscillations
	Amplitude: angle amplitude
	Mean: angle mean
	Perturbation: amplitude of added perturbation
	"""
	tmp = mean + amplitude*np.sin(frequency*t) + perturbation*np.cos(3*frequency*t)
	return np.array([np.cos(t)*np.sin(tmp),np.sin(t)*np.sin(tmp),np.cos(tmp)]).T

