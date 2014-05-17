#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

def radially_constant(gradH, strengths=1.):
	"""
	Give a radially constant Hamiltonian vector field on (R^3)^n.
	Strengths: array of strengths.
	"""
	def vector_field(mvec):
		mmat = mvec.reshape((-1,3))
		# project the point on the sphere
		mproj = (mmat.T/np.sqrt((mmat*mmat).sum(axis=1))).T
		# compute and reshape the gradient
		dH = gradH(mproj.ravel())
		dHmat = dH.reshape((-1,3))
		# scale with the strengths
		dHscaled = dHmat / np.reshape(strengths, (-1,1))
		# return cross product
		return np.cross(mproj,dHscaled).ravel()
	return vector_field

