#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np
import numba as nb
from numba import double, jit

def dot3_python(a,b):
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
dot3_numba = jit(double(double[:],double[:]),nopython=True)(dot3_python)

def _hamiltonian_jet_python(strengths, m,dH,ddH):
	"""
	Private function to Compute H and its gradient dH at the point m.
	This function is not meant to be used directly.
	"""
	n = len(strengths)

	H = 0.0
	for i in xrange(n):
		for j in xrange(i):
			## mi_dot_mj = m[3*i]*m[3*j] + m[3*i+1]*m[3*j+1] + m[3*i+2]*m[3*j+2]
			strength = strengths[i] * strengths[j]
			mi_dot_mj = dot3_numba(m[3*i:3*(i+1)],m[3*j:3*(j+1)])
			lij = 1.0-mi_dot_mj
			lijsq = lij**2
			H += strength*np.log(2.0*lij)
			for k in xrange(3):
				dH[3*i+k] -= strength*m[3*j+k]/lij
				dH[3*j+k] -= strength*m[3*i+k]/lij
				ddH[3*i+k,3*i+k] -= m[3*j+k]**2/lijsq
				ddH[3*j+k,3*j+k] -= m[3*i+k]**2/lijsq
				ddH[3*i+k,3*j+k] -= (1.0/lij + m[3*i+k]*m[3*j+k]/lijsq)
				ddH[3*j+k,3*i+k] -= (1.0/lij + m[3*i+k]*m[3*j+k]/lijsq)
	return H
_hamiltonian_jet_numba = jit(double(double[:],double[:],double[:],double[:,:]),nopython=True)(_hamiltonian_jet_python)


def Xham_python(m,dH):
	"""
	Compute X_H from the gradient dH.
	Not used.
	"""
	n = int(m.shape[0]/3)
	for i in xrange(n):
		y0 = m[3*i+1]*dH[3*i+2]-m[3*i+2]*dH[3*i+1]
		y1 = m[3*i+2]*dH[3*i]-m[3*i]*dH[3*i+2]
		y2 = m[3*i]*dH[3*i+1]-m[3*i+1]*dH[3*i]

def get_jet_computer(strengths):
	"""
	Return: a function computing the jet of H at a point m, with the given strenghts.
	"""
	def compute_H_jet(m):
		"""
		Compute H, dH, ddH at the point m
		"""
		n = m.shape[0]//3
		dH = np.zeros(3*n,dtype=float)
		ddH = np.zeros((3*n,3*n),dtype=float)
		H = _hamiltonian_jet_numba(strengths, m, dH, ddH)
		return H, dH, ddH
	return compute_H_jet

def get_gradient(strengths):
	"""
	Return: function computing the gradient of H for the given strengths.
	"""
	comp = get_jet_computer(strengths)
	def grad(m):
		H, dH, ddH = comp(m)
		return dH
	return grad

def get_energy(strengths):
	"""
	Return: function computing the energy H for the given strengths.
	"""
	comp = get_jet_computer(strengths)
	def energy(m):
		H, dH, ddH = comp(m)
		return H
	return energy
