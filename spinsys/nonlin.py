#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

"""
Some nonlinear solvers.
"""

import numpy as np
import numpy.linalg as npl

default_tol = np.sqrt(np.finfo(1.0).eps) # Sqrt of machine epsilon (for default float type).


class StoppingCriteria(object):
	"""
	Class for testing convergence of a FixpointIterator.
	"""
	def __init__(self, tol=default_tol):
		self.tol = tol

	def distance(self, ynew, yold):
		return abs(ynew-yold).sum()

	def __call__(self, k, ynew, yold, fixit):
		return self.distance(ynew, yold) < self.tol

class FixpointIterator(object):
	"""
	Class for solving nonlinear equations by fixpoint type iterations.
	The default implementation uses vanilla fixpoint iterations, 
	but this class can be subclassed to yield Newton type methods also.
	
	Attributes
	----------
	N/A
	
	Methods
	-------
	N/A
	"""

	def __init__(self, F, params=dict(), stopit=None):
		"""
		Constructor. 
		
		Parameters
		----------
		F : function of the form 
				def F(y,p):
					...
			which should return an object of the same type as y (should belong to a vector space).
			It is assumed that y.flat returns an iterator for the elements of y.
			p denotes parameters.
			The equation to be solved is
				y = F(y,p)
			for some given p.
		"""
		self.F = F
		self.stats = dict()
		self.stats['niterations'] = 0
		self.stats['ncalls'] = 0
		self.stats['niterations/call'] = 0

		self.params = params
		if not self.params.has_key('maxit'):
			params['maxit'] = 10
		if not self.params.has_key('minit'):
			params['minit'] = 1
		if stopit is None:
			self.stopit = StoppingCriteria()
		else:
			self.stopit = stopit

	def __call__(self, y, *args, **kwargs):
		"""
		Carry out a maximum of max_steps iterations and return the results.
		
		Parameters
		----------
		y : the iterate

		p : parameter (defaults to None)
		"""
		self.ylast = y.copy()

		for k in range(self.params['maxit']):
			y = self.F(y,*args,**kwargs)
			if self.stopit(k, y, self.ylast, self) and k >= self.params['minit']:
				break
			np.copyto(self.ylast,y)
		else:
			Exception("No convergence after %i iterations."%k)

		self.stats['niterations'] += k
		self.stats['ncalls'] += 1
		self.stats['niterations/call'] = self.stats['niterations']/self.stats['ncalls']
		return y

class NewtonIterator(FixpointIterator):
	"""
	Newton iterator for the nonlinear problems G(y)=0.
	It implements the fixpoint iteration y -= solve(G'(y),G(y)).
	The linear solve algorithm may be provided (it defaults to numpy's solve).
	"""
	def __init__(self, G, Gprime=None, solvefun=npl.solve, params=dict(), stopit=None):
		"""
		Constructor. 
		
		Parameters
		----------
		G : function of the form 
				def F(y,p1,..,pn):
					...
			which should return an object of the same type as y (should belong to a vector space).
			It is assumed that y is a 1d ndarray.
			p1,...,pn denotes parameters.
			The equation to be solved is
				0 = G(y,p1,...,pn)
			for some given p1,..,pn.

		Gprime : function that should return the Jacobian matrix for G.
				 The default Gprime function uses finite differences on G.
				 Gprime is assumed to take the same extra parameters as G.

		solvefun : function to solve linear system dot(A,x)=b.
				   Defaults to numpy's solve.
		"""
		if Gprime is None:
			Gprime = FiniteDifferenceJacobian(G)
		def F(y, *args, **kwargs):
			b = G(y,*args,**kwargs)
			A = Gprime(y,*args,**kwargs)
			y -= solvefun(A,b)
			return y
		super(NewtonIterator, self).__init__(F, params=params, stopit=stopit)
		self.G = G
		self.Gprime = Gprime
		
class FiniteDifferenceJacobian(object):
	"""
	Approximate the Jacobian matrix by finite differences.
	"""
	def __init__(self, G, delta=1e-10):
		super(FiniteDifferenceJacobian, self).__init__()
		self.G = G
		self.delta = delta

	def __call__(self, y, *args, **kwargs):
		self.Gy = self.G(y,*args,**kwargs).copy()
		self.Gmat = np.eye(y.shape[0])
		for ek in self.Gmat:
			ek[:] = (self.G(y+self.delta*ek,*args,**kwargs)-self.Gy)/self.delta
		return self.Gmat.T

