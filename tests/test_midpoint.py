#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np
import numpy.testing as npt

import unittest
from spinsys import midpoint, sphere, spinchain, vortex

def vec(x):
	return -x

class TestMidpoint(unittest.TestCase):
	def test_increment(self):
		m = np.array([1.,2.])
		dm = np.zeros_like(m)
		idn = np.eye(np.shape(m)[0])
		increment = midpoint.get_increment(vec)
		dm = increment(m, dm, idn, dt=0)
		npt.assert_allclose(dm, 0.)

	def test_run(self):
		increment = midpoint.get_increment(vec)
		m0 = np.array([1.,2.])
		nb_steps = 10
		res = list(midpoint.run(increment, m0, .1, nb_steps))
		self.assertEqual(len(res), nb_steps+1)
		
class TestSphericalMidpoint(unittest.TestCase):
	def test_radially_constant(self):
		grad = spinchain.get_gradient(dssq=.1)
		vec = sphere.radially_constant(grad)
		dir1 = np.array([1.,0,0])
		dir2 = np.array([0,1.,0])
		theta1 = .5
		theta2 = .6
		on_sphere = np.hstack([dir1, dir2])
		scaled = np.hstack([theta1*dir1, theta2*dir2])
		npt.assert_allclose(vec(scaled), vec(on_sphere))

class TestSpinChain(unittest.TestCase):
	def test_spin_chain(self, n=5):
		svec = np.linspace(0, 2*np.pi, n, endpoint=False)
		#{'ittol':1e-12}
		dssq = (svec[1]-svec[0])**2
		grad = spinchain.get_gradient(dssq)
		vec = sphere.radially_constant(grad)
		energy = spinchain.get_energy(dssq)
		energy(np.random.rand(n*4))

class TestVortex(unittest.TestCase):

	def test_dotnumba(self):
		v = np.array([3.,4.,5.])
		w = np.array([3.,4.,5.])
		npt.assert_allclose(vortex.dot3_numba(v,w), np.dot(v,w))

	def test_vortex(self):
		n = 5
		strengths = np.ones(n)
		grad = vortex.get_gradient(strengths)
		vec = sphere.radially_constant(grad, strengths)
		m = np.random.rand(n*3)
		vec(m)

	def test_energy(self):
		npts = 12
		energy = vortex.get_energy(np.ones(npts))
		t = np.linspace(0, 2*np.pi, npts, endpoint=False)
		pert = 1e-1
		initial_data = np.array([np.cos(t), np.sin(t), pert*np.cos(6*t)]).T
		energy(initial_data.ravel())

	def test_radconstant(self):
		npts = 12
		t = np.linspace(0, 2*np.pi, npts, endpoint=False)
		pert = 1e-1
		initial_data = np.array([np.cos(t), np.sin(t), pert*np.cos(6*t)]).T
		#x=cos(t);y=sin(t);z=1e-6*cos(6*t);
		comp = vortex.get_jet_computer(np.ones(npts))

		def vor_grad(m):
			H,dH,ddH = comp(m)
			return dH
		vor_vec = sphere.radially_constant(vor_grad)
		ws = np.array([[-0.0056248 ,  0.5240858 ,  0.85195627],
       [-0.01637811,  0.86613904, -0.65457924],
       [ 0.28757213, -0.94335405, -0.64858281],
       [ 0.46958492, -0.81897607, -0.33014262],
       [-0.29878549, -0.26583188,  0.99412347],
       [ 0.14619857,  0.98105382, -0.1490143 ],
       [-0.56415813, -0.8125091 , -0.1468523 ],
       [-0.15011813, -0.32735439, -0.93290806],
       [ 0.02668312, -0.01198654, -0.99957457],
       [ 0.27531609,  0.36315652,  0.89012602],
       [-0.83458294, -0.03146331,  0.54998707],
       [ 0.66429157,  0.47703894,  0.57546045]])
		vec = vor_vec(ws.ravel())
		vs = vec.reshape(-1,3)
		for (v,w) in zip(vs,ws):
			npt.assert_allclose(np.dot(v,w), 0., atol=1e-15)




