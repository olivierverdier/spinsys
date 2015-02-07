#!/usr/bin/env python
# coding: UTF-8


from distutils.core import setup

setup(
	name         = 'spinsys',
	version      = '0.1',
	description  = 'Symplectic spherical midpoint rule for spin systems',
	author = 'Robert I McLachlan, Klas Modin, Olivier Verdier',
	url = 'https://github.com/olivierverdier/homogint',
	license      = 'MIT',
	keywords = ['Math', 'Integrator', 'Symplectic'],
	packages=['spinsys',],
	classifiers = [
	'Development Status :: 4 - Beta',
	'Intended Audience :: Science/Research',
	'License :: OSI Approved :: MIT License',
	'Operating System :: OS Independent',
	'Programming Language :: Python',
	'Topic :: Scientific/Engineering :: Mathematics',
	],
	)
