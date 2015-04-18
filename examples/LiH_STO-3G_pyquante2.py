#!/usr/bin/env python

from __future__ import print_function

import numpy as np

import pyquante2

water = pyquante2.molecule([
    (8,  0.0000,   0.0000,   0.1173),
    (1,  0.0000,   0.7572,  -0.4692),
    (1,  0.0000,  -0.7572,  -0.4692)],
                 units='Angstrom',
                 charge=0,
                 multiplicity=1,
                 name='water')

print(water)

water_basis = pyquante2.basisset(water, '3-21g')

print(water_basis)
# print('There are {} basis functions.'.format(len(water_basis)))
print('There are {} shells.'.format(len(water_basis.shells)))
print(water_basis.shells)

solver = pyquante2.uhf(water, water_basis)
solver.converge(tol=1e-10, maxiters=1000)

print(solver)

i1 = solver.i1
# print('Converged overlap matrix:')
# print(i1.S)
# print('Converged kinetic energy matrix:')
# print(i1.T)
# print('Converged potential energy matrix:')
# print(i1.V)

# Save the one-electron integral matrices to disk.
np.savetxt('pyquante2.S.txt', i1.S)
np.savetxt('pyquante2.T.txt', i1.T)
np.savetxt('pyquante2.V.txt', i1.V)

