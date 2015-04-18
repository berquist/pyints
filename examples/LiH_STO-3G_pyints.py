from __future__ import print_function

from math import pi

import numpy as np

import pyquante2

# from pyints.integrals import obarasaika
from obarasaika.obara_saika import get_overlap
from obarasaika.obara_saika import get_kinetic
from pyints.utils import fact2

# 1. read in the molecular geometry in an appropriate form
# 2. read in the basis set in an appropriate form
# 3. from the desired basis set and molecular geometry, place the basis functions at atomic centers
# 4. create calls to integral code for step 3
# 5. make calls for step 4

### Step 1
with open('water.xyz') as water_file:
    water_lines = water_file.readlines()[2:]

water = pyquante2.geo.molecule.read_xyz_lines(water_lines,
                                              units='Angstrom',
                                              charge=0,
                                              multiplicity=1,
                                              name='water')

# print(water)
del water_lines

### Step 2 and step 3
water_basis = pyquante2.basisset(water, '3-21g')

# print('Basis:')
# print(water_basis)

bfs = water_basis.bfs
nbfs = len(bfs)

### Step 4

## overlap (S) integrals

def norm(pgbf):
    """Return the normalization constant for a primitive Cartesian basis
    function.
    """
    nx, ny, nz = pgbf.powers
    zeta = pgbf.exponent
    p1 = pow(2*zeta/pi, 0.75)
    p2 = pow(4*zeta, (nx+ny+nz)/0.5)
    p3 = pow(fact2(2*nx-1)*fact2(2*ny-1)*fact2(2*nz-1), -0.5)
    return p1*p2*p3

S_unnorm_pyints = np.zeros(shape=(nbfs, nbfs))
S_unnorm_pyquante2 = np.zeros(shape=(nbfs, nbfs))
S_norm_pyints = np.zeros(shape=(nbfs, nbfs))
S_norm_pyquante2 = np.zeros(shape=(nbfs, nbfs))
for mu, cgbf_a in enumerate(bfs):
    ra, la = cgbf_a.origin, list(cgbf_a.powers)
    for nu, cgbf_b in enumerate(bfs):
        rb, lb = cgbf_b.origin, list(cgbf_b.powers)
        s_norm_pyquante2 = pyquante2.ints.one.S(cgbf_a, cgbf_b)
        S_norm_pyquante2[mu, nu] = s_norm_pyquante2
        # loop over primitives in each contracted function
        for pgbf_a, ca in zip(cgbf_a.pgbfs, cgbf_a.coefs):
            za = pgbf_a.exponent
            for pgbf_b, cb in zip(cgbf_b.pgbfs, cgbf_a.coefs):
                zb = pgbf_b.exponent
                s_unnorm_pyints = get_overlap(za, zb, ra, rb, la + lb)
                s_unnorm_pyquante2 = pyquante2.ints.one.overlap(za, la, ra, zb, lb, rb)
                S_unnorm_pyints[mu, nu] += (ca*cb*s_unnorm_pyints)
                S_unnorm_pyquante2[mu, nu] += (ca*cb*s_unnorm_pyquante2)
                S_norm_pyints[mu, nu] += (ca*cb*norm(pgbf_a)*norm(pgbf_b)*s_unnorm_pyints)

## kinetic energy (T) integrals

# # loop over contracted basis functions
# for cgbf_a in bfs:
#     ra, ca = cgbf_a.origin, list(cgbf_a.powers)
#     for cgbf_b in bfs:
#         rb, cb = cgbf_b.origin, list(cgbf_b.powers)
#         # loop over primitives
#         for pgbf_a in zip(cgbf_a.coefs, cgbf_a.pnorms, cgbf_a.pexps):
#             za = pgbf_a[2]
#             for pgbf_b in zip(cgbf_b.coefs, cgbf_b.pnorms, cgbf_b.pexps):
#                 zb = pgbf_b[2]
#                 t = get_kinetic(za, zb, ra, rb, ca + cb)
#                 print(t)

## nuclear attraction (V) integrals

## Coulomb (TEI/ERI) integrals
