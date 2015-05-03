from __future__ import print_function

from math import pi

import numpy as np

import pyquante2

# from pyints.integrals import obarasaika
import obarasaika.obara_saika as os
from pyints.utils import fact2

with open('LiH.xyz') as molfile:
    mollines = molfile.readlines()[2:]

mol = pyquante2.geo.molecule.read_xyz_lines(mollines,
                                            units='Angstrom',
                                            charge=0,
                                            multiplicity=1,
                                            name='LiH')

# print(mol)
del mollines

mol_basis = pyquante2.basisset(mol, 'STO-3G')

# print('Basis:')
# print(mol_basis)

bfs = mol_basis.bfs

## overlap (S) integrals

def overlap(alpha1, lmn1, A, alpha2, lmn2, B):
    za, zb, la, lb, ra, rb = alpha1, alpha2, lmn1, lmn2, A, B
    return os.get_overlap(za, zb, ra, rb, la + lb)

def S(a, b):
    if b.contracted:
        return sum(cb * S(pb, a) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * S(b, pa) for (ca, pa) in a)
    return a.norm * b.norm * overlap(a.exponent, list(a.powers), a.origin,
                                     b.exponent, list(b.powers), b.origin)

def makeS(mol, bfs):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = S(a, b)
    return ints

S_pyints = makeS(mol, bfs)

np.savetxt('pyints.S.txt', S_pyints)

## kinetic energy (T) integrals

def kinetic(alpha1, lmn1, A, alpha2, lmn2, B):
    za, zb, la, lb, ra, rb = alpha1, alpha2, lmn1, lmn2, A, B
    return os.get_kinetic(za, zb, ra, rb, la + lb)

def T(a, b):
    if b.contracted:
        return sum(cb * T(pb, a) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * T(b, pa) for (ca, pa) in a)
    return a.norm * b.norm * kinetic(a.exponent, list(a.powers), a.origin,
                                     b.exponent, list(b.powers), b.origin)

def makeT(mol, bfs):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = T(a, b)
    return ints

T_pyints = makeT(mol, bfs)

np.savetxt('pyints.T.txt', T_pyints)

## nuclear attraction (V) integrals

def nuclear_attraction(alpha1, lmn1, A, alpha2, lmn2, B, C):
    za, zb, la, lb, ra, rb, rc = alpha1, alpha2, lmn1, lmn2, A, B, C
    return os.get_nuclear(za, zb, ra, rb, rc, la + lb)

def V(a, b, C):
    if b.contracted:
        return sum(cb * V(pb, a, C) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * V(b, pa, C) for (ca, pa) in a)
    return a.norm * b.norm * nuclear_attraction(a.exponent, list(a.powers), a.origin,
                                                b.exponent, list(b.powers), b.origin,
                                                C)

def makeV(mol, bfs):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = sum(at.Z * V(a, b, at.r) for at in mol)
    return ints

V_pyints = makeV(mol, bfs)

np.savetxt('pyints.V.txt', V_pyints)

## Cartesian moment (M) integrals

def cartesian_moment(alpha1, lmn1, A, alpha2, lmn2, B, C, order):
    za, zb, la, lb, ra, rb, rc = alpha1, alpha2, lmn1, lmn2, A, B, C
    return os.get_moment(za, zb, ra, rb, rc, la + lb, order)

def M(a, b, C, order):
    if b.contracted:
        return sum(cb * M(pb, a, C, order) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * M(b, pa, C, order) for (ca, pa) in a)
    return a.norm * b.norm * cartesian_moment(a.exponent, list(a.powers), a.origin,
                                              b.exponent, list(b.powers), b.origin,
                                              C, order)

def makeM(mol, bfs, origin, order):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = M(a, b, origin, order)
    return ints

M001_pyints = makeM(mol, bfs, [0.0, 0.0, 0.0], [0, 0, 1])
M002_pyints = makeM(mol, bfs, [0.0, 0.0, 0.0], [0, 0, 2])
M010_pyints = makeM(mol, bfs, [0.0, 0.0, 0.0], [0, 1, 0])
M011_pyints = makeM(mol, bfs, [0.0, 0.0, 0.0], [0, 1, 1])
M020_pyints = makeM(mol, bfs, [0.0, 0.0, 0.0], [0, 2, 0])
M100_pyints = makeM(mol, bfs, [0.0, 0.0, 0.0], [1, 0, 0])
M101_pyints = makeM(mol, bfs, [0.0, 0.0, 0.0], [1, 0, 1])
M110_pyints = makeM(mol, bfs, [0.0, 0.0, 0.0], [1, 1, 0])
M200_pyints = makeM(mol, bfs, [0.0, 0.0, 0.0], [2, 0, 0])

np.savetxt('pyints.M001.txt', M001_pyints)
np.savetxt('pyints.M002.txt', M002_pyints)
np.savetxt('pyints.M010.txt', M010_pyints)
np.savetxt('pyints.M011.txt', M011_pyints)
np.savetxt('pyints.M020.txt', M020_pyints)
np.savetxt('pyints.M100.txt', M100_pyints)
np.savetxt('pyints.M101.txt', M101_pyints)
np.savetxt('pyints.M110.txt', M110_pyints)
np.savetxt('pyints.M200.txt', M200_pyints)

## electric field (E) integrals

def electric_field(alpha1, lmn1, A, alpha2, lmn2, B, C, order):
    pass

def E(a, b, C, order):
    if b.contracted:
        return sum(cb * E(pb, a, C, order) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * E(b, pa, C, order) for (ca, pa) in a)
    return a.norm * b.norm * electric_field(a.exponent, list(a.powers), a.origin,
                                            b.exponent, list(b.powers), b.origin,
                                            C, order)

def makeE(mol, bfs, origin, order):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = E(a, b, origin, order)
    return ints


## angular momentum (L) integrals

def angular_momentum(alpha1, lmn1, A, alpha2, lmn2, B, C):
    pass

def L(a, b, C):
    if b.contracted:
        return sum(cb * L(pb, a, C) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * L(b, pa, C) for (ca, pa) in a)
    return a.norm * b.norm * angular_momentum(a.exponent, list(a.powers), a.origin,
                                              b.exponent, list(b.powers), b.origin,
                                              C)

def makeL(mol, bfs, origin):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = L(a, b, origin)
    return ints


## spin-orbit interaction (J) integrals

def spin_orbit(alpha1, lmn1, A, alpha2, lmn2, B, C):
    pass

def J(a, b, C):
    if b.contracted:
        return sum(cb * J(pb, a, C) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * J(b, pa, C) for (ca, pa) in a)
    return a.norm * b.norm * spin_orbit(a.exponent, list(a.powers), a.origin,
                                        b.exponent, list(b.powers), b.origin,
                                        C)

def makeJ(mol, bfs, origin):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = J(a, b, origin)
    return ints

## spin-orbit interaction (J) integrals from King and Furlani's
## formulation

# template:
# os.get_nuclear(ai, aj, ri, rj, rc, [li[0], li[1], li[2], lj[0], lj[1], lj[2]])

def spin_orbit_KF(alpha1, lmn1, A, alpha2, lmn2, B, C, component):
    ai, aj, li, lj, ri, rj, rc = alpha1, alpha2, lmn1, lmn2, A, B, C
    xi, yi, zi, xj, yj, zj = li[0], li[1], li[2], lj[0], lj[1], lj[2]
    if component == 2:
        return (   xi*yj*os.get_nuclear(ai, aj, ri, rj, rc, [xi-1, yi, zi, xj, yj-1, zj]) \
                -2*ai*yj*os.get_nuclear(ai, aj, ri, rj, rc, [xi+1, yi, zi, xj, yj-1, zj]) \
                -2*aj*xi*os.get_nuclear(ai, aj, ri, rj, rc, [xi-1, yi, zi, xj, yj+1, zj]) \
                +4*ai*aj*os.get_nuclear(ai, aj, ri, rj, rc, [xi+1, yi, zi, xj, yj+1, zj]) \
                  -yi*xj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi-1, zi, xj-1, yj, zj]) \
                +2*ai*xj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi+1, zi, xj-1, yj, zj]) \
                +2*aj*yi*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi-1, zi, xj+1, yj, zj]) \
                -4*ai*aj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi+1, zi, xj+1, yj, zj]))
    if component == 0:
        return (   yi*zj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi-1, zi, xj, yj, zj-1]) \
                -2*aj*yi*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi-1, zi, xj, yj, zj+1]) \
                -2*ai*zj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi+1, zi, xj, yj, zj-1]) \
                +4*ai*aj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi+1, zi, xj, yj, zj+1]) \
                  -zi*yj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi-1, xj, yj-1, zj]) \
                +2*aj*zi*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi-1, xj, yj+1, zj]) \
                +2*ai*yj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi+1, xj, yj-1, zj]) \
                -4*ai*aj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi+1, xj, yj+1, zj]))
    if component == 1:
        return (   zi*xj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi-1, xj-1, yj, zj]) \
                -2*aj*zi*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi-1, xj+1, yj, zj]) \
                -2*ai*xj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi+1, xj-1, yj, zj]) \
                +4*ai*aj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi+1, xj+1, yj, zj]) \
                  -xi*zj*os.get_nuclear(ai, aj, ri, rj, rc, [xi-1, yi, zi, xj, yj, zj-1]) \
                +2*aj*xi*os.get_nuclear(ai, aj, ri, rj, rc, [xi-1, yi, zi, xj, yj, zj+1]) \
                +2*ai*zj*os.get_nuclear(ai, aj, ri, rj, rc, [xi+1, yi, zi, xj, yj, zj-1]) \
                -4*ai*aj*os.get_nuclear(ai, aj, ri, rj, rc, [xi+1, yi, zi, xj, yj, zj+1]))

def J_KF(a, b, C, component):
    if b.contracted:
        return sum(cb * J_KF(pb, a, C, component) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * J_KF(b, pa, C, component) for (ca, pa) in a)
    return a.norm * b.norm * spin_orbit_KF(a.exponent, list(a.powers), a.origin,
                                           b.exponent, list(b.powers), b.origin,
                                           C, component)

def makeJ_KF(mol, bfs):
    nbfs = len(bfs)
    ints_X = np.zeros(shape=(nbfs, nbfs))
    ints_Y = np.zeros(shape=(nbfs, nbfs))
    ints_Z = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints_X[mu, nu] = sum(at.Z * J_KF(a, b, at.r, 0) for at in mol)
            ints_Y[mu, nu] = sum(at.Z * J_KF(a, b, at.r, 1) for at in mol)
            ints_Z[mu, nu] = sum(at.Z * J_KF(a, b, at.r, 2) for at in mol)
    return ints_X, ints_Y, ints_Z

J1X_pyints, J1Y_pyints, J1Z_pyints = makeJ_KF(mol, bfs)

np.savetxt('pyints.J1X.txt', J1X_pyints)
np.savetxt('pyints.J1Y.txt', J1Y_pyints)
np.savetxt('pyints.J1Z.txt', J1Z_pyints)
