from __future__ import print_function

from itertools import combinations_with_replacement

import numpy as np

import pyquante2

# from pyints.integrals import obarasaika
import obarasaika.obara_saika as os
from pyints.utils import iterator4
from pyints.utils import fact2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--S', action='store_true')
parser.add_argument('--T', action='store_true')
parser.add_argument('--V', action='store_true')
parser.add_argument('--M', action='store_true')
parser.add_argument('--L', action='store_true')
parser.add_argument('--L_from_M', action='store_true')
parser.add_argument('--E', action='store_true')
parser.add_argument('--EF', action='store_true')
parser.add_argument('--EFG', action='store_true')
parser.add_argument('--J', action='store_true')
parser.add_argument('--J_KF', action='store_true')
parser.add_argument('--ERI', action='store_true')
parser.add_argument('--J2_KF', action='store_true')
args = parser.parse_args()

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

def makeS(bfs):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = S(a, b)
    return ints

if args.S:
    print('making S...')
    S_pyints = makeS(bfs)
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

def makeT(bfs):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = T(a, b)
    return ints

if args.T:
    print('making T...')
    T_pyints = makeT(bfs)
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

if args.V:
    print('making V...')
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

def makeM(bfs, origin, order):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = M(a, b, origin, order)
    return ints

if args.M:
    print('making M...')
    M001_pyints = makeM(bfs, [0.0, 0.0, 0.0], [0, 0, 1])
    M002_pyints = makeM(bfs, [0.0, 0.0, 0.0], [0, 0, 2])
    M010_pyints = makeM(bfs, [0.0, 0.0, 0.0], [0, 1, 0])
    M011_pyints = makeM(bfs, [0.0, 0.0, 0.0], [0, 1, 1])
    M020_pyints = makeM(bfs, [0.0, 0.0, 0.0], [0, 2, 0])
    M100_pyints = makeM(bfs, [0.0, 0.0, 0.0], [1, 0, 0])
    M101_pyints = makeM(bfs, [0.0, 0.0, 0.0], [1, 0, 1])
    M110_pyints = makeM(bfs, [0.0, 0.0, 0.0], [1, 1, 0])
    M200_pyints = makeM(bfs, [0.0, 0.0, 0.0], [2, 0, 0])
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

def makeE(bfs, origin, order):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = E(a, b, origin, order)
    return ints

if args.E:
    print('making E...')

## electric field (EF) integrals from nuclear attraction integrals

# def electric_field_from_V(alpha1, lmn1, A, alpha2, lmn2, B, C, component):
def electric_field_from_V(alpha1, alpha2, lmn1, lmn2, A, B, C, component):
    ai, aj, li, lj, ri, rj, rc = alpha1, alpha2, lmn1, lmn2, A, B, C
    xi, yi, zi, xj, yj, zj = li[0], li[1], li[2], lj[0], lj[1], lj[2]
    if component == 0:
        return (   xi*os.get_nuclear(ai, aj, ri, rj, rc, [xi-1, yi, zi, xj, yj, zj]) \
                -2*ai*os.get_nuclear(ai, aj, ri, rj, rc, [xi+1, yi, zi, xj, yj, zj]) \
                  +xj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi, xj-1, yj, zj]) \
                -2*aj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi, xj+1, yj, zj]))
    if component == 1:
        return (   yi*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi-1, zi, xj, yj, zj]) \
                -2*ai*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi+1, zi, xj, yj, zj]) \
                  +yj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj-1, zj]) \
                -2*aj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj+1, zj]))
    if component == 2:
        return (   zi*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi-1, xj, yj, zj]) \
                -2*ai*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi+1, xj, yj, zj]) \
                  +zj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj, zj-1]) \
                -2*aj*os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj, zj+1]))

def EF(a, b, C, component):
    if b.contracted:
        return sum(cb * EF(pb, a, C, component) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * EF(b, pa, C, component) for (ca, pa) in a)
    return a.norm * b.norm * electric_field_from_V(a.exponent, b.exponent,
                                                   list(a.powers), list(b.powers),
                                                   a.origin, b.origin,
                                                   C, component)

def makeEF(bfs, origin, component):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = EF(a, b, origin, component)
    return ints

if args.EF:
    print('making EF...')
    counter = 1
    for at in mol:
        for component in range(3):
            EF_pyints = makeEF(bfs, at.r, component)
            np.savetxt('pyints.EF{}.txt'.format(counter), EF_pyints)
            counter += 1

## electric field gradient (EFG) integrals from electric field integrals

def electric_field_gradient_from_EF(alpha1, lmn1, A, alpha2, lmn2, B, C, component1, component2):
    ai, aj, li, lj, ri, rj, rc = alpha1, alpha2, lmn1, lmn2, A, B, C
    xi, yi, zi, xj, yj, zj = li[0], li[1], li[2], lj[0], lj[1], lj[2]
    if component2 == 0:
        return (   xi*electric_field_from_V(ai, aj, [xi-1, yi, zi], [xj, yj, zj], ri, rj, rc, component1) \
                -2*ai*electric_field_from_V(ai, aj, [xi+1, yi, zi], [xj, yj, zj], ri, rj, rc, component1) \
                  +xj*electric_field_from_V(ai, aj, [xi, yi, zi], [xj-1, yj, zj], ri, rj, rc, component1) \
                -2*aj*electric_field_from_V(ai, aj, [xi, yi, zi], [xj+1, yj, zj], ri, rj, rc, component1))
    if component2 == 1:
        return (   yi*electric_field_from_V(ai, aj, [xi, yi-1, zi], [xj, yj, zj], ri, rj, rc, component1) \
                -2*ai*electric_field_from_V(ai, aj, [xi, yi+1, zi], [xj, yj, zj], ri, rj, rc, component1) \
                  +yj*electric_field_from_V(ai, aj, [xi, yi, zi], [xj, yj-1, zj], ri, rj, rc, component1) \
                -2*aj*electric_field_from_V(ai, aj, [xi, yi, zi], [xj, yj+1, zj], ri, rj, rc, component1))
    if component2 == 2:
        return (   zi*electric_field_from_V(ai, aj, [xi, yi, zi-1], [xj, yj, zj], ri, rj, rc, component1) \
                -2*ai*electric_field_from_V(ai, aj, [xi, yi, zi+1], [xj, yj, zj], ri, rj, rc, component1) \
                  +zj*electric_field_from_V(ai, aj, [xi, yi, zi], [xj, yj, zj-1], ri, rj, rc, component1) \
                -2*aj*electric_field_from_V(ai, aj, [xi, yi, zi], [xj, yj, zj+1], ri, rj, rc, component1))


def EFG(a, b, C, component1, component2):
    if b.contracted:
        return sum(cb * EFG(pb, a, C, component1, component2) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * EFG(b, pa, C, component1, component2) for (ca, pa) in a)
    return a.norm * b.norm * electric_field_gradient_from_EF(a.exponent, list(a.powers), a.origin,
                                                             b.exponent, list(b.powers), b.origin,
                                                             C, component1, component2)

def makeEFG(bfs, origin, component1, component2):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = EFG(a, b, origin, component1, component2)
    return ints

component_map = {
    'X': 0,
    'Y': 1,
    'Z': 2,
}

if args.EFG:
    print('making EFG...')
    for iat, at in enumerate(mol, 1):
        for (c1, c2) in combinations_with_replacement(('X', 'Y', 'Z'), 2):
                EFG_pyints = makeEFG(bfs, at.r, component_map[c1], component_map[c2])
                np.savetxt('pyints.EFG{}{}{}.txt'.format(iat, c1, c2), EFG_pyints)

## angular momentum (L) integrals

def angular_momentum(alpha1, lmn1, A, alpha2, lmn2, B, C, component):
    za, zb, la, lb, ra, rb, rc = alpha1, alpha2, lmn1, lmn2, A, B, C
    return os.get_angmom(za, zb, ra, rb, rc, la + lb, component)

def L(a, b, C, component):
    if b.contracted:
        return sum(cb * L(pb, a, C, component) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * L(b, pa, C, component) for (ca, pa) in a)
    return a.norm * b.norm * angular_momentum(a.exponent, list(a.powers), a.origin,
                                              b.exponent, list(b.powers), b.origin,
                                              C, component)

def makeL(bfs, origin, component):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = L(a, b, origin, component)
    return ints

if args.L:
    print('making L...')
    LX_pyints = makeL(bfs, [0.0, 0.0, 0.0], 0)
    LY_pyints = makeL(bfs, [0.0, 0.0, 0.0], 1)
    LZ_pyints = makeL(bfs, [0.0, 0.0, 0.0], 2)
    np.savetxt('pyints.LX.txt', LX_pyints)
    np.savetxt('pyints.LY.txt', LY_pyints)
    np.savetxt('pyints.LZ.txt', LZ_pyints)

## angular momentum (L) integrals from first moment integrals

def angular_momentum_from_M(alpha1, lmn1, A, alpha2, lmn2, B, C, component):
    ai, aj, li, lj, ri, rj, rc = alpha1, alpha2, lmn1, lmn2, A, B, C
    xi, yi, zi, xj, yj, zj = li[0], li[1], li[2], lj[0], lj[1], lj[2]
    if component == 2:
        # Lz = xpy - ypx
        return ((yj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj-1, zj], [1, 0, 0]) \
                 -2*aj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj+1, zj], [1, 0, 0])) \
                -(xj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj-1, yj, zj], [0, 1, 0]) \
                  -2*aj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj+1, yj, zj], [0, 1, 0])))
    if component == 0:
        # Lx = ypz - zpy
        return ((zj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj, zj-1], [0, 1, 0]) \
                 -2*aj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj, zj+1], [0, 1, 0])) \
                -(yj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj-1, zj], [0, 0, 1]) \
                  -2*aj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj+1, zj], [0, 0, 1])))
    if component == 1:
        # Ly = zpx - xpz
        return ((xj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj-1, yj, zj], [0, 0, 1]) \
                 -2*aj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj+1, yj, zj], [0, 0, 1])) \
                -(zj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj, zj-1], [1, 0, 0]) \
                  -2*aj*os.get_moment(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj, zj+1], [1, 0, 0])))

def L_from_M(a, b, C, component):
    if b.contracted:
        return sum(cb * L_from_M(a, pb, C, component) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * L_from_M(pa, b, C, component) for (ca, pa) in a)
    return a.norm * b.norm * angular_momentum_from_M(a.exponent, list(a.powers), a.origin,
                                                     b.exponent, list(b.powers), b.origin,
                                                     C, component)

def makeL_from_M(bfs, origin, component):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = L_from_M(a, b, origin, component)
    return ints

if args.L_from_M:
    print('making L_from_M...')
    LX_M_pyints = makeL_from_M(bfs, [0.0, 0.0, 0.0], 0)
    LY_M_pyints = makeL_from_M(bfs, [0.0, 0.0, 0.0], 1)
    LZ_M_pyints = makeL_from_M(bfs, [0.0, 0.0, 0.0], 2)
    np.savetxt('pyints.LX_M.txt', LX_M_pyints)
    np.savetxt('pyints.LY_M.txt', LY_M_pyints)
    np.savetxt('pyints.LZ_M.txt', LZ_M_pyints)

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

if args.J:
    print('making J...')

## spin-orbit interaction (J) integrals from King and Furlani's
## formulation

# template:
# os.get_nuclear(ai, aj, ri, rj, rc, [xi, yi, zi, xj, yj, zj])

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

if args.J_KF:
    print('making J_KF...')
    J1X_pyints, J1Y_pyints, J1Z_pyints = makeJ_KF(mol, bfs)
    np.savetxt('pyints.J1X.txt', J1X_pyints)
    np.savetxt('pyints.J1Y.txt', J1Y_pyints)
    np.savetxt('pyints.J1Z.txt', J1Z_pyints)

## two-electron repulsion integrals (ERI)

def coulomb_repulsion(za, la, ra, zb, lb, rb, zc, lc, rc, zd, ld, rd):
    return os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, la + lb + lc + ld)

def ERI(a, b, c, d):
    if d.contracted:
        return sum(cd * ERI(pd, c, a, b) for (cd, pd) in d)
    return a.norm * b.norm * c.norm * d.norm * \
        coulomb_repulsion(a.exponent, list(a.powers), a.origin,
                          b.exponent, list(b.powers), b.origin,
                          c.exponent, list(c.powers), c.origin,
                          d.exponent, list(d.powers), d.origin)

def makeERI(bfs):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs, nbfs, nbfs))
    for i, j, k, l in iterator4(nbfs):
        ints[i, j, k, l] \
            = ints[i, j, l, k] \
            = ints[j, i, k, l] \
            = ints[j, i, l, k] \
            = ints[k, l, i, j] \
            = ints[k, l, j, i] \
            = ints[l, k, i, j] \
            = ints[l, k, j, i] \
            = ERI(bfs[i], bfs[j], bfs[k], bfs[l])
    return ints

if args.ERI:
    print('making ERI...')
    ERI_pyints = makeERI(bfs)
    np.save('pyints.ERI.npy', ERI_pyints)

## two-electron spin-orbit integrals (J2) from King and Furlani's
## formulation.

def spin_orbit_2_KF(za, la, ra, zb, lb, rb, zc, lc, rc, zd, ld, rd, component):
    ai, aj = za, zb
    # ak, al = zc, zd
    xi, yi, zi = la[0], la[1], la[2]
    xj, yj, zj = lb[0], lb[1], lb[2]
    xk, yk, zk = lc[0], lc[1], lc[2]
    xl, yl, zl = ld[0], ld[1], ld[2]
    if component == 2:
        return (   xi*yi*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi-1, yi, zi, xj, yj-1, zj, xk, yk, zk, xl, yl, zl]) \
                -2*ai*yj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi+1, yi, zi, xj, yj-1, zj, xk, yk, zk, xl, yl, zl]) \
                -2*aj*xi*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi-1, yi, zi, xj, yj+1, zj, xk, yk, zk, xl, yl, zl]) \
                +4*ai*aj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi+1, yi, zi, xj, yj+1, zj, xk, yk, zk, xl, yl, zl]) \
                  -yi*xj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi-1, zi, xj-1, yj, zj, xk, yk, zk, xl, yl, zl]) \
                +2*ai*xj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi+1, zi, xj-1, yj, zj, xk, yk, zk, xl, yl, zl]) \
                +2*aj*yi*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi-1, zi, xj+1, yj, zj, xk, yk, zk, xl, yl, zl]) \
                -4*ai*aj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi+1, zi, xj+1, yj, zj, xk, yk, zk, xl, yl, zl]))
    if component == 0:
        return (   yi*zj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi-1, zi, xj, yj, zj-1, xk, yk, zk, xl, yl, zl]) \
                -2*aj*yi*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi-1, zi, xj, yj, zj+1, xk, yk, zk, xl, yl, zl]) \
                -2*ai*zj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi+1, zi, xj, yj, zj-1, xk, yk, zk, xl, yl, zl]) \
                +4*ai*aj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi+1, zi, xj, yj, zj+1, xk, yk, zk, xl, yl, zl]) \
                  -zi*yj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi, zi-1, xj, yj-1, zj, xk, yk, zk, xl, yl, zl]) \
                +2*aj*zi*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi, zi-1, xj, yj+1, zj, xk, yk, zk, xl, yl, zl]) \
                +2*ai*yj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi, zi+1, xj, yj-1, zj, xk, yk, zk, xl, yl, zl]) \
                -4*ai*aj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi, zi+1, xj, yj+1, zj, xk, yk, zk, xl, yl, zl]))
    if component == 1:
        return (   zi*xj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi, zi-1, xj-1, yj, zj, xk, yk, zk, xl, yl, zl]) \
                -2*aj*zi*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi, zi-1, xj+1, yj, zj, xk, yk, zk, xl, yl, zl]) \
                -2*ai*xj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi, zi+1, xj-1, yj, zj, xk, yk, zk, xl, yl, zl]) \
                +4*ai*aj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi, yi, zi+1, xj+1, yj, zj, xk, yk, zk, xl, yl, zl]) \
                  -xi*zj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi-1, yi, zi, xj, yj, zj-1, xk, yk, zk, xl, yl, zl]) \
                +2*aj*xi*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi-1, yi, zi, xj, yj, zj+1, xk, yk, zk, xl, yl, zl]) \
                +2*ai*zj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi+1, yi, zi, xj, yj, zj-1, xk, yk, zk, xl, yl, zl]) \
                -4*ai*aj*os.get_coulomb(za, zb, zc, zd, ra, rb, rc, rd, [xi+1, yi, zi, xj, yj, zj+1, xk, yk, zk, xl, yl, zl]))

def J2_KF(a, b, c, d, component):
    if a.contracted:
        return sum(ca * J2_KF(pa, b, c, d, component) for (ca, pa) in a)
    if b.contracted:
        return sum(cb * J2_KF(a, pb, c, d, component) for (cb, pb) in b)
    if c.contracted:
        return sum(cc * J2_KF(a, b, pc, d, component) for (cc, pc) in c)
    if d.contracted:
        return sum(cd * J2_KF(a, b, c, pd, component) for (cd, pd) in d)
    return a.norm * b.norm * c.norm * d.norm * spin_orbit_2_KF(a.exponent, list(a.powers), a.origin,
                                                               b.exponent, list(b.powers), b.origin,
                                                               c.exponent, list(c.powers), c.origin,
                                                               d.exponent, list(d.powers), d.origin,
                                                               component)

def makeJ2_KF(bfs):
    nbfs = len(bfs)
    ints_X = np.zeros(shape=(nbfs, nbfs, nbfs, nbfs))
    ints_Y = np.zeros(shape=(nbfs, nbfs, nbfs, nbfs))
    ints_Z = np.zeros(shape=(nbfs, nbfs, nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            for lm, c in enumerate(bfs):
                for sg, d in enumerate(bfs):
                    ints_X[mu, nu, lm, sg] = J2_KF(a, b, c, d, 0)
                    ints_Y[mu, nu, lm, sg] = J2_KF(a, b, c, d, 1)
                    ints_Z[mu, nu, lm, sg] = J2_KF(a, b, c, d, 2)
    return ints_X, ints_Y, ints_Z

if args.J2_KF:
    print('making J2_KF...')
    J2X_pyints, J2Y_pyints, J2Z_pyints = makeJ2_KF(bfs)
    np.save('pyints.J2X.npy', J2X_pyints)
    np.save('pyints.J2Y.npy', J2Y_pyints)
    np.save('pyints.J2Z.npy', J2Z_pyints)
