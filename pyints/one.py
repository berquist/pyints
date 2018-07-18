import numpy as np

import obarasaika.obara_saika as os

# from .numerical import linear_momentum


### overlap (S) integrals

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


### Fermi contact (FC) integrals

def fermi_contact(alpha1, lmn1, A, alpha2, lmn2, B, C):
    return os.get_fermi(alpha1, alpha2, A, B, lmn1 + lmn2, C)


def FC(a, b, C):
    if b.contracted:
        return sum(cb * FC(pb, a, C) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * FC(b, pa, C) for (ca, pa) in a)
    return a.norm * b.norm * fermi_contact(a.exponent, list(a.powers), a.origin,
                                           b.exponent, list(b.powers), b.origin,
                                           C)


def makeFC(bfs, origin):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = FC(a, b, origin)
    return ints


### kinetic energy (T) integrals

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


### nuclear attraction (V) integrals

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


### Cartesian moment (M) integrals

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


### electric field (E) integrals

def electric_field(alpha1, lmn1, A, alpha2, lmn2, B, C, order):
    # NYI
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


### electric field (EF) integrals from nuclear attraction integrals

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


### electric field gradient (EFG) integrals from electric field integrals

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


### angular momentum (L) integrals

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


### angular momentum (L) integrals from first moment integrals

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


### dipole velocity / linear momentum / nabla (N) integrals


def linear_momentum_from_S(alpha1, lmn1, A, alpha2, lmn2, B, component):
    xi, yi, zi, xj, yj, zj = lmn1[0], lmn1[1], lmn1[2], lmn2[0], lmn2[1], lmn2[2]
    if component == 0:
        return (-2.0 * alpha2 * os.get_overlap(alpha1, alpha2, A, B, [xi, yi, zi, xj+1, yj, zj])) \
            + (xj * os.get_overlap(alpha1, alpha2, A, B, [xi, yi, zi, xj-1, yj, zj]))
    if component == 1:
        return (-2.0 * alpha2 * os.get_overlap(alpha1, alpha2, A, B, [xi, yi, zi, xj, yj+1, zj])) \
            + (yj * os.get_overlap(alpha1, alpha2, A, B, [xi, yi, zi, xj, yj-1, zj]))
    if component == 2:
        return (-2.0 * alpha2 * os.get_overlap(alpha1, alpha2, A, B, [xi, yi, zi, xj, yj, zj+1])) \
            + (zj * os.get_overlap(alpha1, alpha2, A, B, [xi, yi, zi, xj, yj, zj-1]))


def N_from_S(a, b, component):
    if b.contracted:
        return sum(cb * N_from_S(pb, a, component) for (cb, pb) in b)
    elif a.contracted:
        return sum(ca * N_from_S(b, pa, component) for (ca, pa) in a)
    return a.norm * b.norm * linear_momentum_from_S(a.exponent, list(a.powers), a.origin,
                                                    b.exponent, list(b.powers), b.origin,
                                                    component)


def makeN_from_S(bfs, component):
    nbfs = len(bfs)
    ints = np.zeros(shape=(nbfs, nbfs))
    for mu, a in enumerate(bfs):
        for nu, b in enumerate(bfs):
            ints[mu, nu] = N_from_S(a, b, component)
    return ints


# def N_numerical(a, b, component):
#     if b.contracted:
#         return sum(cb * N_numerical(pb, a, component) for (cb, pb) in b)
#     elif a.contracted:
#         return sum(ca * N_numerical(b, pa, component) for (ca, pa) in a)
#     return a.norm * b.norm * linear_momentum(a.exponent, list(a.powers), a.origin,
#                                              b.exponent, list(b.powers), b.origin,
#                                              component)


# def makeN_numerical(bfs, component):
#     nbfs = len(bfs)
#     ints = np.zeros(shape=(nbfs, nbfs))
#     for mu, a in enumerate(bfs):
#         for nu, b in enumerate(bfs):
#             ints[mu, nu] = N_numerical(a, b, component)
#     return ints


### spin-orbit interaction (J) integrals

def spin_orbit(alpha1, lmn1, A, alpha2, lmn2, B, C):
    # NYI
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


### spin-orbit interaction (J) integrals from King and Furlani's
### formulation

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

