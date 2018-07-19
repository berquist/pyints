import numpy as np

import obarasaika.obara_saika as os

from ...utils import iterator4


### two-electron repulsion integrals (ERI)

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


### two-electron spin-orbit integrals (J2) from King and Furlani's
### formulation.

def spin_orbit_2_KF(za, zb, zc, zd, ra, rb, rc, rd, la, lb, lc, ld, component):
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
    return a.norm * b.norm * c.norm * d.norm * \
        spin_orbit_2_KF(a.exponent, b.exponent, c.exponent, d.exponent,
                        a.origin, b.origin, c.origin, d.origin,
                        list(a.powers), list(b.powers), list(c.powers), list(d.powers),
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
