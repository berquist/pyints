from __future__ import print_function
from __future__ import division

# pylint: disable=invalid-name

import numpy as np

from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as spline


prefac = 1.0 / np.sqrt(np.pi)


def pgto(r, za, la, ra):
    """A primitive Gaussian-type orbital."""
    return np.power(r - ra, la) * np.exp(-za * np.power(r - ra, 2))


def overlap1d(r, za, la, ra, zb, lb, rb, n=0, origin=0.0, london=False):
    """The product of two pGTOs."""
    pgto1 = pgto(r, za, la, ra)
    pgto2 = pgto(r, zb, lb, rb)
    if london:
        return pgto1 * pgto2 * np.power(r - rb, n)
    else:
        return pgto1 * pgto2 * np.power(r - origin, n)


def coulomb1d(r, t, za, la, ra, zb, lb, rb, rc, n, origin):
    pgto12 = overlap1d(r, za, la, ra, zb, lb, rb, n=0, origin=0.0, london=False)
    pgtoc = pgto(r, np.power(t, 2), 0, rc)
    return pgto12 * pgtoc * np.power(r - origin, n)


def kinetic1d(r, za, la, ra, zb, lb, rb, n, origin):
    pgto1 = pgto(r, za, la, ra)
    m = np.arange(r - 0.1, r + 0.1, 0.00075)
    pgto2 = pgto(m, zb, lb, rb)
    pgto2p = spline(m, pgto2).__call__(r, 2)
    return pgto1 * pgto2p * np.power(r - origin, n)


def del1d(r, za, la, ra, zb, lb, rb):
    pgto1 = pgto(r, za, la, ra)
    m = np.arange(r - 0.1, r + 0.1, 0.00075)
    pgto2 = pgto(m, zb, lb, rb)
    pgto2p = spline(m, pgto2).__call__(r, 1)
    return pgto1 * pgto2p


def V(t, za, la, ra, zb, lb, rb, rc, n, origin):
    # Integrate along the 3 Cartesian coordinates.
    res1 = quad(coulomb1d, -np.inf, np.inf, args=(t, za, la[0], ra[0], zb, lb[0], rb[0], rc[0], n[0], origin[0]))
    res2 = quad(coulomb1d, -np.inf, np.inf, args=(t, za, la[1], ra[1], zb, lb[1], rb[1], rc[1], n[1], origin[1]))
    res3 = quad(coulomb1d, -np.inf, np.inf, args=(t, za, la[2], ra[2], zb, lb[2], rb[2], rc[2], n[2], origin[2]))
    # print(res1)
    # print(res2)
    # print(res3)
    return prefac * res1[0] * res2[0] * res3[0]


def overlap(za, la, ra, zb, lb, rb, n, origin):
    sx = quad(overlap1d, -np.inf, np.inf, args=(za, la[0], ra[0], zb, lb[0], rb[0], n[0], origin[0]))
    sy = quad(overlap1d, -np.inf, np.inf, args=(za, la[1], ra[1], zb, lb[1], rb[1], n[1], origin[1]))
    sz = quad(overlap1d, -np.inf, np.inf, args=(za, la[2], ra[2], zb, lb[2], rb[2], n[2], origin[2]))
    return sx[0] * sy[0] * sz[0]


# def overlap_d


def fermi_contact(za, la, ra, zb, lb, rb, rc):
    if np.equal(ra, rc).all() or np.equal(rb, rc).all():
        fcx = quad(overlap1d, -np.inf, np.inf, args=(za, la[0], ra[0], zb, lb[0], rb[0]))
        fcy = quad(overlap1d, -np.inf, np.inf, args=(za, la[1], ra[1], zb, lb[1], rb[1]))
        fcz = quad(overlap1d, -np.inf, np.inf, args=(za, la[2], ra[2], zb, lb[2], rb[2]))
        return fcx[0] * fcy[0] * fcz[0]
    else:
        return 0
# def fermi_contact(za, la, ra, zb, lb, rb, rc):
#     # https://github.com/cp2k/cp2k/blob/master/cp2k/src/aobasis/ai_fermi_contact.F
#     # Two key differences:
#     # 1. we are only working with a single primitive
#     # 2. (4pi/3) is factored out
#     # lax, lay, laz = la
#     # lbx, lby, lbz = lb
#     # las = np.sum(la)
#     # lbs = np.sum(lb)
#     # lamin, lamax = la.min(), la.max()
#     # lbmin, lbmax = lb.min(), lb.max()
#     lamin, lamax = min(la), max(la)
#     lbmin, lbmax = min(lb), max(lb)
#     rac = ra - rc
#     rbc = rb - rc
#     dac2 = np.dot(rac, rac)
#     dbc2 = np.dot(rbc, rbc)
#     f0 = np.exp(-za*dac2-zb*dbc2)
#     # DO lb = lb_min, lb_max
#     #    DO bx = 0, lb
#     #       fbx = 1.0_dp
#     #       IF (bx .GT. 0) fbx = (rbc(1))**bx
#     #       DO by = 0, lb-bx
#     #          bz = lb-bx-by
#     #          cob = coset(bx, by, bz)
#     #          mb = nb+cob
#     #          fby = 1.0_dp
#     #          IF (by .GT. 0) fby = (rbc(2))**by
#     #          fbz = 1.0_dp
#     #          IF (bz .GT. 0) fbz = (rbc(3))**bz
#     #          DO la = la_min, la_max
#     #             DO ax = 0, la
#     #                fax = fbx
#     #                IF (ax .GT. 0) fax = fbx*(rac(1))**ax
#     #                DO ay = 0, la-ax
#     #                   az = la-ax-ay
#     #                   coa = coset(ax, ay, az)
#     #                   ma = na+coa
#     #                   fay = fby
#     #                   IF (ay .GT. 0) fay = fby*(rac(2))**ay
#     #                   faz = fbz
#     #                   IF (az .GT. 0) faz = fbz*(rac(3))**az
#     #                   fcab(ma, mb) = f0*fax*fay*faz
#     #                ENDDO
#     #             ENDDO
#     #          ENDDO      !la
#     #       ENDDO
#     #    ENDDO
#     # ENDDO               !lb
#     val = 0.0
#     for ilb in range(lbmin, lbmax+1):
#         for bx in range(0, ilb+1):
#             fbx = 1.0
#             if bx > 0:
#                 fbx = rbc[0] ** bx
#             for by in range(0, ilb - bx+1):
#                 bz = ilb - bx - by
#                 fby = 1.0
#                 if by > 0:
#                     fby = rbc[1] ** by
#                 fbz = 1.0
#                 if bz > 0:
#                     fbz = rbc[2] ** bz
#                 for ila in range(lamin, lamax+1):
#                     for ax in range(0, ila+1):
#                         fax = fbx
#                         if ax > 0:
#                             fax = fbx * rac[0] ** ax
#                         for ay in range(0, ila - ax+1):
#                             az = ila - ax - ay
#                             fay = fby
#                             if ay > 0:
#                                 fay = fby * rac[1] ** ay
#                             faz = fbz
#                             if ax > 0:
#                                 faz = fbz * rac[2] ** az
#                             # print('fax, fay, faz', fax, fay, faz)
#                             val += fax * fay * faz
#     print(f0, val, f0 * val)
#     return f0 * val


def kinetic_energy(za, la, ra, zb, lb, rb, n, origin):
    sx = quad(overlap1d, -np.inf, np.inf, args=(za, la[0], ra[0], zb, lb[0], rb[0], n[0], origin[0]))[0]
    sy = quad(overlap1d, -np.inf, np.inf, args=(za, la[1], ra[1], zb, lb[1], rb[1], n[1], origin[1]))[0]
    sz = quad(overlap1d, -np.inf, np.inf, args=(za, la[2], ra[2], zb, lb[2], rb[2], n[2], origin[2]))[0]
    tx = quad(kinetic1d, -np.inf, np.inf, args=(za, la[0], ra[0], zb, lb[0], rb[0], n[0], origin[0]))[0]
    ty = quad(kinetic1d, -np.inf, np.inf, args=(za, la[1], ra[1], zb, lb[1], rb[1], n[1], origin[1]))[0]
    tz = quad(kinetic1d, -np.inf, np.inf, args=(za, la[2], ra[2], zb, lb[2], rb[2], n[2], origin[2]))[0]
    return -0.5 * (tx*sy*sz + ty*sz*sx + tz*sx*sy)


def nuclear_attraction(za, la, ra, zb, lb, rb, rc, n, origin):
    t = np.inf
    res = quad(V, -t, t, args=(za, la, ra, zb, lb, rb, rc, n, origin))
    # print(res)
    return res[0]


def linear_momentum(za, la, ra, zb, lb, rb, direction='x'):
    # n is zero, origin is zero too?
    if direction in ('x', 0):
        sx = quad(overlap1d, -np.inf, np.inf, args=(za, la[0], ra[0], zb, lb[0], rb[0], 0, 0.0))[0]
        dx = quad(del1d, -np.inf, np.inf, args=(za, la[0], ra[0], zb, lb[0], rb[0]))[0]
        return sx*dx
    elif direction in ('y', 1):
        sy = quad(overlap1d, -np.inf, np.inf, args=(za, la[1], ra[1], zb, lb[1], rb[1], 0, 0.0))[0]
        dy = quad(del1d, -np.inf, np.inf, args=(za, la[1], ra[1], zb, lb[1], rb[1]))[0]
        return sy*dy
    elif direction in ('z', 2):
        sz = quad(overlap1d, -np.inf, np.inf, args=(za, la[2], ra[2], zb, lb[2], rb[2], 0, 0.0))[0]
        dz = quad(del1d, -np.inf, np.inf, args=(za, la[2], ra[2], zb, lb[2], rb[2]))[0]
        return sz*dz
    # return sx*dx, sy*dy, sz*dz
    # return dx, dy, dz


def angular_momentum(za, la, ra, zb, lb, rb, direction='x', london=False):
    sx = quad(overlap1d, -np.inf, np.inf, args=(za, la[0], ra[0], zb, lb[0], rb[0], 0, 0.0))[0]
    sy = quad(overlap1d, -np.inf, np.inf, args=(za, la[1], ra[1], zb, lb[1], rb[1], 0, 0.0))[0]
    sz = quad(overlap1d, -np.inf, np.inf, args=(za, la[2], ra[2], zb, lb[2], rb[2], 0, 0.0))[0]
    rx = quad(overlap1d, -np.inf, np.inf, args=(za, la[0], ra[0], zb, lb[0], rb[0], 1, 0.0, london))[0]
    ry = quad(overlap1d, -np.inf, np.inf, args=(za, la[1], ra[1], zb, lb[1], rb[1], 1, 0.0, london))[0]
    rz = quad(overlap1d, -np.inf, np.inf, args=(za, la[2], ra[2], zb, lb[2], rb[2], 1, 0.0, london))[0]
    dx = quad(del1d, -np.inf, np.inf, args=(za, la[0], ra[0], zb, lb[0], rb[0]))[0]
    dy = quad(del1d, -np.inf, np.inf, args=(za, la[1], ra[1], zb, lb[1], rb[1]))[0]
    dz = quad(del1d, -np.inf, np.inf, args=(za, la[2], ra[2], zb, lb[2], rb[2]))[0]
    if direction.lower() == 'x':
        return -sx*(ry*dz - rz*dy)
    elif direction.lower() == 'y':
        return -sy*(-rx*dz + rz*dx)
    elif direction.lower() == 'z':
        return -sz*(rx*dy - ry*dx)


def test_overlap():
    za = 0.05
    zb = 0.012
    la = (0, 2, 1)
    lb = (3, 1, 0)
    ra = (0.2, 0.0, 1.0)
    rb = (10.0, -0.4, 0.0)
    n = (1, 0, 1)
    origin = np.asarray([-110.0, 0.0, 0.0])

    thresh = 1.0e-15
    ref = -113234318.25168519

    res = overlap(za, la, ra, zb, lb, rb, n, origin)

    assert abs(ref - res) < thresh

    return


# def test_overlap_der1():

#     za = 0.05
#     zb = 0.012
#     la = (0, 2, 1)
#     lb = (3, 1, 0)
#     ra = (0.2, 0.0, 1.0)
#     rb = (10.0, -0.4, 0.0)
#     n = (1, 0, 1)
#     origin = np.asarray([0.0, 0.0, 0.0])

#     thresh = 1.0e-15

#     return


def test_kinetic_energy():

    za = 0.05
    zb = 0.012
    la = (0, 2, 1)
    lb = (3, 1, 0)
    ra = (0.2, 0.0, 1.0)
    rb = (10.0, -0.4, 0.0)
    n = (1, 0, 1)
    origin = np.asarray([-110.0, 0.0, 0.0])

    # analytic, numerical
    # -2613140.44431
    # -2613140.44838

    thresh = 1.0e-15
    ref = -2613140.448376812

    # IntegrationWarning: The occurrence of roundoff error is
    # detected, which prevents the requested tolerance from being
    # achieved.  The error may be underestimated.
    res = kinetic_energy(za, la, ra, zb, lb, rb, n, origin)

    assert abs(ref - res) < thresh

    return


def test_nuclear_attraction():
    # a = 0.1
    # la = (2, 0, 1)
    # A = np.asarray([0.2, 0.0, 11.0])

    # b = 0.002
    # lb = (0, 3, 2)
    # B = np.asarray([0.1, 0.8, 10.4])

    # C = np.asarray([0.15, 0.5, 10.00])
    # origin = np.array([-2.0, 5.0, 10.0])

    # n = (1, 3, 2)

    # print(nuclear_attraction(a, la, A, b, lb, B, C, n, origin))

    za = 1.8
    zb = 2.0
    la = (0, 0, 0)
    lb = (0, 0, 0)
    ra = [0.0, 0.0, 0.0]
    rb = [0.5, 0.8, -0.2]
    rc = [0.5, 0.8, 0.2]
    origin = (0.0, 0.0, 0.0)
    n = (0, 0, 0)

    res = nuclear_attraction(za, la, ra, zb, lb, rb, rc, n, origin)

    return


def test_linear_momentum():
    za = 0.05
    zb = 0.02
    la = (0, 2, 1)
    lb = (3, 1, 0)
    ra = np.asarray([0.2, 0.0, 18.0])
    rb = np.asarray([0.0, -0.4, 17.0])
    rc = np.asarray([0.0, 0.0, 0.0])

    res = linear_momentum(za, la, ra, zb, lb, rb)
    res_x, res_y, res_z = res

    print(res_x, res_y, res_z)

    return


def test_angular_momentum():
    za = 0.05
    zb = 0.02
    la = (0, 2, 1)
    lb = (3, 1, 0)
    ra = np.asarray([0.2, 0.0, 18.0])
    rb = np.asarray([0.0, -0.4, 17.0])
    rc = np.asarray([0.0, 0.0, 0.0])

    thresh = 1.0e-15
    ref_x = 40334.27925199575
    ref_y = -9086.560257754349
    ref_z = -184059.34720275138

    res_x = angular_momentum(za, la, ra, zb, lb, rb, direction='x', london=False)
    res_y = angular_momentum(za, la, ra, zb, lb, rb, direction='y', london=False)
    res_z = angular_momentum(za, la, ra, zb, lb, rb, direction='z', london=False)

    assert abs(ref_x - res_x) < thresh
    assert abs(ref_y - res_y) < thresh
    assert abs(ref_z - res_z) < thresh

    return

if __name__ == '__main__':
    # test_overlap()
    # test_kinetic_energy()
    # test_nuclear_attraction()
    # test_linear_momentum()
    # test_angular_momentum()
    test_overlap_der1()
