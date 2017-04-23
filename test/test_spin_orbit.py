from __future__ import print_function

import itertools as i

from pyints.two import spin_orbit_2_KF


# def test_spin_orbit_2_KF():
#     za = 0.1688554
#     zb = 0.0480887
#     zc = zb
#     zd = zb

#     ra = [0., 0., 3.01392398]
#     rb = [0.0, 0.0, 1.0]
#     rc = [0.0, 0.0, 2.0]
#     rd = [0.0, 0.0, 4.0]

#     la = [2, 0, 0]
#     lb = [2, 0, 0]
#     lc = [2, 0, 0]
#     ld = [1, 1, 0]

#     component = 2

#     for ta, tb, tc, td in set(i.permutations((tuple(la), tuple(lb), tuple(lc), tuple(ld)), 4)):
#         print(ta, tb, tc, td,
#               spin_orbit_2_KF(za, zb, zc, zd, ra, rb, rc, rd, list(ta), list(tb), list(tc), list(td), component))

if __name__ == '__main__':
    pass
