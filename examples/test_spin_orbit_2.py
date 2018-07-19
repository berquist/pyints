from __future__ import print_function

from pyints.interfaces.obarasaika.one import spin_orbit_KF
from pyints.interfaces.obarasaika.two import spin_orbit_2_KF

from collections import namedtuple

import itertools as i


def _disabled():
    bf = namedtuple('bf', ['z', 'r', 'l'])

    bf_a = bf(z=0.1688554, r=[0., 0., 3.01392398], l=[2, 0, 0])
    bf_b = bf(z=0.0480887, r=[0.0, 0.0, 1.0], l=[2, 0, 0])
    bf_c = bf(z=bf_b.z, r=[0.0, 0.0, 2.0], l=[2, 0, 0])
    bf_d = bf(z=bf_b.z, r=[0.0, 0.0, 4.0], l=[1, 1, 0])

    component = 2

    # for mu, nu, lm, sg in i.permu
    print(spin_orbit_KF(bf_a.z, bf_a.l, bf_a.r, bf_b.z,
                        bf_b.l, bf_b.r, [0.0, 0.0, 0.0], component))
    print(spin_orbit_KF(bf_b.z, bf_b.l, bf_b.r, bf_a.z,
                        bf_a.l, bf_a.r, [0.0, 0.0, 0.0], component))

    return


if __name__ == '__main__':
    pass
