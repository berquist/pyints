"""test_moment.py: compare two methods for computing Cartesian moment integrals:
1. using the Obara-Saika recursion scheme (pyints)
2. using modified overlap integrals in the form of Pyquante
"""

from __future__ import print_function

import obarasaika.obara_saika as os
import math
fact = math.factorial

alpha1 = 0.1688554
lmn1 = [0, 0, 0]
A = [0., 0., 3.01392398]
alpha2 = 0.0480887
lmn2 = [0, 0, 1]
B = [0.0, 0.0, 0.0]
C = [0.0, 0.0, 0.0]
order = [0, 0, 1]

# Why the discrepancy?
# result_pyquante = 306.245173916
result_pyquante = 306.245173977


def binomial(a, b):
    return fact(a) / fact(b) / fact(a - b)

def overlap(alpha1, lmn1, A, alpha2, lmn2, B):
    za, zb, la, lb, ra, rb = alpha1, alpha2, lmn1, lmn2, A, B
    return os.get_overlap(za, zb, ra, rb, la + lb)

def cartesian_moment_pyints(alpha1, lmn1, A, alpha2, lmn2, B, C, order):
    za, zb, la, lb, ra, rb, rc = alpha1, alpha2, lmn1, lmn2, A, B, C
    return os.get_moment(za, zb, ra, rb, rc, la + lb, order)

def cartesian_moment_pyquante(alpha1, lmn1, A, alpha2, lmn2, B, C, order):
    kx, ky, kz = order
    total = 0
    for ix in range(kx+1):
        for iy in range(ky+1):
            for iz in range(kz+1):
                total += binomial(kx, ix) * binomial(ky, iy) * binomial(kz, iz) * (A[0]**(kx-ix)) * (A[1]**(ky-iy)) * (A[2]**(kz-iz)) * overlap(alpha1, [lmn1[0]+ix, lmn1[1]+iy, lmn1[2]+iz], A, alpha2, lmn2, B)
    return total

def test_moment(lmn1=[0, 0, 0], lmn2=[0, 0, 0], order=[0, 0, 0], thresh=1.0e-5):
    pyints = cartesian_moment_pyints(alpha1, lmn1, A, alpha2, lmn2, B, C, order)
    pyquante = cartesian_moment_pyquante(alpha1, lmn1, A, alpha2, lmn2, B, C, order)
    diff = abs(pyints - pyquante)
    print(lmn1, lmn2, order, diff)
    assert diff < thresh
    return


if __name__ == '__main__':

    import itertools as i

    for lmn1 in i.product(range(3), range(3), range(3)):
        for lmn2 in i.product(range(3), range(3), range(3)):
            for order in i.product(range(3), range(3), range(3)):
                test_moment(list(lmn1), list(lmn2), list(order))
