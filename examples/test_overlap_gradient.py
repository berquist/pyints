from __future__ import print_function

from math import sqrt

from obarasaika.obara_saika import get_overlap, get_bi_center
from PyQuante.PGBF import PGBF
from PyQuante.CGBF import CGBF

# from PyQuante.AnalyticDerivatives import der_overlap_element


def der_overlap_element(a, bfi, bfj):
    """
    finds the derivative of the overlap integral with respect to the 
    atomic coordinate of atom "a".  Note there are four possible cases
    for evaluating this integral:
     1. Neither of the basis functions depend on the position of atom a
        ie. they are centered on atoms other than atom a
     2 and 3. One of the basis functions depends on the position of atom a
        so we need to evaluate the derivative of a Gaussian with the 
        recursion (right word?) relation derived on page 442 of Szabo.
     4. Both of the basis functions are centered on atom a, which through the
        recursion relation for the derivative of a Gaussian basis function will
        require the evaluation of 4 overlap integrals...

    this function will return a 3 element list with the derivatives of the overlap
    integrals with respect to the atomic coordinates Xa,Ya,Za.
    """
    dSij_dXa, dSij_dYa, dSij_dZa = 0.0, 0.0, 0.0

    # we use atom ids on the CGBFs to evaluate which of the 4 above case we have
    if bfi.atid == a:  # bfi is centered on atom a
        for upbf in bfj.prims:
            for vpbf in bfi.prims:
                alpha = vpbf.exp
                l, m, n = vpbf.powers
                origin = vpbf.origin
                coefs = upbf.coef*vpbf.coef

                # x component
                v = PGBF(alpha, origin, (l+1, m, n))

                terma = sqrt(alpha*(2.0*l+1.0))*coefs*v.overlap(upbf)

                if l > 0:
                    v = PGBF(alpha, origin, (l-1, m, n))
                    termb = -2*l*sqrt(alpha/(2.0*l-1.0))*coefs*v.overlap(upbf)
                else:
                    termb = 0.0

                dSij_dXa += terma + termb

                # y component
                v = PGBF(alpha, origin, (l, m+1, n))
                terma = sqrt(alpha*(2.0*m+1.0))*coefs*v.overlap(upbf)

                if m > 0:
                    v = PGBF(alpha, origin, (l, m-1, n))
                    termb = -2*m*sqrt(alpha/(2.0*m-1.0))*coefs*v.overlap(upbf)
                else:
                    termb = 0.0

                dSij_dYa += terma + termb

                # z component
                v = PGBF(alpha, origin, (l, m, n+1))
                terma = sqrt(alpha*(2.0*n+1.0))*coefs*v.overlap(upbf)

                if n > 0:
                    v = PGBF(alpha, origin, (l, m, n-1))
                    termb = -2*n*sqrt(alpha/(2.0*n-1.0))*coefs*v.overlap(upbf)
                else:
                    termb = 0.0

                dSij_dZa += terma + termb

    # bfj is centered on atom a
    if bfj.atid == a:
        for upbf in bfi.prims:
            for vpbf in bfj.prims:
                alpha = vpbf.exp
                l, m, n = vpbf.powers
                origin = vpbf.origin
                coefs = upbf.coef*vpbf.coef

                # x component
                v = PGBF(alpha, origin, (l+1, m, n))

                terma = sqrt(alpha*(2.0*l+1.0))*coefs*v.overlap(upbf)

                if l > 0:
                    v = PGBF(alpha, origin, (l-1, m, n))
                    termb = -2*l*sqrt(alpha/(2.0*l-1.0))*coefs*v.overlap(upbf)
                else:
                    termb = 0.0

                dSij_dXa += terma + termb

                # y component
                v = PGBF(alpha, origin, (l, m+1, n))
                terma = sqrt(alpha*(2.0*m+1.0))*coefs*v.overlap(upbf)

                if m > 0:
                    v = PGBF(alpha, origin, (l, m-1, n))
                    termb = -2*m*sqrt(alpha/(2.0*m-1.0))*coefs*v.overlap(upbf)
                else:
                    termb = 0.0

                dSij_dYa += terma + termb

                # z component
                v = PGBF(alpha, origin, (l, m, n+1))
                terma = sqrt(alpha*(2.0*n+1.0))*coefs*v.overlap(upbf)

                if n > 0:
                    v = PGBF(alpha, origin, (l, m, n-1))
                    termb = -2*n*sqrt(alpha/(2.0*n-1.0))*coefs*v.overlap(upbf)
                else:
                    termb = 0.0

                dSij_dZa += terma + termb

    return dSij_dXa, dSij_dYa, dSij_dZa


if __name__ == '__main__':

    za = 0.01
    zb = 0.02
    ra = [0.0, 0.0, 0.0]
    rb = [0.05, 0.10, 0.15]

    thresh = 1.0e-16

    la = [0, 0, 0]
    lb = [0, 0, 0]
    integral_os = get_overlap(za, zb, ra, rb, la + lb)
    print(integral_os)

    # take the gradient in the X direction
    rp = get_bi_center(za, zb, ra, rb)
    ta = 2 * za * (rp[0] - ra[0])
    tb = 2 * zb * (rp[0] - rb[0])
    print(-(ta + tb) * integral_os)

    # ga = PGBF(za, tuple(ra), tuple(la))
    # gb = PGBF(zb, tuple(rb), tuple(lb))
    # print(ga)
    # print(gb)
    # # PyQuante includes the normalization prefactor
    # integral_pq = ga.overlap(gb) / (ga.norm * gb.norm)

    ga = CGBF(origin=tuple(ra), powers=tuple(la), atid=0)
    gb = CGBF(origin=tuple(rb), powers=tuple(lb), atid=0)
    ga.add_primitive(exponent=za, coefficient=1.0)
    gb.add_primitive(exponent=zb, coefficient=1.0)
    # print(ga)
    # print(gb)

    # PyQuante includes the normalization prefactor
    integral_pq = ga.overlap(gb) / (ga.pnorms[0] * gb.pnorms[0])
    print(integral_pq)
    print(der_overlap_element(0, ga, gb))
    # print(ga.doverlap(gb, 0))
    # print(ga.doverlap_num(gb, 0))

    # from mmd.integrals
