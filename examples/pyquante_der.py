from __future__ import print_function

import os.path

from math import sqrt

import numpy as np

from PyQuante.Molecule import Molecule
from PyQuante.Ints import getbasis
from PyQuante.PGBF import PGBF


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
                v.normalize()

                terma = sqrt(alpha*(2.0*l+1.0))*coefs*v.overlap(upbf)

                if l > 0:
                    v.reset_powers(l-1, m, n)
                    v.normalize()
                    termb = -2*l*sqrt(alpha/(2.0*l-1.0))*coefs*v.overlap(upbf)
                else:
                    termb = 0.0

                dSij_dXa += terma + termb

                # y component
                v.reset_powers(l, m+1, n)
                v.normalize()
                terma = sqrt(alpha*(2.0*m+1.0))*coefs*v.overlap(upbf)

                if m > 0:
                    v.reset_powers(l, m-1, n)
                    v.normalize()
                    termb = -2*m*sqrt(alpha/(2.0*m-1.0))*coefs*v.overlap(upbf)
                else:
                    termb = 0.0

                dSij_dYa += terma + termb

                # z component
                v.reset_powers(l, m, n+1)
                v.normalize()
                terma = sqrt(alpha*(2.0*n+1.0))*coefs*v.overlap(upbf)

                if n > 0:
                    v.reset_powers(l, m, n-1)
                    v.normalize()
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
                v.normalize()

                terma = sqrt(alpha*(2.0*l+1.0))*coefs*v.overlap(upbf)

                if l > 0:
                    v.reset_powers(l-1, m, n)
                    v.normalize()
                    termb = -2*l*sqrt(alpha/(2.0*l-1.0))*coefs*v.overlap(upbf)
                else:
                    termb = 0.0

                dSij_dXa += terma + termb

                # y component
                v.reset_powers(l, m+1, n)
                v.normalize()
                terma = sqrt(alpha*(2.0*m+1.0))*coefs*v.overlap(upbf)

                if m > 0:
                    v.reset_powers(l, m-1, n)
                    v.normalize()
                    termb = -2*m*sqrt(alpha/(2.0*m-1.0))*coefs*v.overlap(upbf)
                else:
                    termb = 0.0

                dSij_dYa += terma + termb

                # z component
                v.reset_powers(l, m, n+1)
                v.normalize()
                terma = sqrt(alpha*(2.0*n+1.0))*coefs*v.overlap(upbf)

                if n > 0:
                    v.reset_powers(l, m, n-1)
                    v.normalize()
                    termb = -2*n*sqrt(alpha/(2.0*n-1.0))*coefs*v.overlap(upbf)
                else:
                    termb = 0.0

                dSij_dZa += terma + termb

    return dSij_dXa, dSij_dYa, dSij_dZa


def der_overlap_matrix(a, bset):
    """
    Evaluates the derivative of the overlap integrals
    with respect to atomic coordinate
    """
    # initialize dS/dR matrices
    nbf = len(bset)
    dS_dXa = np.zeros((nbf, nbf), 'd')
    dS_dYa = np.zeros((nbf, nbf), 'd')
    dS_dZa = np.zeros((nbf, nbf), 'd')

    for i in range(nbf):
        for j in range(nbf):
            dS_dXa[i][j], dS_dYa[i][j], dS_dZa[i][j] = der_overlap_element(
                a, bset[i], bset[j])

    # if a==1: print "dS_dXa"; pad_out(dS_dXa); print "dS_dYa"; pad_out(dS_dYa); print "dS_dZa"; pad_out(dS_dZa);
    return dS_dXa, dS_dYa, dS_dZa


def main():
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dirname, 'LiH.xyz')

    mol = Molecule.from_file(filename, format='xyz')
    bfs = getbasis(mol.atoms, 'sto-3g')

    print(der_overlap_matrix(0, bfs))

    return


if __name__ == '__main__':
    main()
