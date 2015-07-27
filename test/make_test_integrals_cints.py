from __future__ import print_function

import numpy as np


def make_test_integrals_cints(args, mol):
    """A driver for making integrals for test comparisons using cints.
    """

    # One-electron integrals

    if args.S:
        from pyints.cints import makeS
        print('making S...')
        S_cints = makeS(mol)
        np.savetxt('cints.S.txt', S_cints)

    if args.T:
        from pyints.cints import makeT
        print('making T...')
        T_cints = makeT(mol)
        np.savetxt('cints.T.txt', T_cints)

    if args.V:
        from pyints.cints import makeV
        print('making V...')
        V_cints = makeV(mol)
        np.savetxt('cints.V.txt', V_cints)
