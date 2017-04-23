from __future__ import print_function

from itertools import combinations_with_replacement

import numpy as np


def make_test_integrals_pyints(args, mol, mol_basis):
    """A driver for making integrals for test comparisons using pyints.
    """

    bfs = mol_basis.bfs

    # One-electron integrals

    if args.S:
        from pyints.one import makeS
        print('making S...')
        S_pyints = makeS(bfs)
        np.savetxt('pyints.S.txt', S_pyints)

    if args.FC:
        from pyints.one import makeFC
        print('making FC...')
        for iat, at in enumerate(mol, 1):
            FC_pyints = makeFC(bfs, at.r)
            np.savetxt('pyints.FC{}.txt'.format(iat), FC_pyints)

    if args.T:
        from pyints.one import makeT
        print('making T...')
        T_pyints = makeT(bfs)
        np.savetxt('pyints.T.txt', T_pyints)

    if args.V:
        from pyints.one import makeV
        print('making V...')
        V_pyints = makeV(mol, bfs)
        np.savetxt('pyints.V.txt', V_pyints)

    if args.M:
        from pyints.one import makeM
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

    component_map = {
        'X': 0,
        'Y': 1,
        'Z': 2,
    }

    if args.E:
        from pyints.one import makeE
        print('making E...')

    if args.EF_from_V:
        from pyints.one import makeEF
        print('making EF from V...')
        counter = 1
        for at in mol:
            for component in range(3):
                EF_pyints = makeEF(bfs, at.r, component)
                np.savetxt('pyints.EF{}.txt'.format(counter), EF_pyints)
                counter += 1

    if args.EFG_from_EF_from_V:
        from pyints.one import makeEFG
        print('making EFG from EF from V...')
        for iat, at in enumerate(mol, 1):
            for (c1, c2) in combinations_with_replacement(('X', 'Y', 'Z'), 2):
                EFG_pyints = makeEFG(bfs, at.r, component_map[c1], component_map[c2])
                np.savetxt('pyints.EFG{}{}{}.txt'.format(iat, c1, c2), EFG_pyints)

    if args.L:
        from pyints.one import makeL
        print('making L...')
        LX_pyints = makeL(bfs, [0.0, 0.0, 0.0], 0)
        LY_pyints = makeL(bfs, [0.0, 0.0, 0.0], 1)
        LZ_pyints = makeL(bfs, [0.0, 0.0, 0.0], 2)
        np.savetxt('pyints.LX.txt', LX_pyints)
        np.savetxt('pyints.LY.txt', LY_pyints)
        np.savetxt('pyints.LZ.txt', LZ_pyints)

    if args.L_from_M:
        from pyints.one import makeL_from_M
        print('making L_from_M...')
        LX_M_pyints = makeL_from_M(bfs, [0.0, 0.0, 0.0], 0)
        LY_M_pyints = makeL_from_M(bfs, [0.0, 0.0, 0.0], 1)
        LZ_M_pyints = makeL_from_M(bfs, [0.0, 0.0, 0.0], 2)
        np.savetxt('pyints.LX_M.txt', LX_M_pyints)
        np.savetxt('pyints.LY_M.txt', LY_M_pyints)
        np.savetxt('pyints.LZ_M.txt', LZ_M_pyints)

    if args.J:
        from pyints.one import makeJ
        print('making J...')

    if args.J_KF:
        from pyints.one import makeJ_KF
        print('making J_KF...')
        J1X_pyints, J1Y_pyints, J1Z_pyints = makeJ_KF(mol, bfs)
        np.savetxt('pyints.J1X.txt', J1X_pyints)
        np.savetxt('pyints.J1Y.txt', J1Y_pyints)
        np.savetxt('pyints.J1Z.txt', J1Z_pyints)

    # Two-electron integrals

    if args.ERI:
        from pyints.two import makeERI
        print('making ERI...')
        ERI_pyints = makeERI(bfs)
        np.savez_compressed('pyints.ERI.npz', ERI_pyints)

    if args.J2_KF:
        from pyints.two import makeJ2_KF
        print('making J2_KF...')
        J2X_pyints, J2Y_pyints, J2Z_pyints = makeJ2_KF(bfs)
        np.savez_compressed('pyints.J2X.npz', J2X_pyints)
        np.savez_compressed('pyints.J2Y.npz', J2Y_pyints)
        np.savez_compressed('pyints.J2Z.npz', J2Z_pyints)
