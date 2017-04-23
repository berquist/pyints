from __future__ import division
from __future__ import print_function

import pyquante2

from pyints.one import makeL
from pyints.one import makeL_from_M


def print_L(bfs, origin):

    print("origin: {}".format(origin))

    LX_pyints = makeL_from_M(bfs, origin, 0)
    LY_pyints = makeL_from_M(bfs, origin, 1)
    LZ_pyints = makeL_from_M(bfs, origin, 2)

    print(LX_pyints)
    print(LY_pyints)
    print(LZ_pyints)

    return


def print_L_example():
    with open('LiH.xyz') as molfile:
        mollines = molfile.readlines()[2:]

    mol = pyquante2.geo.molecule.read_xyz_lines(mollines,
                                                units='Angstrom',
                                                charge=0,
                                                multiplicity=1,
                                                name='LiH')

    del mollines

    mol_basis = pyquante2.basisset(mol, 'STO-3G')
    bfs = mol_basis.bfs

    print(mol)
    print(bfs)

    print_L(bfs, [0.0, 0.0, 0.0])
    print_L(bfs, [0.0, 0.0, 0.937852])
    print_L(bfs, [100.0, 200.0, 300.0])

    return


if __name__ == '__main__':
    pass
