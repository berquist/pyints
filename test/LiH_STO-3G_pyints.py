from __future__ import print_function

import os.path

import pyquante2

from pyints.args import getargs
from pyints.make_test_integrals_pyints import make_test_integrals_pyints


def main():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirname, 'LiH.xyz')) as molfile:
        mollines = molfile.readlines()[2:]

    mol = pyquante2.geo.molecule.read_xyz_lines(mollines,
                                                units='Angstrom',
                                                charge=0,
                                                multiplicity=1,
                                                name='LiH')

    del mollines

    mol_basis = pyquante2.basisset(mol, 'STO-3G'.lower())

    args = getargs()
    make_test_integrals_pyints(args, mol, mol_basis)

    return


if __name__ == '__main__':
    main()
