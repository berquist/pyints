from __future__ import print_function

import pyquante2

from pyints import getargs
from pyints.test import make_test_integrals_pyquante


with open('LiH.xyz') as molfile:
    mollines = molfile.readlines()[2:]

mol = pyquante2.geo.molecule.read_xyz_lines(mollines,
                                            units='Angstrom',
                                            charge=0,
                                            multiplicity=1,
                                            name='LiH')

del mollines

mol_basis = pyquante2.basisset(mol, 'STO-3G')

args = getargs()
make_test_integrals_pyquante(args, mol, mol_basis)
