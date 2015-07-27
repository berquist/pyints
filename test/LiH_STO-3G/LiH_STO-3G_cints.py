from __future__ import print_function

from pyscf.gto import Mole

from pyints import getargs
from pyints.test import make_test_integrals_cints


with open('LiH.xyz') as molfile:
    mollines = molfile.readlines()[2:]

mol = Mole(atom=mollines, basis='STO-3G', charge=0)
mol.build()

args = getargs()
make_test_integrals_cints(args, mol)
