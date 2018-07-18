from __future__ import print_function

import numpy as np

import PyQuante
import pyquante2


def pyquante_2_to_1_molecule(old_molecule):

    from PyQuante.Atom import Atom
    from PyQuante import Molecule

    # new_atoms = []
    # for old_atom in old_molecule.atoms:
    #     new_atom = Atom(atno=old_atom.Z,
    #                     x=old_atom.r[0],
    #                     y=old_atom.r[1],
    #                     z=old_atom.r[2])
    #     new_atoms.append(new_atom)

    # new_molecule = Molecule(name=old_molecule.name,
    #                         atomlist=new_atoms,
    #                         units=old_molecule.units,
    #                         charge=old_molecule.charge,
    #                         multiplicity=old_molecule.multiplicity)

    new_molecule = Molecule(name=old_molecule.name,
                            units=old_molecule.units,
                            charge=old_molecule.charge,
                            multiplicity=old_molecule.multiplicity)

    for old_atom in old_molecule.atoms:
        new_atom = Atom(atno=old_atom.Z,
                        x=old_atom.r[0],
                        y=old_atom.r[1],
                        z=old_atom.r[2])
        new_molecule.add_atom(new_atom)

    return new_molecule


def pyquante_2_to_1_basis(old_basis):

    from PyQuante import CGBF

    new_basis = []

    for old_cgbf in old_basis:
        # new_cgbf.norm = old_cgbf.norm
        # conversion from prims?
        # new_cgbf.pnorms = old_cgbf.pnorms
        # new_cgbf.pexps = old_cgbf.pexps
        # new_cgbf.pcoefs = old_cgbf.pcoefs
        new_cgbf = CGBF(old_cgbf.origin, old_cgbf.powers)
        for ccoef, old_pgbf in old_cgbf:
            # new_pgbf = PGBF(exponent=old_pgbf.exponent,
            #                 origin=old_pgbf.origin,
            #                 powers=old_pgbf.powers,
            #                 norm=old_pgbf.norm)
            new_cgbf.add_primitive(exponent=old_pgbf.exponent,
                                   coefficient=ccoef)
        new_basis.append(new_cgbf)

    return new_basis


def make_test_integrals_pyquante(args, mol, mol_basis):
    """A driver for making integrals for test comparisons using PyQuante 1
    and 2.
    """

    solver = pyquante2.uhf(mol, mol_basis)
    solver.converge(tol=1e-10, maxiters=1000)

    i1 = solver.i1

    C = solver.orbsa
    D = np.dot(C[:, :mol.nocc()], C[:, :mol.nocc()].transpose())

    # Save the one-electron integral matrices to disk.
    np.savetxt('pyquante2.S.txt', i1.S)
    np.savetxt('pyquante2.T.txt', i1.T)
    np.savetxt('pyquante2.V.txt', i1.V)
    np.savetxt('pyquante2.D.txt', D)

    # Save the energies to disk.
    np.savetxt('pyquante2.total_energy.txt', np.array([solver.energy]))
    np.savetxt('pyquante2.nuclear_repulsion_energy.txt', np.array([mol.nuclear_repulsion()]))

    # That was all pyquante2 stuff. Now for PyQuante!

    mol1 = pyquante_2_to_1_molecule(mol)
    bfs1 = pyquante_2_to_1_basis(mol_basis.bfs)

    from PyQuante.Ints import getFC

    for iat, at in enumerate(mol1.atoms, 1):
        FC_atom = getFC(bfs1, at.r)
        np.savetxt('PyQuante.FC{}.txt'.format(iat), FC_atom)
