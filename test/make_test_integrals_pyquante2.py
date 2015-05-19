from __future__ import print_function

import numpy as np

import pyquante2


def make_test_integrals_pyquante2(args, mol, mol_basis):
    """A driver for making integrals for test comparisons using
    pyquante2."""

    solver = pyquante2.uhf(mol, mol_basis)
    solver.converge(tol=1e-10, maxiters=1000)

    print(solver)

    i1 = solver.i1
    # print('Converged overlap matrix:')
    # print(i1.S)
    # print('Converged kinetic energy matrix:')
    # print(i1.T)
    # print('Converged potential energy matrix:')
    # print(i1.V)

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
