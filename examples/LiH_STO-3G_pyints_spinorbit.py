from __future__ import print_function

import itertools as i

import pyquante2

from pyints.interfaces.obarasaika.two import J2_KF


if __name__ == '__main__':
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
    # for bf in bfs:
    #     print(bf)

    # X non-zero elements (lower triangle):
    # (3, 0), (3, 1), (4, 3), (5, 3)
    # indices = (
    #     (3, 0),
    #     (3, 1),
    #     (4, 3),
    #     (5, 3),
    # )
    # indices = sorted(set(i.permutations([0, 0, 3, 4], 2)))

    # for mu, nu in indices:
    #     print((mu, nu), sum(at.Z * J_KF(bfs[mu], bfs[nu], at.r, 0) for at in mol))

    dim = len(bfs)
    indices = i.product(range(dim), range(dim), range(dim), range(dim))

    # for mu, nu, lm, sg in indices:
    #     element = J2_KF(bfs[mu], bfs[nu], bfs[lm], bfs[sg], 0)
    #     #if abs(element) > 1.0e-16:
    #     print((mu, nu, lm, sg), element)

    # cm = {0: 'a', 3: 'b', 1: 'c', 4: 'd'}
    # indices = sorted(set(i.permutations([0, 1, 3, 4])))
    # for mu, nu, lm, sg in indices:
    #     element = J2_KF(bfs[mu], bfs[nu], bfs[lm], bfs[sg], 0)
    #     print((mu, nu, lm, sg),
    #           ''.join([cm[mu], cm[nu], cm[lm], cm[sg]]),
    #           element)

    indices = sorted(set(i.permutations([2, 3, 4, 5])))
    for mu, nu, lm, sg in indices:
        element = J2_KF(bfs[mu], bfs[nu], bfs[lm], bfs[sg], 2)
        print((mu, nu, lm, sg),
              element)
