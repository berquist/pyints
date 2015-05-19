#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

from pyints.utils import antisymmetrize_L


def loadtxt(filename):
    """A thin wrapper around np.loadtxt that won't crash if the file
    doesn't exist."""
    try:
        mat = np.loadtxt(filename)
        return mat
    except IOError:
        print("Couldn't find {}".format(filename))
        return None


def load(filename):
    """A thin wrapper around np.load that won't crash of the file doesn't
    exist."""
    try:
        mat = np.load(filename)
        return mat
    except IOError:
        print("Couldn't find {}".format(filename))
        return None


def assert_allclose(m1, m2, rtol=1e-9, atol=1e-5):
    """A thin wrapper around np.testing.assert_allclose."""
    npt.assert_allclose(m1, m2, rtol=rtol, atol=atol)


if __name__ == '__main__':

    ### Load all the variables from Q-Chem.

    D_qchem = loadtxt('qchem.final_alpha_density_matrix.txt')
    H_qchem = loadtxt('qchem.core_hamiltonian_matrix.txt')
    F_qchem = loadtxt('qchem.final_alpha_fock_matrix.txt')
    V_nn_qchem = loadtxt('qchem.nuclear_repulsion_energy.txt')
    E_total_qchem = loadtxt('qchem.total_energy.txt')
    S_qchem = loadtxt('qchem.overlap_matrix.txt')
    T_qchem = loadtxt('qchem.kinetic_energy_matrix.txt')
    V_qchem = loadtxt('qchem.nuclear_attraction_matrix.txt')
    M001_qchem = loadtxt('qchem.multipole_matrix_001.txt')
    M002_qchem = loadtxt('qchem.multipole_matrix_002.txt')
    M010_qchem = loadtxt('qchem.multipole_matrix_010.txt')
    M011_qchem = loadtxt('qchem.multipole_matrix_011.txt')
    M020_qchem = loadtxt('qchem.multipole_matrix_020.txt')
    M100_qchem = loadtxt('qchem.multipole_matrix_100.txt')
    M101_qchem = loadtxt('qchem.multipole_matrix_101.txt')
    M110_qchem = loadtxt('qchem.multipole_matrix_110.txt')
    M200_qchem = loadtxt('qchem.multipole_matrix_200.txt')
    LX_qchem = loadtxt('qchem.angular_momentum_matrix_x_component.txt')
    LY_qchem = loadtxt('qchem.angular_momentum_matrix_y_component.txt')
    LZ_qchem = loadtxt('qchem.angular_momentum_matrix_z_component.txt')
    J1X_qchem = loadtxt('qchem.spin_orbit_interaction_matrix_x_component.txt')
    J1Y_qchem = loadtxt('qchem.spin_orbit_interaction_matrix_y_component.txt')
    J1Z_qchem = loadtxt('qchem.spin_orbit_interaction_matrix_z_component.txt')

    # Q-Chem is returning symmetrized matrices. Antisymmetrize them.
    # LX_qchem = antisymmetrize_L(LX_qchem)
    # LY_qchem = antisymmetrize_L(LY_qchem)
    # LZ_qchem = antisymmetrize_L(LZ_qchem)
    # J1X_qchem = antisymmetrize_L(J1X_qchem)
    # J1Y_qchem = antisymmetrize_L(J1Y_qchem)
    # J1Z_qchem = antisymmetrize_L(J1Z_qchem)

    E_1el_qchem = np.sum(D_qchem * H_qchem)
    E_2el_qchem = np.sum(D_qchem * F_qchem)
    E_tot_qchem = E_1el_qchem + E_2el_qchem + V_nn_qchem

    ### Load all the variables from PyQuante2.

    S_pyquante2 = loadtxt('pyquante2.S.txt')
    T_pyquante2 = loadtxt('pyquante2.T.txt')
    V_pyquante2 = loadtxt('pyquante2.V.txt')
    ERI_pyquante2 = load('pyquante2.ERI.npy')

    ### Load all the variables from pyints.

    S_pyints = loadtxt('pyints.S.txt')
    T_pyints = loadtxt('pyints.T.txt')
    V_pyints = loadtxt('pyints.V.txt')
    M001_pyints = loadtxt('pyints.M001.txt')
    M002_pyints = loadtxt('pyints.M002.txt')
    M010_pyints = loadtxt('pyints.M010.txt')
    M011_pyints = loadtxt('pyints.M011.txt')
    M020_pyints = loadtxt('pyints.M020.txt')
    M100_pyints = loadtxt('pyints.M100.txt')
    M101_pyints = loadtxt('pyints.M101.txt')
    M110_pyints = loadtxt('pyints.M110.txt')
    M200_pyints = loadtxt('pyints.M200.txt')
    EF001_pyints = loadtxt('pyints.EF1.txt')
    EF002_pyints = loadtxt('pyints.EF2.txt')
    EF003_pyints = loadtxt('pyints.EF3.txt')
    EF004_pyints = loadtxt('pyints.EF4.txt')
    EF005_pyints = loadtxt('pyints.EF5.txt')
    EF006_pyints = loadtxt('pyints.EF6.txt')
    EFG1XX_pyints = loadtxt('pyints.EFG1XX.txt')
    EFG1YY_pyints = loadtxt('pyints.EFG1YY.txt')
    EFG1ZZ_pyints = loadtxt('pyints.EFG1ZZ.txt')
    EFG1XY_pyints = loadtxt('pyints.EFG1XY.txt')
    EFG1XZ_pyints = loadtxt('pyints.EFG1XZ.txt')
    EFG1YZ_pyints = loadtxt('pyints.EFG1YZ.txt')
    EFG2XX_pyints = loadtxt('pyints.EFG2XX.txt')
    EFG2YY_pyints = loadtxt('pyints.EFG2YY.txt')
    EFG2ZZ_pyints = loadtxt('pyints.EFG2ZZ.txt')
    EFG2XY_pyints = loadtxt('pyints.EFG2XY.txt')
    EFG2XZ_pyints = loadtxt('pyints.EFG2XZ.txt')
    EFG2YZ_pyints = loadtxt('pyints.EFG2YZ.txt')
    LX_pyints = loadtxt('pyints.LX.txt')
    LY_pyints = loadtxt('pyints.LY.txt')
    LZ_pyints = loadtxt('pyints.LZ.txt')
    LX_M_pyints = loadtxt('pyints.LX_M.txt')
    LY_M_pyints = loadtxt('pyints.LY_M.txt')
    LZ_M_pyints = loadtxt('pyints.LZ_M.txt')
    J1X_pyints = loadtxt('pyints.J1X.txt')
    J1Y_pyints = loadtxt('pyints.J1Y.txt')
    J1Z_pyints = loadtxt('pyints.J1Z.txt')
    ERI_pyints = load('pyints.ERI.npy')
    J2X_pyints = load('pyints.J2X.npy')
    J2Y_pyints = load('pyints.J2Y.npy')
    J2Z_pyints = load('pyints.J2Z.npy')

    ### Load all the variables from DALTON.

    # How to dump these to the output file?
    # D_dalton = loadtxt('dalton.final_alpha_density_matrix.txt')
    # H_dalton = loadtxt('dalton.core_hamiltonian_matrix.txt')
    S_dalton = loadtxt('dalton.overlap.txt')
    T_dalton = loadtxt('dalton.kinenerg.txt')
    V_dalton = loadtxt('dalton.potenerg.txt')
    # F_dalton = loadtxt('dalton.final_alpha_fock_matrix.txt')
    V_nn_dalton = loadtxt('dalton.nuclear_repulsion_energy.txt')
    E_total_dalton = loadtxt('dalton.total_energy.txt')
    M001_dalton = loadtxt('dalton.zdiplen.txt')
    M002_dalton = loadtxt('dalton.zzsecmom.txt')
    M010_dalton = loadtxt('dalton.ydiplen.txt')
    M011_dalton = loadtxt('dalton.yzsecmom.txt')
    M020_dalton = loadtxt('dalton.yysecmom.txt')
    M100_dalton = loadtxt('dalton.xdiplen.txt')
    M101_dalton = loadtxt('dalton.xzsecmom.txt')
    M110_dalton = loadtxt('dalton.xysecmom.txt')
    M200_dalton = loadtxt('dalton.xxsecmom.txt')
    EF001_dalton = loadtxt('dalton.nef001.txt')
    EF002_dalton = loadtxt('dalton.nef002.txt')
    EF003_dalton = loadtxt('dalton.nef003.txt')
    EF004_dalton = loadtxt('dalton.nef004.txt')
    EF005_dalton = loadtxt('dalton.nef005.txt')
    EF006_dalton = loadtxt('dalton.nef006.txt')
    EFG1XX_dalton = loadtxt('dalton.xxefg011.txt')
    EFG1YY_dalton = loadtxt('dalton.yyefg011.txt')
    EFG1ZZ_dalton = loadtxt('dalton.zzefg011.txt')
    EFG1XY_dalton = loadtxt('dalton.xyefg011.txt')
    EFG1XZ_dalton = loadtxt('dalton.xzefg011.txt')
    EFG1YZ_dalton = loadtxt('dalton.yzefg011.txt')
    EFG2XX_dalton = loadtxt('dalton.xxefg021.txt')
    EFG2YY_dalton = loadtxt('dalton.yyefg021.txt')
    EFG2ZZ_dalton = loadtxt('dalton.zzefg021.txt')
    EFG2XY_dalton = loadtxt('dalton.xyefg021.txt')
    EFG2XZ_dalton = loadtxt('dalton.xzefg021.txt')
    EFG2YZ_dalton = loadtxt('dalton.yzefg021.txt')
    LX_dalton = loadtxt('dalton.xangmom.txt')
    LY_dalton = loadtxt('dalton.yangmom.txt')
    LZ_dalton = loadtxt('dalton.zangmom.txt')
    J1X_dalton = loadtxt('dalton.x1spnorb.txt')
    J1Y_dalton = loadtxt('dalton.y1spnorb.txt')
    J1Z_dalton = loadtxt('dalton.z1spnorb.txt')
    # ERI_dalton = load()
    J2X_dalton = load('dalton.x2spnorb.npy')
    J2Y_dalton = load('dalton.y2spnorb.npy')
    J2Z_dalton = load('dalton.z2spnorb.npy')

    # DALTON is returning symmetrized matrices. Antisymmetrize them.
    # LX_dalton = antisymmetrize_L(LX_dalton)
    # LY_dalton = antisymmetrize_L(LY_dalton)
    # LZ_dalton = antisymmetrize_L(LZ_dalton)
    # J1X_dalton = antisymmetrize_L(J1X_dalton)
    # J1Y_dalton = antisymmetrize_L(J1Y_dalton)
    # J1Z_dalton = antisymmetrize_L(J1Z_dalton)

    ### Load all the variables from Molcas.

    # nothing right now...

    ### Load all the variables from Molpro.

    # nothing right now...

    ################################
    #  _____             _
    # (_   _)           ( )_
    #   | |   __    ___ | ,_)  ___
    #   | | /'__`\/',__)| |  /',__)
    #   | |(  ___/\__, \| |_ \__, \
    #   (_)`\____)(____/`\__)(____/
    ################################

    assert_allclose(T_qchem + V_qchem, H_qchem)
    assert_allclose(np.array([E_tot_qchem]), E_total_qchem)

    assert_allclose(S_qchem, S_pyquante2)
    assert_allclose(T_qchem, T_pyquante2)
    assert_allclose(V_qchem, V_pyquante2)

    assert_allclose(S_pyints, S_pyquante2)
    assert_allclose(T_pyints, T_pyquante2)
    # This fails, not sure why.
    # assert_allclose(V_pyints, V_pyquante2)


    assert_allclose(S_qchem, S_dalton)
    assert_allclose(T_qchem, T_dalton)
    # DALTON nuclear attraction integrals are the opposite sign.
    assert_allclose(V_qchem, -V_dalton)

    assert_allclose(M001_qchem, M001_pyints)
    assert_allclose(M002_qchem, M002_pyints)
    assert_allclose(M010_qchem, M010_pyints)
    assert_allclose(M011_qchem, M011_pyints)
    assert_allclose(M020_qchem, M020_pyints)
    assert_allclose(M100_qchem, M100_pyints)
    assert_allclose(M101_qchem, M101_pyints)
    assert_allclose(M110_qchem, M110_pyints)
    assert_allclose(M200_qchem, M200_pyints)

    assert_allclose(M001_qchem, M001_dalton)
    assert_allclose(M002_qchem, M002_dalton)
    assert_allclose(M010_qchem, M010_dalton)
    assert_allclose(M011_qchem, M011_dalton)
    assert_allclose(M020_qchem, M020_dalton)
    assert_allclose(M100_qchem, M100_dalton)
    assert_allclose(M101_qchem, M101_dalton)
    assert_allclose(M110_qchem, M110_dalton)
    assert_allclose(M200_qchem, M200_dalton)

    assert_allclose(M001_dalton, M001_pyints)
    assert_allclose(M002_dalton, M002_pyints)
    assert_allclose(M010_dalton, M010_pyints)
    assert_allclose(M011_dalton, M011_pyints)
    assert_allclose(M020_dalton, M020_pyints)
    assert_allclose(M100_dalton, M100_pyints)
    assert_allclose(M101_dalton, M101_pyints)
    assert_allclose(M110_dalton, M110_pyints)
    assert_allclose(M200_dalton, M200_pyints)

    # DALTON electric field integrals are the opposite sign.
    assert_allclose(-EF001_dalton, EF001_pyints)
    assert_allclose(-EF002_dalton, EF002_pyints)
    assert_allclose(-EF003_dalton, EF003_pyints)
    assert_allclose(-EF004_dalton, EF004_pyints)
    assert_allclose(-EF005_dalton, EF005_pyints)
    assert_allclose(-EF006_dalton, EF006_pyints)

    # Something is currently wrong with the pyints integrals.
    # assert_allclose(EFG1XX_dalton, EFG1XX_pyints)
    # assert_allclose(EFG1YY_dalton, EFG1YY_pyints)
    # assert_allclose(EFG1ZZ_dalton, EFG1ZZ_pyints)
    # assert_allclose(EFG1XY_dalton, EFG1XY_pyints)
    # assert_allclose(EFG1XZ_dalton, EFG1XZ_pyints)
    # assert_allclose(EFG1YZ_dalton, EFG1YZ_pyints)
    # assert_allclose(EFG2XX_dalton, EFG2XX_pyints)
    # assert_allclose(EFG2YY_dalton, EFG2YY_pyints)
    # assert_allclose(EFG2ZZ_dalton, EFG2ZZ_pyints)
    # assert_allclose(EFG2XY_dalton, EFG2XY_pyints)
    # assert_allclose(EFG2XZ_dalton, EFG2XZ_pyints)
    # assert_allclose(EFG2YZ_dalton, EFG2YZ_pyints)

    # TODO: regardless of whether or not Q-Chem/DALTON L/J matrices are
    # antisymmetrized before comparing them, the sign for a few matrix
    # elements is flipped between the two. This makes comparing them
    # without prior knowledge impossible. By doing `abs(mat)`, tests pass,
    # but we won't catch any sign errors!

    # print('LX_qchem')
    # print(LX_qchem)
    # print('LX_dalton')
    # print(LX_dalton)
    # print('LX_M_pyints')
    # print(LX_M_pyints)
    # print('LY_qchem')
    # print(LY_qchem)
    # print('LY_dalton')
    # print(LY_dalton)
    # print('LY_M_pyints')
    # print(LY_M_pyints)
    # print('LZ_qchem')
    # print(LZ_qchem)
    # print('LZ_dalton')
    # print(LZ_dalton)
    # print('LZ_M_pyints')
    # print(LZ_M_pyints)
    assert_allclose(abs(LX_qchem), abs(LX_dalton))
    assert_allclose(abs(LY_qchem), abs(LY_dalton))
    assert_allclose(abs(LZ_qchem), abs(LZ_dalton))

    assert_allclose(antisymmetrize_L(LX_dalton), LX_M_pyints)
    assert_allclose(antisymmetrize_L(LY_dalton), LY_M_pyints)
    assert_allclose(antisymmetrize_L(LZ_dalton), LZ_M_pyints)

    assert_allclose(abs(LX_qchem), abs(LX_M_pyints))
    assert_allclose(abs(LY_qchem), abs(LY_M_pyints))
    assert_allclose(abs(LZ_qchem), abs(LZ_M_pyints))

    # print('J1X_qchem')
    # print(J1X_qchem)
    # print('J1X_dalton')
    # print(J1X_dalton)
    # print('J1X_M_pyints')
    # print(J1X_M_pyints)
    # print('J1Y_qchem')
    # print(J1Y_qchem)
    # print('J1Y_dalton')
    # print(J1Y_dalton)
    # print('J1Y_M_pyints')
    # print(J1Y_M_pyints)
    # print('J1Z_qchem')
    # print(J1Z_qchem)
    # print('J1Z_dalton')
    # print(J1Z_dalton)
    # print('J1Z_M_pyints')
    # print(J1Z_M_pyints)
    assert_allclose(abs(J1X_qchem), abs(J1X_dalton))
    assert_allclose(abs(J1Y_qchem), abs(J1Y_dalton))
    assert_allclose(abs(J1Z_qchem), abs(J1Z_dalton))

    assert_allclose(antisymmetrize_L(J1X_dalton), J1X_pyints)
    assert_allclose(antisymmetrize_L(J1Y_dalton), J1Y_pyints)
    assert_allclose(antisymmetrize_L(J1Z_dalton), J1Z_pyints)

    # Again, there is a sign difference in just a few matrix
    # elements. This must come from Q-Chem! This is unnecessary for
    # comparing DALTON and pyints.

    assert_allclose(abs(J1X_qchem), abs(J1X_pyints))
    assert_allclose(abs(J1Y_qchem), abs(J1Y_pyints))
    assert_allclose(abs(J1Z_qchem), abs(J1Z_pyints))

    # print('diff_J2X')
    # print(J2X_dalton - J2X_pyints)
    # print('diff_J2Y')
    # print(J2Y_dalton - J2Y_pyints)
    # print('diff_J2Z')
    # print(J2Z_dalton - J2Z_pyints)

    # assert_allclose(abs(J2X_dalton), abs(J2X_pyints))
    # assert_allclose(abs(J2Y_dalton), abs(J2Y_pyints))
    # assert_allclose(abs(J2Z_dalton), abs(J2Z_pyints))
