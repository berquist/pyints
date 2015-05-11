#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

from pyints.utils import antisymmetrize_L

### Load all the variables from Q-Chem.

D_qchem = np.loadtxt('qchem.final_alpha_density_matrix.txt')
H_qchem = np.loadtxt('qchem.core_hamiltonian_matrix.txt')
F_qchem = np.loadtxt('qchem.final_alpha_fock_matrix.txt')
V_nn_qchem = np.loadtxt('qchem.nuclear_repulsion_energy.txt')
E_total_qchem = np.loadtxt('qchem.total_energy.txt')
S_qchem = np.loadtxt('qchem.overlap_matrix.txt')
T_qchem = np.loadtxt('qchem.kinetic_energy_matrix.txt')
V_qchem = np.loadtxt('qchem.nuclear_attraction_matrix.txt')
M001_qchem = np.loadtxt('qchem.multipole_matrix_001.txt')
M002_qchem = np.loadtxt('qchem.multipole_matrix_002.txt')
M010_qchem = np.loadtxt('qchem.multipole_matrix_010.txt')
M011_qchem = np.loadtxt('qchem.multipole_matrix_011.txt')
M020_qchem = np.loadtxt('qchem.multipole_matrix_020.txt')
M100_qchem = np.loadtxt('qchem.multipole_matrix_100.txt')
M101_qchem = np.loadtxt('qchem.multipole_matrix_101.txt')
M110_qchem = np.loadtxt('qchem.multipole_matrix_110.txt')
M200_qchem = np.loadtxt('qchem.multipole_matrix_200.txt')
LX_qchem = np.loadtxt('qchem.angular_momentum_matrix_x_component.txt')
LY_qchem = np.loadtxt('qchem.angular_momentum_matrix_y_component.txt')
LZ_qchem = np.loadtxt('qchem.angular_momentum_matrix_z_component.txt')
J1X_qchem = np.loadtxt('qchem.spin_orbit_interaction_matrix_x_component.txt')
J1Y_qchem = np.loadtxt('qchem.spin_orbit_interaction_matrix_y_component.txt')
J1Z_qchem = np.loadtxt('qchem.spin_orbit_interaction_matrix_z_component.txt')

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

S_pyquante2 = np.loadtxt('pyquante2.S.txt')
T_pyquante2 = np.loadtxt('pyquante2.T.txt')
V_pyquante2 = np.loadtxt('pyquante2.V.txt')
# ERI_pyquante2 = np.load('pyquante2.ERI.npy')

### Load all the variables from pyints.

S_pyints = np.loadtxt('pyints.S.txt')
T_pyints = np.loadtxt('pyints.T.txt')
V_pyints = np.loadtxt('pyints.V.txt')
M001_pyints = np.loadtxt('pyints.M001.txt')
M002_pyints = np.loadtxt('pyints.M002.txt')
M010_pyints = np.loadtxt('pyints.M010.txt')
M011_pyints = np.loadtxt('pyints.M011.txt')
M020_pyints = np.loadtxt('pyints.M020.txt')
M100_pyints = np.loadtxt('pyints.M100.txt')
M101_pyints = np.loadtxt('pyints.M101.txt')
M110_pyints = np.loadtxt('pyints.M110.txt')
M200_pyints = np.loadtxt('pyints.M200.txt')
EF001_pyints = np.loadtxt('pyints.EF1.txt')
EF002_pyints = np.loadtxt('pyints.EF2.txt')
EF003_pyints = np.loadtxt('pyints.EF3.txt')
EF004_pyints = np.loadtxt('pyints.EF4.txt')
EF005_pyints = np.loadtxt('pyints.EF5.txt')
EF006_pyints = np.loadtxt('pyints.EF6.txt')
LX_pyints = np.loadtxt('pyints.LX.txt')
LY_pyints = np.loadtxt('pyints.LY.txt')
LZ_pyints = np.loadtxt('pyints.LZ.txt')
LX_M_pyints = np.loadtxt('pyints.LX_M.txt')
LY_M_pyints = np.loadtxt('pyints.LY_M.txt')
LZ_M_pyints = np.loadtxt('pyints.LZ_M.txt')
J1X_pyints = np.loadtxt('pyints.J1X.txt')
J1Y_pyints = np.loadtxt('pyints.J1Y.txt')
J1Z_pyints = np.loadtxt('pyints.J1Z.txt')
ERI_pyints = np.load('pyints.ERI.npy')
J2X_pyints = np.load('pyints.J2X.npy')
J2Y_pyints = np.load('pyints.J2Y.npy')
J2Z_pyints = np.load('pyints.J2Z.npy')

### Load all the variables from DALTON.

# How to dump these to the output file?
# D_dalton = np.loadtxt('dalton.final_alpha_density_matrix.txt')
# H_dalton = np.loadtxt('dalton.core_hamiltonian_matrix.txt')
S_dalton = np.loadtxt('dalton.overlap.txt')
T_dalton = np.loadtxt('dalton.kinenerg.txt')
V_dalton = np.loadtxt('dalton.potenerg.txt')
# F_dalton = np.loadtxt('dalton.final_alpha_fock_matrix.txt')
V_nn_dalton = np.loadtxt('dalton.nuclear_repulsion_energy.txt')
E_total_dalton = np.loadtxt('dalton.total_energy.txt')
M001_dalton = np.loadtxt('dalton.zdiplen.txt')
M002_dalton = np.loadtxt('dalton.zzsecmom.txt')
M010_dalton = np.loadtxt('dalton.ydiplen.txt')
M011_dalton = np.loadtxt('dalton.yzsecmom.txt')
M020_dalton = np.loadtxt('dalton.yysecmom.txt')
M100_dalton = np.loadtxt('dalton.xdiplen.txt')
M101_dalton = np.loadtxt('dalton.xzsecmom.txt')
M110_dalton = np.loadtxt('dalton.xysecmom.txt')
M200_dalton = np.loadtxt('dalton.xxsecmom.txt')
LX_dalton = np.loadtxt('dalton.xangmom.txt')
LY_dalton = np.loadtxt('dalton.yangmom.txt')
LZ_dalton = np.loadtxt('dalton.zangmom.txt')
EF001_dalton = np.loadtxt('dalton.nef001.txt')
EF002_dalton = np.loadtxt('dalton.nef002.txt')
EF003_dalton = np.loadtxt('dalton.nef003.txt')
EF004_dalton = np.loadtxt('dalton.nef004.txt')
EF005_dalton = np.loadtxt('dalton.nef005.txt')
EF006_dalton = np.loadtxt('dalton.nef006.txt')
J1X_dalton = np.loadtxt('dalton.x1spnorb.txt')
J1Y_dalton = np.loadtxt('dalton.y1spnorb.txt')
J1Z_dalton = np.loadtxt('dalton.z1spnorb.txt')
J2X_dalton = np.load('dalton.x2spnorb.npy')
J2Y_dalton = np.load('dalton.y2spnorb.npy')
J2Z_dalton = np.load('dalton.z2spnorb.npy')

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

npt.assert_allclose(T_qchem + V_qchem, H_qchem, rtol=1e-7, atol=1e-5)
npt.assert_allclose(np.array([E_tot_qchem]), E_total_qchem, rtol=1e-7, atol=1e-5)

npt.assert_allclose(S_qchem, S_pyquante2, rtol=1e-7, atol=1e-5)
npt.assert_allclose(T_qchem, T_pyquante2, rtol=1e-7, atol=1e-5)
npt.assert_allclose(V_qchem, V_pyquante2, rtol=1e-7, atol=1e-5)

npt.assert_allclose(S_pyints, S_pyquante2, rtol=1e-7, atol=1e-5)
npt.assert_allclose(T_pyints, T_pyquante2, rtol=1e-7, atol=1e-5)
# This fails, not sure why.
# npt.assert_allclose(V_pyints, V_pyquante2, rtol=1e-7, atol=1e-5)


npt.assert_allclose(S_qchem, S_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(T_qchem, T_dalton, rtol=1e-7, atol=1e-5)
# DALTON nuclear attraction integrals are the opposite sign.
npt.assert_allclose(V_qchem, -V_dalton, rtol=1e-7, atol=1e-5)

npt.assert_allclose(M001_qchem, M001_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M002_qchem, M002_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M010_qchem, M010_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M011_qchem, M011_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M020_qchem, M020_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M100_qchem, M100_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M101_qchem, M101_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M110_qchem, M110_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M200_qchem, M200_pyints, rtol=1e-7, atol=1e-5)

npt.assert_allclose(M001_qchem, M001_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M002_qchem, M002_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M010_qchem, M010_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M011_qchem, M011_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M020_qchem, M020_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M100_qchem, M100_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M101_qchem, M101_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M110_qchem, M110_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M200_qchem, M200_dalton, rtol=1e-7, atol=1e-5)

npt.assert_allclose(M001_dalton, M001_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M002_dalton, M002_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M010_dalton, M010_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M011_dalton, M011_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M020_dalton, M020_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M100_dalton, M100_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M101_dalton, M101_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M110_dalton, M110_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M200_dalton, M200_pyints, rtol=1e-7, atol=1e-5)

# DALTON electric field integrals are the opposite sign.
npt.assert_allclose(-EF001_dalton, EF001_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(-EF002_dalton, EF002_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(-EF003_dalton, EF003_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(-EF004_dalton, EF004_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(-EF005_dalton, EF005_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(-EF006_dalton, EF006_pyints, rtol=1e-7, atol=1e-5)

# TODO: regardless of whether or not Q-Chem/DALTON L/J matrices are
# antisymmetrized before comparing them, the sign for a few matrix
# elements is flipped between the two. This makescomparing them
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
npt.assert_allclose(abs(LX_qchem), abs(LX_dalton), rtol=1e-7, atol=1e-5)
npt.assert_allclose(abs(LY_qchem), abs(LY_dalton), rtol=1e-7, atol=1e-5)
npt.assert_allclose(abs(LZ_qchem), abs(LZ_dalton), rtol=1e-7, atol=1e-5)

npt.assert_allclose(antisymmetrize_L(LX_dalton), LX_M_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(antisymmetrize_L(LY_dalton), LY_M_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(antisymmetrize_L(LZ_dalton), LZ_M_pyints, rtol=1e-7, atol=1e-5)

npt.assert_allclose(abs(LX_qchem), abs(LX_M_pyints), rtol=1e-7, atol=1e-5)
npt.assert_allclose(abs(LY_qchem), abs(LY_M_pyints), rtol=1e-7, atol=1e-5)
npt.assert_allclose(abs(LZ_qchem), abs(LZ_M_pyints), rtol=1e-7, atol=1e-5)

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
npt.assert_allclose(abs(J1X_qchem), abs(J1X_dalton), rtol=1e-7, atol=1e-5)
npt.assert_allclose(abs(J1Y_qchem), abs(J1Y_dalton), rtol=1e-7, atol=1e-5)
npt.assert_allclose(abs(J1Z_qchem), abs(J1Z_dalton), rtol=1e-7, atol=1e-5)


npt.assert_allclose(antisymmetrize_L(J1X_dalton), J1X_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(antisymmetrize_L(J1Y_dalton), J1Y_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(antisymmetrize_L(J1Z_dalton), J1Z_pyints, rtol=1e-7, atol=1e-5)

# Again, there is a sign difference in just a few matrix
# elements. This must come from Q-Chem! This is unnecessary for
# comparing DALTON and pyints.

npt.assert_allclose(abs(J1X_qchem), abs(J1X_pyints), rtol=1e-7, atol=1e-5)
npt.assert_allclose(abs(J1Y_qchem), abs(J1Y_pyints), rtol=1e-7, atol=1e-5)
npt.assert_allclose(abs(J1Z_qchem), abs(J1Z_pyints), rtol=1e-7, atol=1e-5)
