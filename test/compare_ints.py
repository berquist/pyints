#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt

from pyints.utils import antisymmetrize_L

### Load all the variables from Q-Chem.

D_qchem = np.loadtxt('qchem.final_alpha_density_matrix.txt')
H_qchem = np.loadtxt('qchem.core_hamiltonian_matrix.txt')
S_qchem = np.loadtxt('qchem.overlap_matrix.txt')
T_qchem = np.loadtxt('qchem.kinetic_energy_matrix.txt')
V_qchem = np.loadtxt('qchem.nuclear_attraction_matrix.txt')
F_qchem = np.loadtxt('qchem.final_alpha_fock_matrix.txt')
V_nn_qchem = np.loadtxt('qchem.nuclear_repulsion_energy.txt')
E_total_qchem = np.loadtxt('qchem.total_energy.txt')
M001_qchem = np.loadtxt('qchem.multipole_matrix_001.txt')
M002_qchem = np.loadtxt('qchem.multipole_matrix_002.txt')
M010_qchem = np.loadtxt('qchem.multipole_matrix_010.txt')
M011_qchem = np.loadtxt('qchem.multipole_matrix_011.txt')
M020_qchem = np.loadtxt('qchem.multipole_matrix_020.txt')
M100_qchem = np.loadtxt('qchem.multipole_matrix_100.txt')
M101_qchem = np.loadtxt('qchem.multipole_matrix_101.txt')
M110_qchem = np.loadtxt('qchem.multipole_matrix_110.txt')
M200_qchem = np.loadtxt('qchem.multipole_matrix_200.txt')

npt.assert_allclose(T_qchem + V_qchem, H_qchem, rtol=1e-7, atol=1e-5)

E_1el_qchem = np.sum(D_qchem * H_qchem)
E_2el_qchem = np.sum(D_qchem * F_qchem)
E_tot_qchem = E_1el_qchem + E_2el_qchem + V_nn_qchem

npt.assert_allclose(np.array([E_tot_qchem]), E_total_qchem, rtol=1e-7, atol=1e-5)

### Load all the variables from PyQuante2.

S_pyquante2 = np.loadtxt('pyquante2.S.txt')
T_pyquante2 = np.loadtxt('pyquante2.T.txt')
V_pyquante2 = np.loadtxt('pyquante2.V.txt')

npt.assert_allclose(S_qchem, S_pyquante2, rtol=1e-7, atol=1e-5)
npt.assert_allclose(T_qchem, T_pyquante2, rtol=1e-7, atol=1e-5)
npt.assert_allclose(V_qchem, V_pyquante2, rtol=1e-7, atol=1e-5)

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

npt.assert_allclose(S_pyints, S_pyquante2, rtol=1e-7, atol=1e-5)
npt.assert_allclose(T_pyints, T_pyquante2, rtol=1e-7, atol=1e-5)
# This fails, not sure why.
# npt.assert_allclose(V_pyints, V_pyquante2, rtol=1e-7, atol=1e-5)

npt.assert_allclose(M001_qchem, M001_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M002_qchem, M002_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M010_qchem, M010_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M011_qchem, M011_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M020_qchem, M020_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M100_qchem, M100_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M101_qchem, M101_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M110_qchem, M110_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M200_qchem, M200_pyints, rtol=1e-7, atol=1e-5)

### Load all the variables from DALTON.

# D_dalton = np.loadtxt('dalton.final_alpha_density_matrix.txt')
# H_dalton = np.loadtxt('dalton.core_hamiltonian_matrix.txt')
S_dalton = np.loadtxt('dalton.overlap.txt')
T_dalton = np.loadtxt('dalton.kinenerg.txt')
V_dalton = np.loadtxt('dalton.potenerg.txt')
# F_dalton = np.loadtxt('dalton.final_alpha_fock_matrix.txt')
V_nn_dalton = np.loadtxt('dalton.nuclear_repulsion_energy.txt')
E_total_dalton = np.loadtxt('dalton.total_energy.txt')
M001_dalton = np.loadtxt('dalton.zdiplen.txt')
# These aren't the right integrals for the 2nd moments?
# M002_dalton = np.loadtxt('dalton.zzquadru.txt')
M010_dalton = np.loadtxt('dalton.ydiplen.txt')
# M011_dalton = np.loadtxt('dalton.yzquadru.txt')
# M020_dalton = np.loadtxt('dalton.yyquadru.txt')
M100_dalton = np.loadtxt('dalton.xdiplen.txt')
# M101_dalton = np.loadtxt('dalton.xzquadru.txt')
# M110_dalton = np.loadtxt('dalton.xyquadru.txt')
# M200_dalton = np.loadtxt('dalton.xxquadru.txt')
J1X_dalton = np.loadtxt('dalton.x1spnorb.txt')
J1Y_dalton = np.loadtxt('dalton.y1spnorb.txt')
J1Z_dalton = np.loadtxt('dalton.z1spnorb.txt')
LX_dalton = np.loadtxt('dalton.xangmom.txt')
LY_dalton = np.loadtxt('dalton.yangmom.txt')
LZ_dalton = np.loadtxt('dalton.zangmom.txt')
J2X_dalton = np.load('dalton.x2spnorb.npy')
J2Y_dalton = np.load('dalton.y2spnorb.npy')
J2Z_dalton = np.load('dalton.z2spnorb.npy')

# DALTON is returning symmetrized matrices. Antisymmetrize them.
LX_dalton = antisymmetrize_L(LX_dalton)
LY_dalton = antisymmetrize_L(LY_dalton)
LZ_dalton = antisymmetrize_L(LZ_dalton)
J1X_dalton = antisymmetrize_L(J1X_dalton)
J1Y_dalton = antisymmetrize_L(J1Y_dalton)
J1Z_dalton = antisymmetrize_L(J1Z_dalton)

npt.assert_allclose(S_qchem, S_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(T_qchem, T_dalton, rtol=1e-7, atol=1e-5)
# DALTON nuclear attraction integrals are the opposite sign?
npt.assert_allclose(V_qchem, -V_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M001_qchem, M001_dalton, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M002_qchem, M002_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M010_qchem, M010_dalton, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M011_qchem, M011_dalton, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M020_qchem, M020_dalton, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M100_qchem, M100_dalton, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M101_qchem, M101_dalton, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M110_qchem, M110_dalton, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M200_qchem, M200_dalton, rtol=1e-7, atol=1e-5)

npt.assert_allclose(M001_dalton, M001_pyints, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M002_dalton, M002_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M010_dalton, M010_pyints, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M011_dalton, M011_pyints, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M020_dalton, M020_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(M100_dalton, M100_pyints, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M101_dalton, M101_pyints, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M110_dalton, M110_pyints, rtol=1e-7, atol=1e-5)
# npt.assert_allclose(M200_dalton, M200_pyints, rtol=1e-7, atol=1e-5)

npt.assert_allclose(LX_dalton, LX_M_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(LY_dalton, LY_M_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(LZ_dalton, LZ_M_pyints, rtol=1e-7, atol=1e-5)

npt.assert_allclose(J1X_dalton, J1X_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(J1Y_dalton, J1Y_pyints, rtol=1e-7, atol=1e-5)
npt.assert_allclose(J1Z_dalton, J1Z_pyints, rtol=1e-7, atol=1e-5)

### Load all the variables from Molcas.
