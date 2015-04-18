#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import numpy.linalg as npl
import numpy.testing as npt


D_qchem = np.loadtxt('qchem.final_alpha_density_matrix.txt')
H_qchem = np.loadtxt('qchem.core_hamiltonian_matrix.txt')
S_qchem = np.loadtxt('qchem.overlap_matrix.txt')
T_qchem = np.loadtxt('qchem.kinetic_energy_matrix.txt')
V_qchem = np.loadtxt('qchem.nuclear_attraction_matrix.txt')
F_qchem = np.loadtxt('qchem.final_alpha_fock_matrix.txt')
V_nn_qchem = np.loadtxt('qchem.nuclear_repulsion_energy.txt')
E_total_qchem = np.loadtxt('qchem.total_energy.txt')

npt.assert_allclose(T_qchem + V_qchem, H_qchem, rtol=1e-7, atol=1e-5)

E_1el_qchem = np.sum(D_qchem * H_qchem)
E_2el_qchem = np.sum(D_qchem * F_qchem)
E_tot_qchem = E_1el_qchem + E_2el_qchem + V_nn_qchem

npt.assert_allclose(np.array([E_tot_qchem]), E_total_qchem, rtol=1e-7, atol=1e-5)

S_pyquante2 = np.loadtxt('pyquante2.S.txt')
T_pyquante2 = np.loadtxt('pyquante2.T.txt')
V_pyquante2 = np.loadtxt('pyquante2.V.txt')
