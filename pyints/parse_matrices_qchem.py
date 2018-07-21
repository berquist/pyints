#!/usr/bin/env python

"""parse_matrices_qchem.py: Parse matrices from Q-Chem output files
and save them to disk either as plain text or NumPy binary files.
"""

from __future__ import print_function
from __future__ import division

import os
import re

import numpy as np

from .utils import print_matrices, dump_matrices


def qchem_parse_matrix(outputfile, matrix):
    """Parse a matrix from a Q-Chem output file. `outputfile` is the file
    object being read from. `matrix` is the NumPy array that we modify
    in-place.
    """
    # the shape of the matrix is the same as the dimension of the basis
    dim = matrix.shape[0]
    # The first line will always tell us the maximum number of columns wide
    # a block will be.
    line = next(outputfile)
    ncols = int(line.split()[-1])
    colcounter = 0
    while colcounter < dim:
        if line[:5].strip() == '':
            line = next(outputfile)
        rowcounter = 0
        while rowcounter < dim:
            row = list(map(float, line.split()[1:]))
            matrix[rowcounter][colcounter:colcounter+ncols] = row
            line = next(outputfile)
            rowcounter += 1
        colcounter += ncols
    return matrix


# These are the default matrices printed when calling
# `OneEMtrx().dump()`, plus a few I added in...:)
_MATRIX_HEADERS = (
    # 'Final Alpha MO Coefficients',
    'Final Alpha density matrix.',
    'Final Alpha Fock Matrix',
    'Core Hamiltonian Matrix',
    'Orthonormalization Matrix',
    'Overlap Matrix',
    'Kinetic Energy Matrix',
    'Nuclear Attraction Matrix',
    'Multipole Matrix (1,0,0)',
    'Multipole Matrix (0,1,0)',
    'Multipole Matrix (0,0,1)',
    'Multipole Matrix (2,0,0)',
    'Multipole Matrix (1,1,0)',
    'Multipole Matrix (0,2,0)',
    'Multipole Matrix (1,0,1)',
    'Multipole Matrix (0,1,1)',
    'Multipole Matrix (0,0,2)',
    'Angular Momentum Matrix (X-component)',
    'Angular Momentum Matrix (Y-component)',
    'Angular Momentum Matrix (Z-component)',
    'Spin-Orbit Interaction Matrix (X-component)',
    'Spin-Orbit Interaction Matrix (Y-component)',
    'Spin-Orbit Interaction Matrix (Z-component)'
)


NBASIS_RE_STRING = r'There are(\s+[0-9]+) shells and(\s+[0-9]+) basis functions'


def _varnameify(matname):
    """Turn a Q-Chem matrix header into an appropriate variable name."""
    varname = '_'.join(matname.split())
    varname = varname.replace(',', '').replace('(', '').replace(')', '').replace('.', '').replace('-', '_').lower()
    return varname


def _filenameify(stub, varname):
    """Turn a Q-Chem matrix variable name into an appropriate filename."""
    return '.'.join([stub, varname, 'txt'])


def parse_matrices_qchem(outputfilename, _matrix_headers=_MATRIX_HEADERS):
    """The main routine. Finds matrices with the given headers and parses
    them."""

    stub = os.path.splitext(os.path.basename(outputfilename))[0]

    # Turn the matrix headers into reasonable variable and filenames.
    matrix_headers = dict()
    matrix_filenames = dict()
    for matrix_name in _matrix_headers:
        varname = _varnameify(matrix_name)
        filename = _filenameify(stub, varname)
        matrix_headers[varname] = matrix_name
        matrix_filenames[varname] = filename

    # Determine the size of the basis. Assume N_AO == N_MO!
    with open(outputfilename) as outputfile:
        for line in outputfile:
            match = re.findall(NBASIS_RE_STRING, line)
            if match:
                nbasis = int(match[0][1].strip())

    # Parse the output file for each matrix.
    for matrix_name in matrix_headers:
        with open(outputfilename) as outputfile:
            for line in outputfile:
                if matrix_name in line:
                    matrices[matrix_name] = np.ones(shape=(nbasis, nbasis)) * -99
                    matrices[matrix_name] = qchem_parse_matrix(outputfile, matrices[matrix_name])
                    break

    # Find the total energy and the nuclear repulsion energy.
    # with open(outputfilename) as outputfile:
    #     for line in outputfile:
    #         if 'Nuclear Repulsion Energy' in line:
    #             V_nn = float(line.split()[4])
    #         if 'Total energy in the final basis set' in line:
    #             E_total = float(line.split()[-1])

    # np.savetxt('qchem.nuclear_repulsion_energy.txt', np.array([V_nn]))
    # np.savetxt('qchem.total_energy.txt', np.array([E_total]))

    return matrix_headers, matrix_filenames, matrices


if __name__ == '__main__':

    def getargs():
        """Get and parse command-line arguments."""

        import argparse

        parser = argparse.ArgumentParser()
        arg = parser.add_argument

        arg("outputfilename",
            help="""Name of the Q-Chem output file to parse for matrices.""")
        arg("--print-stdout",
            action="store_true",
            help="""Should all the parsed matrices be printed to stdout (usually
            the screen)?""")

        args = parser.parse_args()

        return args

    args = getargs()
    for outputfilename in args.outputfilename:
        matrix_headers, matrix_filenames, matrices = parse_matrices_qchem(outputfilename)
        if args.print_stdout:
            print('=' * 70)
            print(stub)
            print_matrices(matrix_headers, matrices)
        stub = os.path.splitext(os.path.basename(outputfilename))[0]
        dump_matrices(stub, matrix_filenames, matrices)
