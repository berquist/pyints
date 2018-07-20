#!/usr/bin/env python

"""parse_matrices_dalton.py: Parse matrices from DALTON output files
and save them to disk either as plain text or NumPy binary files.
"""

from __future__ import print_function
from __future__ import division

import re
import numpy as np


def getargs():
    """Get and parse command-line arguments."""

    import argparse

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg("outputfilename",
        help="""Name of the DALTON output file to parse for matrices.""")
    arg("--print-stdout",
        action="store_true",
        help="""Should all the parsed matrices be printed to stdout (usually
        the screen)?""")

    args = parser.parse_args()

    return args


# RE_ELEMENT = re.compile(r'(([1-9][0-9]*\.?[0-9]*)|(\.[0-9]+))([EeDd]?[+-]?[0-9]+)?')
# RE_EXPT = re.compile(r'[+-]?\d*$')
RE_ELEMENT = re.compile(r'([+-]?[0-9]*\.?[0-9]*|[+-]?\.[0-9]+)[EeDd]?([+-]?[0-9]+)?')
NBASIS_RE_STRING = r'total:\s+([0-9]+)\s+([-+]?\d*[.]?\d+)\s+([0-9]+)\s+([0-9]+)'


def parse_element_dalton(element):
    """Given a number that might appear in a DALTON output, especially one
    printed in a matrix, convert it to a float.

    The 'expt' regex specifically captures the exponent for when a 'D'
    isn't present in the number.

    Parameters
    ----------
    element : str

    Returns
    -------
    float
    """

    # res = RE_ELEMENT.findall(element)[0][0]
    # expt = RE_EXPT.findall(element)[0][0]

    # return float(res + 'e' + expt)

    if any(x in element for x in ['e', 'E', 'd', 'D']):
        match = [(p, q) for (p, q) in RE_ELEMENT.findall(element)
                 if p is not '' if q is not ''][0]
        return float('e'.join(match))
    else:
        return float(element)


def parse_matrix_dalton(outputfile):
    """Parse a matrix from a DALTON output file. `outputfile` is the file
    object being read from.

    Since DALTON doesn't print every line or column if all its
    elements fall below a certain threshold (is there an option for
    this?), we will parse and form a "sparse" (dictionary of
    dictionaries) representation, and separately/later convert this
    sparse representation to dense array format.

    ...which is why there's no matrix passed as an argument.

    Parameters
    ----------
    outputfile : file handle
        The file handle should be at a position near the desired matrix.

    Returns
    -------
    dict
        A sparse matrix of floats packed into a 2D dictionary, where each
        keys are single indices into non-zero matrix elements.
    """

    spmat = dict()

    line = ''
    while 'Column' not in line:
        line = next(outputfile)
    while '==== End of matrix output ====' not in line:
        chomp = line.split()
        if chomp == []:
            pass
        elif chomp[0] == 'Column':
            colindices = list(map(int, chomp[1::2]))
        else:
            rowindex = int(chomp[0])
            if rowindex not in spmat:
                spmat[rowindex] = dict()
            for colindex, element in zip(colindices, chomp[1:]):
                spmat[rowindex][colindex] = parse_element_dalton(element)
        line = next(outputfile)
    return spmat


def sparse_to_dense_matrix_dalton(spmat, dim):
    """Convert a "sparse" DALTON matrix to its full representation.

    TODO FIXME this almost certainly doesn't work properly for
    antisymmetric matrices or matrices without symmetry.

    Parameters
    ----------
    spmat : dictionary of dictionaries of floats
    dim : int

    Returns
    -------
    np.ndarray
    """
    densemat = np.zeros(shape=(dim, dim))
    for ridx in spmat:
        for cidx in spmat[ridx]:
            densemat[ridx-1][cidx-1] \
                = densemat[cidx-1][ridx-1] \
                = spmat[ridx][cidx]
    return densemat


# pylint: disable=invalid-name
def parse_spin_orbit_2el(outputfile_reversed, dim):
    """Read two-electron spin-orbit integrals from a DALTON output file.

    Parameters
    ----------
    outputfile_reversed : file handle
        The output file is iterative over in reverse order for ease of
        terminating parsing.
    dim : int

    Returns
    -------
    tuple of 3 4-dimensional np.ndarrays, one for each Cartesian
    component
    """
    x2spnorb = np.zeros(shape=(dim, dim, dim, dim))
    y2spnorb = np.zeros(shape=(dim, dim, dim, dim))
    z2spnorb = np.zeros(shape=(dim, dim, dim, dim))
    line = ''
    while '##' not in line:
        line = next(outputfile_reversed)
    while '##' in line:
        tokens = line.split()
        mu, nu, lm, sg = list(map(int, tokens[1:5]))
        component = tokens[5][1]
        element = parse_element_dalton(tokens[7])
        if component == 'X':
            x2spnorb[mu-1, nu-1, lm-1, sg-1] = element
        elif component == 'Y':
            y2spnorb[mu-1, nu-1, lm-1, sg-1] = element
        elif component == 'Z':
            z2spnorb[mu-1, nu-1, lm-1, sg-1] = element
        else:
            raise ValueError("Shouldn't be here...")
        line = next(outputfile_reversed)
    return x2spnorb, y2spnorb, z2spnorb


# We could parse these, but it's pointless since we don't have
# anything to compare them to from any other program package.
IGNORE_THESE_MATRICES = (
    'HUCKOVLP',
    'HUCKEL',
)

def parse_matrices_dalton(outputfilename, dump=False, ignore_headers=IGNORE_THESE_MATRICES):
    """The main routine. Finds all possible integral matrices in a DALTON
    outputfile automatically and parses them.
    """

    import os.path

    stub = os.path.splitext(outputfilename)[0]

    # Find all the matrix headers in an output file and turn them into
    # reasonable variable and filenames.
    matrix_headers_unmodified = []
    matrix_headers_varnames = []
    matrix_headers_filenames = []
    matchline = 'Integrals of operator: '
    with open(outputfilename) as outputfile:
        for line in outputfile:
            if matchline in line:
                if not any(ignore in line for ignore in ignore_headers):
                    # need to handle headers with spaces/multiple parts
                    start = line.index(matchline) + len(matchline)
                    matname = line[start:][:-2].strip()
                    varname = ''.join(matname.split()).replace(',', '_').replace('(', '_').replace(')', '_').replace('.', '_').replace('-', '_').lower()
                    filename = '.'.join([stub, 'integrals_AO_' + varname, 'txt'])
                    matrix_headers_unmodified.append(matname)
                    matrix_headers_varnames.append(varname)
                    matrix_headers_filenames.append(filename)

    # Determine the size of the basis. Assume N_AO == N_MO!
    with open(outputfilename) as outputfile:
        for line in outputfile:
            match = re.search(NBASIS_RE_STRING, line)
            if match:
                nbasis = int(match.groups()[-1])

    # Parse the output file for each matrix.
    parse_matrix_string = '{matrix_name} = sparse_to_dense_matrix_dalton(parse_matrix_dalton(outputfile), nbasis)'
    save_matrix_string = 'np.savetxt("{file_name}", {matrix_name})'
    matrices = dict()
    for i, matrix_name in enumerate(matrix_headers_unmodified):
        with open(outputfilename) as outputfile:
            for line in outputfile:
                if 'Integrals of operator: {}'.format(matrix_name) in line:
                    matrices[matrix_headers_varnames[i]] = sparse_to_dense_matrix_dalton(parse_matrix_dalton(outputfile), nbasis)
                    if dump:
                        np.savetxt(matrix_headers_filenames[i], matrices[matrix_headers_varnames[i]])
                    break

    # If present, dump the two-electron spin-orbit integrals.
    with open(outputfilename) as outputfile:
        outputfile_reversed = reversed(outputfile.readlines())
    for line in outputfile_reversed:
        if 'Number of written 2-el. spin-orbit integrals' in line:
            x2spnorb, y2spnorb, z2spnorb = parse_spin_orbit_2el(outputfile_reversed, nbasis)
            if dump:
                np.save('.'.join([stub, 'integrals_AO_x2spnorb.npy']), x2spnorb)
                np.save('.'.join([stub, 'integrals_AO_y2spnorb.npy']), y2spnorb)
                np.save('.'.join([stub, 'integrals_AO_z2spnorb.npy']), z2spnorb)
            break

    # Print all the parsed matrices to stdout if asked.
    if args.print_stdout:
        for matrix_name in matrix_headers_varnames:
            print(matrix_name)
            print(matrices[matrix_name])

    # Find the total energy and the nuclear repulsion energy.
    with open(outputfilename) as outputfile:
        for line in outputfile:
            if '@    Nuclear repulsion:' in line:
                V_nn = parse_element_dalton(line.split()[3])
            if '@     Converged SCF energy' in line:
                E_total = parse_element_dalton(line.split()[-2])

    if dump:
        np.savetxt('.'.join([stub, 'nuclear_repulsion_energy.txt']), np.array([V_nn]))
        np.savetxt('.'.join([stub, 'total_energy.txt']), np.array([E_total]))

    return locals()


if __name__ == '__main__':
    args = getargs()
    _locals = parse_matrices_dalton(args.outputfilename, dump=True)
