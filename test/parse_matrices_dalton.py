"""parse_matrices_dalton.py: Parse matrices from DALTON output files
and save them to disk either as plain text or NumPy binary files.
"""

from __future__ import print_function

import re
import numpy as np


def getargs():
    """Get and parse command-line arguments."""

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('outputfilename',
                        help="""Name of the DALTON output file to parse for matrices.""")
    parser.add_argument('--print-stdout',
                        action='store_true',
                        help="""Should all the parsed matrices be printed to stdout (usually the
screen)?""")

    args = parser.parse_args()

    return args


def parse_matrix_dalton(outputfile):
    """Parse a matrix from a DALTON output file. `outputfile` is the file
    object being read from.

    Since DALTON doesn't print every line or column if all its
    elements fall below a certain threshold (is there an option for
    this?), we will parse and form a "sparse" (dictionary of
    dictionaries) representation, and separately/later convert this
    sparse representation to dense array format.

    ...which is why there's no matrix passed as an argument.
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
                spmat[rowindex][colindex] = float(element)
        line = next(outputfile)
    return spmat


def sparse_to_dense_matrix_dalton(spmat, dim):
    densemat = np.zeros(shape=(dim, dim))
    for ridx in spmat:
        for cidx in spmat[ridx]:
            densemat[ridx-1][cidx-1] = densemat[cidx-1][ridx-1] = spmat[ridx][cidx]
    return densemat


def parse_spin_orbit_2el(outputfile_reversed, dim, nlines):
    x2spnorb = np.zeros(shape=(dim, dim, dim, dim))
    y2spnorb = np.zeros(shape=(dim, dim, dim, dim))
    z2spnorb = np.zeros(shape=(dim, dim, dim, dim))
    line = ''
    while '##' not in line:
        line = next(outputfile_reversed)
    for _ in range(nlines):
        sline = line.split()
        mu, nu, lm, sg = list(map(int, sline[1:5]))
        component = sline[5][1]
        element = float(sline[7].replace('D', 'E'))
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


def main(args):

    import os.path

    outputfilename = args.outputfilename
    stub = os.path.splitext(outputfilename)[0]

    # We could parse these, but it's pointless since we don't have
    # anything to compare them to from any other program package.
    ignore_these_matrices = (
        'HUCKOVLP',
        'HUCKEL',
    )

    # Find all the matrix headers in an output file and turn them into
    # reasonable variable and filenames.
    matrix_headers_unmodified = []
    matrix_headers_varnames = []
    matrix_headers_filenames = []
    matchline = 'Integrals of operator: '
    with open(outputfilename) as outputfile:
        for line in outputfile:
            if matchline in line:
                if not any(ignore in line for ignore in ignore_these_matrices):
                    # need to handle headers with spaces/multiple parts
                    start = line.index(matchline) + len(matchline)
                    matname = line[start:][:-2].strip()
                    varname = ''.join(matname.split()).replace(',', '_').replace('(', '_').replace(')', '_').replace('.', '_').replace('-', '_').lower()
                    filename = '.'.join([stub, 'integrals_AO_' + varname, 'txt'])
                    matrix_headers_unmodified.append(matname)
                    matrix_headers_varnames.append(varname)
                    matrix_headers_filenames.append(filename)

    nbasis_re_string = 'total:\s+([0-9]+)\s+([-+]?\d*[.]?\d+)\s+([0-9]+)\s+([0-9]+)'

    # Determine the size of the basis. Assume N_AO == N_MO!
    with open(outputfilename) as outputfile:
        for line in outputfile:
            match = re.search(nbasis_re_string, line)
            if match:
                nbasis = int(match.groups()[-1])

    # Parse the output file for each matrix.
    parse_matrix_string = '{matrix_name} = sparse_to_dense_matrix_dalton(parse_matrix_dalton(outputfile), nbasis)'
    save_matrix_string = 'np.savetxt("{file_name}", {matrix_name})'
    for i, matrix_name in enumerate(matrix_headers_unmodified):
        with open(outputfilename) as outputfile:
            for line in outputfile:
                if 'Integrals of operator: {}'.format(matrix_name) in line:
                    exec(parse_matrix_string.format(matrix_name=matrix_headers_varnames[i]), globals(), locals())
                    exec(save_matrix_string.format(matrix_name=matrix_headers_varnames[i],
                                                   file_name=matrix_headers_filenames[i]), globals(), locals())
                    break

    # If present, dump the two-electron spin-orbit integrals.
    with open(outputfilename) as outputfile:
        outputfile_reversed = reversed(outputfile.readlines())
    for line in outputfile_reversed:
        if 'spin-orbit two-electron integrals and' in line:
            nintegrals = int(line.split()[0])
        if 'Number of written 2-el. spin-orbit integrals' in line:
            x2spnorb, y2spnorb, z2spnorb = parse_spin_orbit_2el(outputfile_reversed, nbasis, nintegrals)
            np.save('.'.join([stub, 'integrals_AO_x2spnorb.npy']), x2spnorb)
            np.save('.'.join([stub, 'integrals_AO_y2spnorb.npy']), y2spnorb)
            np.save('.'.join([stub, 'integrals_AO_z2spnorb.npy']), z2spnorb)
            break

    # Print all the parsed matrices to stdout if asked.
    if args.print_stdout:
        for matrix_name in matrix_headers_varnames:
            print_matrix_string = 'print({})'.format(matrix_name)
            print(matrix_name)
            exec(print_matrix_string)

    # Find the total energy and the nuclear repulsion energy.
    with open(outputfilename) as outputfile:
        for line in outputfile:
            if '@    Nuclear repulsion:' in line:
                V_nn = float(line.split()[3])
            if '@     Converged SCF energy' in line:
                E_total = float(line.split()[-2])

    np.savetxt('.'.join([stub, 'nuclear_repulsion_energy.txt']), np.array([V_nn]))
    np.savetxt('.'.join([stub, 'total_energy.txt']), np.array([E_total]))

    return locals()


if __name__ == '__main__':
    args = getargs()
    main_locals = main(args)
