from __future__ import print_function

import re
import numpy as np


def getargs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('outputfilename')
    parser.add_argument('--print-stdout', action='store_true')
    args = parser.parse_args()
    return args


def parse_matrix_dalton(outputfile):
    """Since DALTON doesn't print every line or column if all its
    elements fall below a certain threshold (is there an option for
    this?), we will parse and form a "sparse" (dictionary of
    dictionaries) representation, and separately/later convert this
    sparse representation to dense array format.
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
            colindices = map(int, chomp[1::2])
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


def main(args):

    outputfilename = args.outputfilename


    matrix_headers = (
        # default
        'OVERLAP',
        # .DIPLEN
        'XDIPLEN',
        'YDIPLEN',
        'ZDIPLEN',
        # .QUADRU
        'XXQUADRU',
        'XYQUADRU',
        'XZQUADRU',
        'YYQUADRU',
        'YZQUADRU',
        'ZZQUADRU',
        # .KINENE
        'KINENERG',
        # .DIPVEL
        'XDIPVEL',
        'YDIPVEL',
        'ZDIPVEL',
        # .1ELPOT
        'POTENERG',
        # .SPIN-ORBIT
        'X1SPNORB',
        'Y1SPNORB',
        'Z1SPNORB',
        # .ANGMOM
        'XANGMOM',
        'YANGMOM',
        'ZANGMOM'
    )

    # turn the matrix headers into reasonable variable and filenames
    matrix_headers_varnames = []
    matrix_headers_filenames = []
    for matrix_name in matrix_headers:
        newname = '_'.join(matrix_name.split())
        newname = newname.replace(',', '').replace('(', '').replace(')', '').replace('.', '').lower()
        filename = '.'.join(['dalton', newname, 'txt'])
        matrix_headers_varnames.append(newname)
        matrix_headers_filenames.append(filename)

    nbasis_re_string = 'total:\s+([0-9]+)\s+([-+]?\d*[.]?\d+)\s+([0-9]+)\s+([0-9]+)'

    # determine the size of the basis
    with open(outputfilename) as outputfile:
        for line in outputfile:
            match = re.search(nbasis_re_string, line)
            if match:
                nbasis = int(match.groups()[-1])

    # parse the output file for each matrix
    parse_matrix_string = '{matrix_name} = sparse_to_dense_matrix_dalton(parse_matrix_dalton(outputfile), nbasis)'
    save_matrix_string = 'np.savetxt("{file_name}", {matrix_name})'
    for i, matrix_name in enumerate(matrix_headers):
        with open(outputfilename) as outputfile:
            for line in outputfile:
                if 'Integrals of operator: {}'.format(matrix_name) in line:
                    exec(parse_matrix_string.format(matrix_name=matrix_headers_varnames[i]))
                    exec(save_matrix_string.format(matrix_name=matrix_headers_varnames[i],
                                                   file_name=matrix_headers_filenames[i]))
                    break

    # print all the parsed matrices to stdout if asked
    if args.print_stdout:
        for matrix_name in matrix_headers_varnames:
            print_matrix_string = 'print({})'.format(matrix_name)
            print(matrix_name)
            exec(print_matrix_string)

    # Find the total energy and the nuclear repulsion energy
    with open(outputfilename) as outputfile:
        for line in outputfile:
            if '@    Nuclear repulsion:' in line:
                V_nn = float(line.split()[3])
            if '@     Converged SCF energy' in line:
                E_total = float(line.split()[-2])

    np.savetxt('dalton.nuclear_repulsion_energy.txt', np.array([V_nn]))
    np.savetxt('dalton.total_energy.txt', np.array([E_total]))

    return locals()


if __name__ == '__main__':
    args = getargs()
    main_locals = main(args)
