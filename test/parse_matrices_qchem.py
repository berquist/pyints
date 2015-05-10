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


def qchem_parse_matrix(outputfile, matrix):
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


def main(args):

    outputfilename = args.outputfilename

    # These are the default matrices printed when calling OneEMtrx().dump()
    matrix_headers = (
#        'Final Alpha MO Coefficients',
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

    # turn the matrix headers into reasonable variable and filenames
    matrix_headers_varnames = []
    matrix_headers_filenames = []
    for matrix_name in matrix_headers:
        newname = '_'.join(matrix_name.split())
        newname = newname.replace(',', '').replace('(', '').replace(')', '').replace('.', '').replace('-', '_').lower()
        filename = '.'.join(['qchem', newname, 'txt'])
        matrix_headers_varnames.append(newname)
        matrix_headers_filenames.append(filename)

    nbasis_re_string = 'There are(\s+[0-9]+) shells and(\s+[0-9]+) basis functions'

    # determine the size of the basis
    with open(outputfilename) as outputfile:
        for line in outputfile:
            match = re.findall(nbasis_re_string, line)
            if match:
                nbasis = int(match[0][1].strip())

    # preallocate the appropriate matrices
    for matrix_name in matrix_headers_varnames:
        exec_string = '{matrix_name} = np.ones(shape=({nbasis}, {nbasis})) * -99'
        exec(exec_string.format(matrix_name=matrix_name, nbasis=nbasis))


    # parse the output file for each matrix
    parse_matrix_string = '{matrix_name} = qchem_parse_matrix(outputfile, {matrix_name})'
    save_matrix_string = 'np.savetxt("{file_name}", {matrix_name})'
    for i, matrix_name in enumerate(matrix_headers):
        with open(outputfilename) as outputfile:
            for line in outputfile:
                if matrix_name in line:
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
            if 'Nuclear Repulsion Energy' in line:
                V_nn = float(line.split()[4])
            if 'Total energy in the final basis set' in line:
                E_total = float(line.split()[-1])

    np.savetxt('qchem.nuclear_repulsion_energy.txt', np.array([V_nn]))
    np.savetxt('qchem.total_energy.txt', np.array([E_total]))


if __name__ == '__main__':
    args = getargs()
    main(args)
