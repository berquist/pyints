from functools import reduce
from itertools import combinations_with_replacement

import numpy as np


def fact2(n):
    return reduce(int.__mul__, range(n, 0, -2), 1)


def is_symmetric(arr):
    return (arr.transpose() == arr).all()


def antisymmetrize_L(mat):
    return np.triu(mat) - np.tril(mat) - np.diag(np.diag(mat))


def antisymmetrize_U(mat):
    return np.tril(mat) - np.triu(mat) - np.diag(np.diag(mat))


def pairs(it):
    return combinations_with_replacement(it, 2)


def iterator4(nbfs):
    for i, j in pairs(range(nbfs)):
        ij = (i*(i+1)/2) + j
        for k, l in pairs(range(nbfs)):
            kl = (k*(k+1)/2) + l
            if ij <= kl:
                yield i, j, k, l
    return


def print_matrices(matrix_headers, matrices): # pragma: no cover
    """Print all the parsed matrices to stdout."""
    for matrix_name in matrix_headers:
        print(matrix_name)
        print(matrices[matrix_name])


def dump_matrices(stub, matrix_filenames, matrices):
    """Save all of the matrices to disk."""
    for matrix_name in matrices:
        if len(matrices[matrix_name].shape) <= 2:
            filename = os.path.join(os.getcwd(),
                                    matrix_filenames[matrix_name])
            np.savetxt(filename, matrices[matrix_name])
        else:
            filename = os.path.join(os.getcwd(),
                                    '.'.join([stub, 'integrals_AO_{}.npy'.format(matrix_name)]))
            np.save(filename, matrices[matrix_name])
