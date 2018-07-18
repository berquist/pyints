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
