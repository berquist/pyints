from functools import reduce

import numpy as np


def fact2(n):
    return reduce(int.__mul__, range(n, 0, -2), 1)


def is_symmetric(arr):
    return (arr.transpose() == arr).all()


def antisymmetrize_L(mat):
    return np.triu(mat) - np.tril(mat) - np.diag(np.diag(mat))


def antisymmetrize_U(mat):
    return np.tril(mat) - np.triu(mat) - np.diag(np.diag(mat))
