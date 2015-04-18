from functools import reduce

def fact2(n):
    return reduce(int.__mul__, range(n, 0, -2), 1)


def is_symmetric(arr):
    return (arr.transpose() == arr).all()
