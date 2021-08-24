import numpy as np


def logistic_map(r, x):
    a = r * x * (1 - x)
    return a


def iterative_f(it, x, r):
    y = logistic_map(r, x)
    if it == 1:
        return [y]
    else:
        return [y] + iterative_f(it - 1, y, r)

