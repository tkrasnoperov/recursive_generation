from sys import exit
from time import time
import numpy as np
from numpy.linalg import pinv, norm
from scipy.linalg import null_space

def solution_space(A, y):
    ''' Solves Ax = y '''
    A_pinv = pinv(A)
    n  = A_pinv.shape[1]

    if norm(A.dot(A_pinv).dot(y) - y) > .00001:
        raise BaseException("impossible y value")

    I = np.identity(n)

    x = np.dot(A_pinv, y)
    null_basis = null_space(A).T

    return x, null_basis

def check_solution_space(A, x, null_basis, y, threshold=.001, n_checks=100):
    for _ in range(n_checks):
        xk = x + 0
        for v in null_basis:
            c = 2 * (np.random.random() - .5)
            xk += c * v

        error = norm(A.dot(xk) - y)
        if error > threshold:
            raise BaseException("invalid solution space")
