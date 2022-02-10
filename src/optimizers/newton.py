from numba import njit
from numpy.linalg import inv, norm
import numpy as np

@njit
def yacoby_matrix(VF, X, h=1e-3):
    '''
    Get vector function VF and return yacoby matrix in point X
    '''
    N = len(X)
    y_m = np.empty((N, N))
    X_1, X_2 = X.copy(), X.copy()

    for i in range(N):
        X_1[i] -= h
        X_2[i] += h
        y_m[:, i] = (VF(X_2) - VF(X_1)) / 2 / h
        X_1[i] += h
        X_2[i] -= h
    return y_m

@njit
def newton(VF, X0, eps=1e-3, K_MAX=100):
    '''Explanation here https://www.wikiwand.com/en/Newton%27s_method'''
    X = X0
    Y_1 = inv(yacoby_matrix(VF, X))
    d = Y_1 @ VF(X)
    iteration = 0
    while norm(d) > eps and iteration < K_MAX:
        iteration += 1
        Y_1 = inv(yacoby_matrix(VF, X))
        d = Y_1 @ VF(X)
        X -= d
    return X
