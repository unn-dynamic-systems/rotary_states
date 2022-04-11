from numba import njit
from numpy.linalg import inv, norm
import numpy as np

@njit
def jit_yacoby_matrix_numeric(VF, X, h=1e-3):
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
def jit_newton(VF, X0, eps, K_MAX, jit_yacoby_matrix, verify_x):
    X = X0
    Y_1 = inv(jit_yacoby_matrix(X))
    d = Y_1 @ VF(X)
    iteration = 0
    while norm(d) > eps and iteration < K_MAX:
        if not verify_x(X):
            raise ArithmeticError("We can't find the point")
        iteration += 1
        Y_1 = inv(jit_yacoby_matrix(X))
        d = Y_1 @ VF(X)
        X -= d
    if iteration == K_MAX:
        raise ArithmeticError("Too much iterations")
    return X
