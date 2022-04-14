from numba import njit
from numpy.linalg import inv, norm
import numpy as np

@njit
def __jit_yacoby_matrix_numeric(VF, X, args, h=1e-3):
    N = len(X)
    y_m = np.empty((N, N))
    X_1, X_2 = X.copy(), X.copy()

    for i in range(N):
        X_1[i] -= h
        X_2[i] += h
        y_m[:, i] = (VF(X_2, *args) - VF(X_1, *args)) / 2 / h
        X_1[i] += h
        X_2[i] -= h
    return y_m

@njit
def jit_newton_without_yacoby(VF, X0, eps, K_MAX, args, verify_x):
    X = X0
    Y_1 = inv(__jit_yacoby_matrix_numeric(VF, X, args))
    d = Y_1 @ VF(X, *args)
    iteration = 0
    while norm(d) > eps and iteration < K_MAX:
        if not verify_x(X):
            raise ArithmeticError("We can't find the point")
        iteration += 1
        Y_1 = inv(__jit_yacoby_matrix_numeric(VF, X, args))
        d = Y_1 @ VF(X, *args)
        X -= d
    if iteration == K_MAX:
        raise ArithmeticError("Too much iterations")
    return X

@njit
def jit_newton_with_yacoby(VF, X0, eps, K_MAX, args, jit_yacoby_matrix, verify_x):
    X = X0
    Y_1 = inv(jit_yacoby_matrix(X))
    d = Y_1 @ VF(X, *args)
    iteration = 0
    while norm(d) > eps and iteration < K_MAX:
        if not verify_x(X):
            raise ArithmeticError("We can't find the point")
        iteration += 1
        Y_1 = inv(jit_yacoby_matrix(X))
        d = Y_1 @ VF(X, *args)
        X -= d
    if iteration == K_MAX:
        raise ArithmeticError("Too much iterations")
    return X
