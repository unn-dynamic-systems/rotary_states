from numba import njit

@njit
def lastState(RS, q0, t, h, args):
    X = q0
    for t_i in t:
        X = X + stepRK4(RS, X, t_i, h, args)
    return X

@njit
def stepRK4(RS, X, t_i, h, args):
    k1 = RS(X, t_i, *args)
    k2 = RS(X + h * k1 / 2, t_i + h / 2, *args)
    k3 = RS(X + h * k2 / 2, t_i + h / 2, *args)
    k4 = RS(X + h * k3, t_i + h, *args)
    return h * (k1 + 2 * k2 + 2 * k3 + k4) / 6