
import numpy as np
from numba import njit
import math as mt
# from ..optimizers import newton
from ..integrators import RK4


def create_vf(RS, args, phase_period = 4 * mt.pi): 
    @njit
    def VF(X):
        # usually
        # X = [T, der0, ph1, der1, ph2, der2, ph3, der3, ..., phN, derN]
        # T = X[0]

        X = X.copy()

        T = X[0]; X[0] = 0
        last_state = RK4.last_state(RS, X, 0, T, args)

        phase_period_arr = np.zeros(len(X))
        phase_period_arr[::2] = phase_period

        return last_state - X - phase_period_arr
    return VF
