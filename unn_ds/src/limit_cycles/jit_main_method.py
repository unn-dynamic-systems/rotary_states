
import numpy as np
import math as mt
from ..integrators import jit_RK4
from numba import njit

TIME_MAX = 5000

@njit
def verify_x(X):
    '''Check the time is valid'''
    return X[0] < TIME_MAX and X[0] > 0

def create_vf(RS, args, phase_period = 2 * mt.pi, h=1e-3):

    @njit
    def VF(X):
        # Convention
        # X = [T, der0, ph1, der1, ph2, der2, ph3, der3, ..., phN, derN]
        # T = X[0]

        X = X.copy()

        T = X[0]; X[0] = 0
        last_state = jit_RK4.jit_last_state(RS, X, 0, T, args, h)

        phase_period_arr = np.zeros(len(X))
        phase_period_arr[::2] = phase_period

        return last_state - X - phase_period_arr
    return VF


def create_super_rs(rs_orig, rs_linear, rs_size):

    @njit
    def RS(q, t, rs_linear_args, rs_orig_args):
        # q = [ linear ... , orig ... ]
        q_linear = q[:rs_size]
        q_orig = q[rs_size:]
        rotate_phases = q_orig[::2] # phases

        lrs = rs_linear(q_linear, t, *rs_linear_args, rotate_phases)
        rs = rs_orig(q_orig, t, *rs_orig_args)

        super_rs = np.zeros(2 * rs_size)
        for i in range(rs_size):
            super_rs[i] = lrs[i]
            super_rs[i + rs_size] = rs[i]
        return super_rs

    return RS
