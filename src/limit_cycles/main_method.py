
import numpy as np
from numba import njit
import math as mt
from ..optimizers import newton
from ..integrators import RK4


def create_vf(RS, args, phase_period = 4 * mt.pi, h=1e-3):
    '''TODO: Docs'''
    @njit
    def VF(X):
        # Convention
        # X = [T, der0, ph1, der1, ph2, der2, ph3, der3, ..., phN, derN]
        # T = X[0]

        X = X.copy()

        T = X[0]; X[0] = 0
        last_state = RK4.last_state(RS, X, 0, T, args, h)

        phase_period_arr = np.zeros(len(X))
        phase_period_arr[::2] = phase_period

        return last_state - X - phase_period_arr
    return VF

def find_limit_cycle(RS, args, IC0, T0, phase_period = 4 * mt.pi, h=1e-3, eps=1e-3):
    '''TODO: Docs'''
    assert IC0[0] == 0 # Main Convention
    IC0[0] = T0
    VF = create_vf(RS, args, phase_period, h)
    IC = newton(VF, IC0, eps)
    T = IC[0]; IC[0] = 0
    return T, IC

def create_super_rs(rs_orig, rs_linear, rs_size):
    '''TODO: Docs'''
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

def get_monogrommy_matrix(rs_orig, rs_linear,
                          q0_limit_cycle, T,
                          args_linear, args_orig):
    '''TODO: Docs'''
    RS_SIZE = len(q0_limit_cycle)
    SUPER_RS = create_super_rs(rs_orig, rs_linear, RS_SIZE)
    SUPER_ARGS = (args_linear, args_orig)


    M = np.empty((RS_SIZE, RS_SIZE))
    for i in range(RS_SIZE):
        q0_linear = np.zeros(RS_SIZE); q0_linear[i] = 1.0
        q0 = np.array([q0_linear, q0_limit_cycle]).flatten()
        M[i] = RK4.last_state(SUPER_RS, q0, 0, T, SUPER_ARGS)[:RS_SIZE]
    return M
