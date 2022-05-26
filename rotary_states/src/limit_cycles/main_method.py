
import numpy as np
import math as mt
from ..optimizers import newton as __newton
from ..integrators import jit_RK4 as __jit_RK4

from ..utils.do_jit import do_jit as __do_jit

from .jit_main_method import vf_special as __vf_special
from .jit_main_method import super_rs as __super_rs
from .jit_main_method import verify_x as __verify_x 

def find_limit_cycle(RS, args, IC0, T0, phase_period = 2 * mt.pi, h=1e-3, eps=1e-3):
    '''TODO: Docs'''

    IC0 = IC0.copy()
    assert IC0[0] == 0 # Main Convention
    IC0[0] = T0
    # too slow :(
    # IC0 = __newton(lambda X: VF(X) ** 2, IC0, 1e-1, verify_x=__verify_x)
    IC = __newton(__vf_special, IC0, eps, args=(__do_jit(RS), args, phase_period, h), verify_x=__verify_x)
    T = IC[0]; IC[0] = 0
    return T, IC

def get_monogrommy_matrix(rs_orig, rs_linear,
                          q0_limit_cycle, T,
                          args_linear, args_orig,
                          h_integrate=1e-3):

    '''TODO: Docs'''
    RS_SIZE = len(q0_limit_cycle)
    SUPER_RS = __super_rs
    SUPER_ARGS = (__do_jit(rs_orig), __do_jit(rs_linear), args_linear, args_orig, RS_SIZE)

    M = np.empty((RS_SIZE, RS_SIZE))
    for i in range(RS_SIZE):
        q0_linear = np.zeros(RS_SIZE); q0_linear[i] = 1.0
        q0 = np.array([q0_linear, q0_limit_cycle]).flatten()
        M[i] = __jit_RK4.jit_last_state(SUPER_RS, q0, 0, T, SUPER_ARGS, h_integrate)[:RS_SIZE]
    return M
