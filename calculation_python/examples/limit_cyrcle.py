import numpy as np
from numba import njit
import math as mt
from numpy.linalg import eig as get_eigen_vaues

import os; cwd = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.normpath(os.path.join(cwd, "..", ".."))
import sys; sys.path.append(package_path)

from rh_sides import R_SIDES

# Every import of our library should looks like this
from calculation_python import optimizers, integrators

def create_vf(rs, integrate_method, args):
    @njit
    def VF(X):
        # X = [T, der0, ph1, der1, ph2, der2, ph3, der3, ..., phN, derN]
        # T = X[0]

        X = X.copy()
        period = 4 * mt.pi
        h = 1e-3

        T = X[0]; X[0] = 0
        last_state = integrate_method(rs, X, 0, h, T, args)

        period_arr = np.zeros(len(X))
        period_arr[::2] = period

        return last_state - X - period_arr
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

def get_monogrommy_matrix(SUPER_RS, q0_limit_cycle, T, SUPER_ARGS, h=1e-3):
    N = len(q0_limit_cycle)
    M = np.empty((N, N))
    for i in range(N):
        q0_linear = np.zeros(N); q0_linear[i] = 1.0
        q0 = np.array([q0_linear, q0_limit_cycle]).flatten()
        M[i] = integrators.RK4.last_state(SUPER_RS, q0, 0, h, T, SUPER_ARGS)[:N]
    return M

def main():
    N, L, G, K, h_k = 6, 0.3, 0.97, 1.33, 1e-3

    X0 = np.array([4.01739166e+00, 4.00489756e+00, 6.83138901e-01, 2.81285526e+00,
                   6.83138901e-01, 2.81285526e+00, 0, 4.00489756e+00,
                   0, 4.00489756e+00, 6.83138901e-01, 2.81285526e+00])

    x_limit_cycle = X0
    iteration = 0

    while True:
        K += h_k

        args_orig = (N, L, G, K)
        args_linear = (N, L, K)

        print("args_orig: ", args_orig)
        VF = create_vf(R_SIDES.coupled_pendulums_rs, integrators.RK4.last_state, args_orig)

        # get limit cycle initial condition
        x_limit_cycle = optimizers.newton(VF, x_limit_cycle, eps=1e-3)
        print("Limit cycle initial condition")
        print(x_limit_cycle)

        T = x_limit_cycle[0]; x_limit_cycle[0] = 0

        SUPER_RS = create_super_rs(rs_orig=R_SIDES.coupled_pendulums_rs,
                                rs_linear=R_SIDES.coupled_pendulums_linear_rs,
                                rs_size=2 * N)

        M = get_monogrommy_matrix(SUPER_RS, x_limit_cycle, T, (args_linear, args_orig))
        e, _ = get_eigen_vaues(M)
        print("Eigen values of monogrommy matrix")
        print(e)

        x_limit_cycle[0] = T
        
        iteration += 1
        if iteration == 10:
            break


if __name__ == "__main__":
    main()
