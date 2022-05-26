import math as mt
import numpy as np
from numpy.linalg import eig as get_eigen_vaues_from_M

import os; cwd = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.normpath(os.path.join(cwd, ".."))
import sys; sys.path.append(package_path)

from rh_sides import R_SIDES

# Every import of our library should looks like this
from rotary_states import limit_cycles

def main():
    N, L, G, K, h_k = 6, 0.3, 0.97, 1.33, 1e-3

    T0 = 4.01739166e+00
    IC0 = np.array([0, 4.00489756e+00, 6.83138901e-01, 2.81285526e+00,
                   6.83138901e-01, 2.81285526e+00, 0, 4.00489756e+00,
                   0, 4.00489756e+00, 6.83138901e-01, 2.81285526e+00])
    assert IC0[0] == 0 # Main Convention
    IC, T = IC0, T0
    iteration = 0

    while True:
        K += h_k

        args_orig = (N, L, G, K)
        args_linear = (N, L, K)

        T, IC = limit_cycles.find_limit_cycle(R_SIDES.coupled_pendulums_rs, args_orig, IC, T,
                                              phase_period=4 * mt.pi)

        print("Limit cycle initial condition")
        print(IC)
        print("Limit cycle period")
        print(T)

        M = limit_cycles.get_monogrommy_matrix(R_SIDES.coupled_pendulums_rs,
                                               R_SIDES.coupled_pendulums_linear_rs,
                                               IC, T,
                                               args_linear, args_orig)

        e, _ = get_eigen_vaues_from_M(M)
        print("Eigen values of monogrommy matrix")
        print(e)

        iteration += 1
        if iteration == 10:
            break


if __name__ == "__main__":
    main()
