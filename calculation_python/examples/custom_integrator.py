import os
import pathlib

cwd = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.normpath(os.path.join(cwd, "../.."))
import sys; sys.path.append(package_path)


from numba import njit
import numpy as np
import math as mt

# Every import of our library should looks like this
from calculation_python import integrators

@njit
def RS(q, t, N, L, G, K):
    X = np.empty(2 * N)
    X[0] = q[1]
    X[1] = -L * q[1] - mt.sin(q[0]) + G + K * (mt.sin(q[2] - q[0]))
    n = 0
    while n < 2 * N - 4: 
        X[n + 2] = q[n + 3]
        X[n + 3] = -L * q[n + 3] - mt.sin(q[n + 2]) + G + K * \
            (mt.sin(q[n + 4] - q[n + 2]) + mt.sin(q[n] - q[n + 2]))
        n += 2
    X[2 * N - 2] = q[2 * N - 1]
    X[2 * N - 1] = -L * q[2 * N - 1] - mt.sin(q[2 * N - 2]) + G  \
            + K * (mt.sin(q[2 * N - 4] - q[2 * N - 2]))
    return X

h = 1e-3; t = np.arange(0, 15000, h)
N, L, G, K = 10, 0.4, 0.9, 1
args = (N, L, G, K)

np.random.seed(42); q0 = np.random.rand(2 * N)

integrators.RK4.lastState(RS, q0, t, h, args)
print("CUSTOM INTEGRATOR OK")

