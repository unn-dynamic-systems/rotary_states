import os
import numpy as np
from rh_sides import R_SIDES

cwd = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.normpath(os.path.join(cwd, ".."))
import sys; sys.path.append(package_path)

# Every import of our library should looks like this
from unn_ds import integrators


t_0, t_end = 0, 5000
N, L, G, K = 10, 0.4, 0.9, 1
args = (N, L, G, K)

np.random.seed(42); q0 = np.random.rand(2 * N)

integrators.RK4.last_state(R_SIDES.coupled_pendulums_rs, q0, t_0, t_end, args)
print("CUSTOM INTEGRATOR OK")
