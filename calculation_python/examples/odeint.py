import numpy as np
from scipy.integrate import odeint
from rh_sides import R_SIDES


h = 1e-3; t = np.arange(0, 15000, h)
N, L, G, K = 10, 0.4, 0.9, 1
args = (N, L, G, K)

np.random.seed(42); q0 = np.random.rand(2 * N)

odeint(R_SIDES.coupled_pendulums_rs, q0, t, args)
print("ODEINT OK")
