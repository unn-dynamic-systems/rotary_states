from numba import njit
from numba.core.extending import is_jitted

def do_jit(F):
    return njit(F) if not is_jitted(F) else F
