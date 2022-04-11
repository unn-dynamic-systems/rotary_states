from numba import njit
from numba.core.extending import is_jitted
from functools import lru_cache

@lru_cache(maxsize=None)
def do_jit(F):
    return njit(F) if not is_jitted(F) else F
