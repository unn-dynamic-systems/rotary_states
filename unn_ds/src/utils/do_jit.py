from numba import njit

def do_jit(F):
    return njit(F) if 'targetoptions' not in F.__dict__ else F
