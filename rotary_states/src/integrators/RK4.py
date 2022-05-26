from .. utils.do_jit import do_jit as __do_jit
from . jit_RK4 import jit_last_state as __jit_last_state

def last_state(RS, q0, t0, t_end, args = (), h = 1e-3):
    return __jit_last_state(__do_jit(RS), q0, t0, t_end, args, h)
