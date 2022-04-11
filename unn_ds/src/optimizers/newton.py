from . jit_newton import jit_newton as __jit_newton, jit_newton_without_yacoby as __jit_newton_without_yacoby
from .. utils.do_jit import do_jit as __do_jit

def newton(VF, X0, eps=1e-3, K_MAX=100, yacoby_matrix = False, verify_x=lambda _: True):
    '''Explanation here https://www.wikiwand.com/en/Newton%27s_method'''
    
    if not yacoby_matrix:
        return __jit_newton_without_yacoby(__do_jit(VF), X0, eps, K_MAX, __do_jit(verify_x))

    return __jit_newton(__do_jit(VF), X0, eps, K_MAX, __do_jit(yacoby_matrix), __do_jit(verify_x))
