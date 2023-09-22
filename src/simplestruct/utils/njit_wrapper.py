import functools
import sys

from numba import njit

try:
    import numba

except ImportError:
    pass

def njit_if_loaded(*args, **kwargs):
    if "numba" in sys.modules:
        return njit(*args, **kwargs)
    else:
        return lambda x: x