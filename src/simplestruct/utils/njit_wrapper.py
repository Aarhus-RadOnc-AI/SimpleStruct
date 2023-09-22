try:
    from numba import njit
except ImportError:
    from ._shim import njit
