"""Adaptor for composed Numba math functions
    
This is necessary because creating a device function will not recursively compile nested functions.

See: https://stackoverflow.com/questions/52807489/programmatic-nested-numba-cuda-function-calls
"""
from . import math
import numba 

r = numba.njit(math.r, nogil=True)
r2 = numba.njit(lambda gn0, gn1: r(gn0, gn1) ** 2, nogil=True)
