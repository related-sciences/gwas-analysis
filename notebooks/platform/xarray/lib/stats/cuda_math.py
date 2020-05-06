"""Adaptor for composed CUDA math functions
    
This is necessary because creating a device function will not recursively compile nested functions.

See: https://stackoverflow.com/questions/52807489/programmatic-nested-numba-cuda-function-calls
"""
from . import math
from numba import cuda 

r = cuda.jit(math.r, device=True)
r2 = cuda.jit(lambda gn0, gn1: r(gn0, gn1) ** 2, device=True)
