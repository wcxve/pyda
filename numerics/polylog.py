# -*- coding: utf-8 -*-
"""

@author: xuewc
"""

from numba import njit, vectorize

@vectorize(nopython=True)
@njit('float64(int64, float64)')
def polylog(n, z):
    tol = 1.0e-8
    l = 0.0
    k = 1.0
    zk = z
    while True:
        term = zk / k**n
        l += term
        if term < tol:
            return l
        zk *= z
        k += 1.0

@vectorize(nopython=True)
@njit('float64(float64)')
def polylog2(z):
    tol = 1.0e-8
    l = 0.0
    k = 1.0
    zk = z
    while True:
        term = zk / (k*k)
        l += term
        if term < tol:
            return l
        zk *= z
        k += 1.0

@vectorize(nopython=True)
@njit('float64(float64)')
def polylog3(z):
    tol = 1.0e-8
    l = 0.0
    k = 1.0
    zk = z
    while True:
        term = zk / (k*k*k)
        l += term
        if term < tol:
            return l
        zk *= z
        k += 1.0
