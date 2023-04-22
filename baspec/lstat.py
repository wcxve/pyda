# -*- coding: utf-8 -*-
"""
@author: xuewc
"""

import numpy as np

from numba import njit
from numpy import exp, log
from scipy.special import gammaln

_NCOEFF = 6
_COEFF = np.array([
    76.18009172947146, -86.50532032941677,  24.01409824083091 ,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5
])
_STEP = 2.5066282746310005
_NFACT = 1001
_LOGFACT = gammaln(np.arange(_NFACT) + 1) # ln(0!) to ln((_NFACT-1)!)


@njit('float64(int64)')
def lfact(n):
    if n >= 0 and n < _NFACT:
        return _LOGFACT[n]
    else:
        dsum = 1.000000000190015
        xiter = n + 1.0
        for i in range(_NCOEFF):
            xiter += 1.0
            dsum += _COEFF[i]/xiter
        return (n + 1.5)*log(n + 6.5) - (n + 6.5) + log(_STEP*dsum/(n + 1.0))

@njit('float64(int64, int64, int64)')
def lterm(n1, n2, n3):
    return lfact(n1+n2-n3) - lfact(n1-n3) - lfact(n3)


@njit('float64(int64, float64, int64)')
def lcalc(j, dlogS, flag):
    if flag == 0:
        return j*dlogS
    elif flag == 1:
        return log(j) + (j-1)*dlogS
    else:
        return log(j*(j-1)) + (j-2)*dlogS


@njit('float64(float64, int64, int64, int64)')
def lsum(S, n_on, n_off, flag):
    # The lsum routine returns the log of the summation from 0 to nobs.
    # Calculate the log of the summation over 0 to C of
    #             S^j (C+B-j)!/j!/(C-j)!             if flag = 0
    #             j S^(j-1) (C+B-j)!/j!/(C-j)!       if flag = 1
    #             j(j-1) S^(j-2) (C+B-j)!/j!/(C-j)!  if flag = 2
    # Avoids numerical overflows and speeds up calculation by first finding
    # the largest term, dividing it out of the sum, and then including only
    # those terms that are > exp(-20) times the largest term.
    # Arguments :
    #     S      d       i: the model rate times the sum of the exposure times
    #     C      i       i: the counts in the source observation
    #     B      i       i: the counts in the background observation
    #     Flag   i       i: indicates which of 3 summations is to be performed

    result = 0.0

    # First trap out the special case of C < flag. In this case the summation
    # is zero so we return -250 for the log. Also trap case of flag having an
    # unexpected value.
    if n_on < flag:
        return -250.0

    # Now handle the easy case of S and C > 0
    j = 0
    jmax = 0
    dtmax = 0.0
    if S > 0 and n_on > 0:
        dlogS = log(S)

        # Find the maximum term in the summation from 0 to C
        if n_on > flag:
            j1 = flag
            j2 = n_on
            dt1 = lcalc(j1, dlogS, flag) + lterm(n_on, n_off, j1)
            dt2 = lcalc(j2, dlogS, flag) + lterm(n_on, n_off, j2)
            is_done = False
            while not is_done:
                j = (j1+j2)//2
                dtry1 = lcalc(j, dlogS, flag) + lterm(n_on, n_off, j)
                dtry2 = lcalc(j+1, dlogS, flag) + lterm(n_on, n_off, j+1)

                if dtry2 > dtry1:
                    j1 = j+1
                    dt1 = dtry2
                else:
                    j2 = j
                    dt2 = dtry1
                is_done = (j1+1 == j2 or j1 == j2)

            if dt2 > dt1:
                jmax = j2
                dtmax = dt2
            else:
                 jmax = j1
                 dtmax = dt1

        elif n_on == flag:
            jmax = flag
            dtmax = lcalc(jmax, dlogS, flag) + lterm(n_on, n_off, jmax)

        # Now sum over all the terms bigger than EXP(-20) times the maximum.
        # Sum up from the maximum then down from the maximum using the fact
        # that the terms are decrease monotonically on both sides of the maximum.
        j = jmax - 1
        dtemp = 0.0
        while (j < n_on and dtemp > -20.0):
            j += 1
            dtemp = lcalc(j, dlogS, flag) + lterm(n_on, n_off, j) - dtmax
            result += exp(dtemp)

        j = jmax
        dtemp = 0.0
        while (j > flag and dtemp > -20.0):
            j -= 1
            dtemp = lcalc(j, dlogS, flag) + lterm(n_on, n_off, j) - dtmax
            result += exp(dtemp)

        result = log(result) + dtmax

    elif S == 0 and n_on > 0:
        # Do the special cases for S = 0. If S = 0 then only the j=FLAG term
        # contributes
        result = lterm(n_on, n_off, flag)

    elif n_on == 0:
        # Do the special cases for C = 0. If C = 0 then only the j=0 term
        # contributes.
        result = lterm(0, n_off, 0)

    return result


@njit(
    'Tuple((float64, float64[:], float64[:]))'
    '(int64[:], int64[:], float64, float64, float64[:])'
)
def lstat(n_on, n_off, t_on, t_off, rate_src):
    n = len(n_on)
    stat = 0.0
    d1 = np.empty(n)
    d2 = np.empty(n)

    for i in range(n):
        on = n_on[i]
        off = n_off[i]
        s = rate_src[i]

        t = t_on + t_off
        sat = on/t_on - off/t_off
        sat = max(0.0, sat)

        v1 = lsum(s*t, on, off, flag=0)
        v2 = lsum(sat*t, on, off, flag=0) - v1
        stat += (s - sat)*t_on + v2

        log_t = log(t)
        v3 = exp(lsum(s*t, on, off, flag=1) + log_t - v1)
        v4 = exp(lsum(s*t, on, off, flag=2) + 2.0*log_t - v1)

        d1[i] = v3 - t_on
        d2[i] = v3*v3 - v4

    return stat, -d1, d2
