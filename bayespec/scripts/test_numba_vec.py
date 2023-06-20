# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 19:46:55 2023

@author: Wang-Chen Xue < https://orcid.org/0000-0001-8664-5085 >
"""

import numpy as np
from numba import vectorize, guvectorize, prange, njit

if __name__ == '__main__':
    @guvectorize(
        ['void(float64,float64,float64,float64[:],float64[:])'],
        '(),(),(),(n)->(n)',
        nopython=True,
        target='parallel',
        cache=True
    )
    def pl(index, norm, const, freqs, powers):
        for i in prange(len(freqs)):
            powers[i] = norm * freqs[i]**index + const

    @guvectorize(
        ['void(float64,float64,float64,float64,float64,float64[:],float64[:])'],
        '(),(),(),(),(),(n)->(n)',
        nopython=True,
        target='parallel'
    )
    def bknpl(gamma1, v_break, gamma2, norm, const, freqs, powers):
        ngamma1 = -gamma1
        ngamma2 = -gamma2
        tmp = v_break ** (gamma2 - gamma1)
        for i in prange(len(freqs)):
            freq = freqs[i]
            if freq <= v_break:
                powers[i] = norm * freq**ngamma1 + const
            else:
                powers[i] = norm * tmp * freq**ngamma2 + const

    # @guvectorize(
    #     ['void(float64[:], float64[:], float64[:])'],
    #     '(n),(n)->()',
    #     nopython=True,
    # )
    # def ln_likelihood(model, data, res):
    #     r = 0.0
    #     for i in range(len(model)):
    #         mi = model[i]
    #         di = data[i]
    #         r -= np.log(mi) + di/mi
    #     res[0] = r

    @guvectorize(
        ['void(float64[:], float64[:], float64[:])'],
        '(n),(n)->()',
        nopython=True,
        target='parallel'
    )
    def ln_likelihood(model, data, res):
        r = 0.0
        for i in prange(len(model)):
            mi = model[i]
            di = data[i]
            r -= np.log(mi) + di/mi
        res[0] = r


    def lnprob(pars, bounds, freqs, data):
        res = np.empty(len(pars))
        mask = True
        for i in range(len(bounds)):
            mask &= (bounds[i, 0] <= pars[:, i]) & (pars[:, i] <= bounds[i, 1])
        res[~mask] =  -np.inf
        res[mask] = ln_likelihood(pars[mask], freqs, data)
