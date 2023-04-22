# -*- coding: utf-8 -*-
"""
@author: xuewc
"""

import numpy as np

from numba import njit
from numpy import log, sqrt

@njit(
    'Tuple((float64, float64[:], float64[:], float64[:]))'
    '(float64[:], float64[:], float64, float64, float64[:])'
)
def wstat(n_on, n_off, t_on, t_off, rate_src):
    n = len(n_on)
    stat = 0.0
    mu_bkg = np.empty(n)
    d1 = np.empty(n)
    d2 = np.empty(n)

    mu_src = rate_src * t_on
    a = t_on / t_off
    v1 = a + 1.0      # a + 1
    v2 = 1.0 + 1.0/a  # 1 + 1/a
    v3 = 2.0 * v1     # 2*(a+1)
    v4 = 4 * a * v1   # 4*a*(a+1)
    v5 = a - 1        # a - 1
    v6 = 2 * a        # 2*a

    for i in range(n):
        on = n_on[i]
        off = n_off[i]
        s = mu_src[i]

        if on == 0.0:
            stat += s + off*log(v1)
            mu_bkg[i] = off / v2 if off > 0.0 else 0.0
            d1[i] = 1.0
            d2[i] = 0.0
        else:
            if off == 0.0:
                if s <= on / v2:
                    stat += -s/a + on*log(v2)
                    mu_bkg[i] = on / v2 - s
                    d1[i] = -1.0/a
                    d2[i] = 0.0
                else:
                    v7 = on/s
                    stat += s + on*(log(v7) - 1.0)
                    mu_bkg[i] = 0.0
                    d1[i] = 1.0 - v7
                    d2[i] = v7/s
            else:
                c = a * (on + off) - v1 * s
                d = sqrt(c*c + v4 * off * s)
                b = (c + d) / v3
                stat += s + v2 * b \
                        - on * (log((s + b)/on) + 1) \
                        - off * (log(b/a/off) + 1)
                mu_bkg[i] = b
                v7 = on + off
                d1[i] = (v5*s - a*v7 + d) / (v6*s)
                d2[i] = (v7 + (v1*(on-off)*s - a*v7*v7)/d) / (2*s*s)

    return stat, mu_bkg/t_on, t_on*d1, t_on*t_on*d2


@njit(
    'Tuple((float64, float64[:], float64[:], float64[:]))'
    '(float64[:], float64[:], float64, float64, float64[:])'
)
def wstat_xspec(obs_counts, back_counts, obs_exp, back_exp, mod_rates):
    stat = 0.0
    n_chans = obs_counts.shape[0]
    back = np.zeros(n_chans)
    diff1 = np.empty(n_chans)
    diff2 = np.empty(n_chans)
    for ch in range(n_chans):
            si = obs_counts[ch]
            bi = back_counts[ch]
            tsi = obs_exp
            tbi = back_exp
            yi = mod_rates[ch]

            ti = tsi + tbi
            yi = max(yi, 1.0e-5/tsi)
            if si == 0.0:
                stat += tsi*yi - bi*log(tbi/ti)
                back[ch] = bi / ti if bi > 0.0 else 0.0
                diff1[ch] = -tsi
                diff2[ch] = 0.0
            else:
                if bi == 0.0:
                    if yi <= si/ti:
                        stat += -tbi*yi - si*log(tsi/ti)
                        back[ch] = si / ti - yi
                        diff1[ch] = tbi
                        diff2[ch] = 0.0
                    else:
                        stat += tsi*yi + si*(log(si)-log(tsi*yi)-1)
                        back[ch] = 0.0
                        diff1[ch] = (si/yi) - tsi
                        diff2[ch] = si/(yi*yi)
                else:
                    a = ti
                    b = ti*yi - si - bi
                    c = -bi*yi
                    d = sqrt(b*b - 4.0*a*c)
                    f = -2*c / (b + d) if b >= 0.0 else -(b - d) / (2*a)
                    g = (ti*yi - si + bi - d)/(2.0*d)
                    h = 2.0*ti*si*bi/(d*d*d)
                    stat += tsi*yi + ti*f - si*log(tsi*yi+tsi*f) \
                            - bi*log(tbi*f) - si*(1-log(si)) \
                            - bi*(1-log(bi))
                    back[ch] = f
                    diff1[ch] = si*(1+g)/(yi+f) + bi*g/f - tsi -g*ti
                    diff2[ch] = -si*h/(yi+f) + si*(1+g)*(1+g)/((yi+f)*(yi+f)) \
                                - bi*h/f + bi*g*g/(f*f) + ti*h
    return stat, back, -diff1, diff2
