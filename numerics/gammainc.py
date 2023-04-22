# -*- coding: utf-8 -*-
"""
@author: xuewc
"""

from math import lgamma

import numpy as np
from numba import njit
from spycial.digamma import _digamma as digamma
from polygamma import polygamma_fn

@njit
def _s(P, X, XLOG, GP1LOG, PSIP1, PSIDP1, E, NMAX):
    F = np.exp(P * XLOG - GP1LOG - X)
    DFP = F * (XLOG - PSIP1)
    # DFPP = DFP*DFP/F - F*PSIDP1

    C = 1.0
    S = 1.0
    CP = 0.0
    # CPP = 0.0
    DSP = 0.0
    # DSPP = 0.0
    A = P

    for i in range(NMAX):
        A += 1.0
        CPC = CP / C
        CP = CPC - 1.0/A
        # CPP = CPP/C - CPC*CPC + 1.0/A/A
        C *= X/A
        CP *= C
        # CPP = CPP*C + CP*CP/C
        S += C
        DSP += CP
        # DSPP += CPP

        if C <= E*S:
            D3 = S*DFP + F*DSP
            # D4 = S*DFPP + 2.0*DFP*DSP + F*DSPP
            D6 = S*F
            break

    return D3, D6

@njit
def _cf(P, X, XLOG, GPLOG, PSIP, PSIDP, PM1, PN, DP, E, NMAX):
    F = np.exp(P*XLOG - GPLOG - X)
    DFP = F * (XLOG - PSIP)
    # DFPP = DFP*DFP/F - F*PSIDP

    A = PM1
    B = X + 1.0 - A
    TERM = 0
    PN[0] = 1.0
    PN[1] = X
    PN[2] = X + 1.0
    PN[3] = X * B
    S0 = PN[2] / PN[3]
    for I in range(4):
        DP[I] = 0.0
        # DPP[I] = 0.0
    DP[3] = -X

    for i in range(NMAX):
        A -= 1.0
        B += 2.0
        TERM += 1
        AN = A*TERM
        PN[4] = B*PN[2] + AN*PN[0]
        PN[5] = B*PN[3] + AN*PN[1]
        DP[4] = B*DP[2] - PN[2] + AN*DP[0] + PN[0]*TERM
        DP[5] = B*DP[3] - PN[3] + AN*DP[1] + PN[1]*TERM
        # DPP[4] = B*DPP[2] + AN*DPP[0] + 2.0*(TERM*DP[0] - DP[2])
        # DPP[5] = B*DPP[3] + AN*DPP[1] + 2.0*(TERM*DP[1] - DP[3])

        if abs(PN[5]) > 1.0e-30:
            S = PN[4] / PN[5]
            C = abs(S - S0)
            if C*P <= E and C >= E*S:
                break
            S0 = S

        for I in range(4):
            I2 = I + 2
            DP[I] = DP[I2]
            # DPP[I] = DPP[I2]
            PN[I] = PN[I2]

        if abs(PN[4]) > 1.0e30:
            for I in range(4):
            	DP[I] = DP[I] / 1.0e30
            	# DPP[I] = DPP[I] / 1.0e30
            	PN[I] = PN[I] / 1.0e30

    DSP = (DP[4] - S*DP[5]) / PN[5]
    # DSPP = (DPP[4] - S*DPP[5] - 2.0*DSP*DP[5]) / PN[5]
    D3 = -F*DSP - S*DFP
    # D4 = -F*DSPP - 2.0*DSP*DSP - S*DFPP
    D6 = 1.0 - F*S

    return D3, D6

@njit('Tuple((float64[:],float64[:]))(float64, float64[:])')
def dligamma(P, X):
    E=1e-5
    NMAX=100

    PN = np.empty(6)
    DP = np.empty(6)
    # DPP = np.empty(6)

    GPLOG = lgamma(abs(P))
    GP1LOG = lgamma(abs(P + 1.0))
    PSIP = digamma(abs(P))
    PSIP1 = digamma(abs(P + 1.0))
    PSIDP = polygamma_fn(1, P)
    PSIDP1 = polygamma_fn(1, P + 1.0)
    print(GPLOG,GP1LOG,PSIP,PSIP1,PSIDP,PSIDP1)

    PM1 = P - 1.0
    XLOG = np.log(X)
    # D1 = np.exp(-GPLOG + PM1*XLOG - X)
    # D2 = D1 * (PM1/X - 1.0)
    D3 = np.zeros(len(X))
    # D4 = np.zeros(len(X))
    # D5 = D1 * (XLOG - PSIP)
    D6 = np.zeros(len(X))

    mask_s = (P > X) | (X <= 1.0)

    for i in range(len(X)):
        if mask_s[i]:
            D3[i], D6[i] = _s(P, X[i], XLOG[i], GP1LOG, PSIP1, PSIDP1, E, NMAX)
        else:
            D3[i], D6[i] = _cf(P, X[i], XLOG[i], GPLOG, PSIP, PSIDP, PM1, PN,
                               DP, E, NMAX)
    intergal = D6 * np.exp(GPLOG)
    deriv = intergal * (PSIP1 + D3/D6)
    return intergal, deriv

if __name__ == '__main__':
    P = -1.5#np.random.uniform(-5, 5)
    X = [1,2.]#np.random.uniform(0, 1000)
    print(P, X, sep=',')
    print(np.diff(dligamma(P, np.atleast_1d(X)), axis=-1))
