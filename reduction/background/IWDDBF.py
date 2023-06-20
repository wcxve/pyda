# -*- coding: utf-8 -*-
"""
@author: xuewc<xuewc@ihep.ac.cn>
"""

import numpy as np

# np.linalg.svd
# np.linalg.pinv

# mathematical detail about rank deficeint:
# https://math.stackexchange.com/a/2866447

# notes on polyfit:
# Note that fitting polynomial coefficients is inherently badly
# conditioned when the degree of the polynomial is large or the
# interval of sample points is badly centered. The quality of the
# fit should always be checked in these cases. When polynomial fits
# are not satisfactory, splines may be a good alternative.

# an example about rank deficeint:
# https://stackoverflow.com/a/49903262

# design matrix effective rank means degree of free pars
# m × n矩阵的秩最大为m和n中的较小者，表示为 min(m,n)。
# 有尽可能大的秩的矩阵被称为有满秩；类似的，否则矩阵是秩不足（或称为“欠秩”）的。

def iwls(X, y, return_cov=True, max_iter=50, omit_zero=False, scale_cov=False, equal_weight=False):
    """
    Parameters
    ----------
    X : (M, N) array_like
        Design matrix.
    y : (M,) array_like
        Ordinate or "dependent variable" values.
    """
    # check the shape of design matrix
    if len(X.shape) == 1:
        X = X[:, None]
    
    ndeg = X.shape[1]
    b = np.full(ndeg, 0.0)
    
    # return if all values of y are zeros
    if all(y == 0.0):
        if return_cov:
            cov = np.full((ndeg, ndeg), 0.0)
            return b, cov
        else:
            return b
    
    delta = 1.0
    niter = 0
    min_delta = ndeg * 1e-6
    while delta > min_delta and niter < max_iter:
        # set up least squares equation with new weight
        if niter:
            m = X @ b
            idx = m > 0.0
            w[idx] = 1.0 / np.sqrt(m[idx])
        else:
            if omit_zero:
                w = np.zeros_like(y, dtype=np.float64)
            else:
                w = np.ones_like(y, dtype=np.float64)
            idx = y > 0.0
            w[idx] = 1.0 / np.sqrt(y[idx])
            if equal_weight:
                w = np.ones_like(y, dtype=np.float64)
        
        WX = w[:, None] * X
        Wy = w * y

        # scale WX to improve condition number and solve
        scale = np.sqrt(np.square(WX).sum(axis=0))
        scale[scale == 0.0] = 1.0
        prev_b = b
        b, resids, rank, s = np.linalg.lstsq(WX / scale, Wy, rcond=None)
        b = b / scale
        
        delta = np.linalg.norm(b - prev_b)
        niter += 1
        #print(niter, b, resids / (X.shape[0] - order), np.all(idx))
        
    if return_cov:
        # scale the covariance to reduce the potential bias of weights
        fac = resids / (X.shape[0] - ndeg) if scale_cov else 1.0
        scaled_cov = fac * np.linalg.inv(WX.T @ WX)
        return b, scaled_cov
    else:
        return b


import matplotlib.pyplot as plt
import ppigrf
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
import sys
sys.path.append('/Users/xuewc/OneDrive/Documents/MyWork/BurstAnalyzer')
from CalcAngle import met_to_utc
if __name__ == '__main__':
    ra = 288.263
    dec = 19.803
    t0 = 55862220.05
    tstart = 1700
    tstop = 2000
    aux_file = '/Users/xuewc/BurstData/GRB221009A/gc_aux_221009_13_v06.fits'
    with fits.open(aux_file) as hdul:
        SME = hdul['SME'].data
        mask = (t0 + tstart <= SME['TIME']) & (SME['TIME'] <= t0 + tstop)
        SME = SME[mask]
        lon, lat = SME['LONLAT'].T
        alt = SME['ALT']
        dates = Time(met_to_utc(SME['TIME'], 'HEBS')).datetime
    # B, incl, decl, Be, Bn, Bu
    b = ppigrf.igrf(lon, lat, alt, dates)
    plt.plot(SME['TIME']-t0, b[:,0]/b[:,0].max())



























