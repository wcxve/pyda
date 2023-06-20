# -*- coding: utf-8 -*-
"""
@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import numpy as np

from numba import njit


def calc_duration(tstart, tstop, counts, exposure, stime, order, f, nboot=100):
    """
    Calculate signal duration.

    Parameters
    ----------
    tstart : array_like
        Left time bin of signal.
    tstop : array_like
        Right time bin of signal.
    counts : array_like
        Signal counts.
    exposure : array_like
        Exposure of time bins.
    stime : array_like
        Signal region.
    order : int
        Polynomial order used in background estimation.
    f : float
        Quantile of signal, 0 < f <= 1.
    nboot : int
        Number of bootstrap.

    Returns
    -------
    duration : float
        Duration of signal.
    uncertainty : float
        Uncertainty of duration.

    """
    tstart = np.asarray(tstart, dtype=np.float64)
    tstop = np.asarray(tstop, dtype=np.float64)
    tmid = (tstart + tstop) / 2
    exposure = np.asarray(exposure, dtype=np.float64)

    counts = np.asarray(counts, dtype=np.float64)
    stime = np.asarray(stime, dtype=np.float64)
    order = int(order)

    smask = (stime[0] <= tstart) & (tstop <= stime[1])

    boot = np.random.poisson(counts, size=(nboot, counts.size))
    data = np.row_stack((counts, boot))
    src = np.ascontiguousarray(data[:, smask])
    bkg = np.ascontiguousarray(data[:, ~smask])

    if order == 0:
        bkg = bkg.sum(1)
        bkg_exposure = exposure[~smask].sum()
        src_exposure = exposure[smask]
        duration = np.empty(nboot + 1)
        for i in range(nboot+1):
            idx1, idx2 = _eval0(f, src[i], src_exposure, bkg[i], bkg_exposure)
            duration[i] = tmid[idx2] - tmid[idx1]
    else:
        factor = exposure / (tstart - tstop)
        basis = np.array([
            (tstop**(i + 1.0) - tstart**(i + 1.0)) * factor
            for i in range(order + 1)
        ]).T
        basis_src = np.ascontiguousarray(basis[smask, :])
        basis_bkg = np.ascontiguousarray(basis[~smask, :])

        duration = np.empty(nboot + 1)
        for i in range(nboot+1):
            idx1, idx2 = _eval(f, src[i], bkg[i], basis_src, basis_bkg)
            duration[i] = tmid[idx2] - tmid[idx1]

    return duration[0], duration[1:].std(ddof=1)#, duration.mean(), np.quantile(duration, 0.5)


@njit('float64[::1](float64[::1])')
def _get_weight(y):
    w = np.empty(y.size)
    for i in range(y.size):
        yi = y[i]
        if yi > 0.0:
            w[i] = 1.0 / np.sqrt(yi)
        else:
            w[i] = 0.0

    return w


@njit('float64[::1](float64[:,::1], float64[::1], float64[::1])')
def _wlstsq(X, y, w):
    if np.all(y == 0.0):
        return np.full(X.shape[1], 0.0)

    # set up least squares equation with weight
    WX = np.expand_dims(w, 1) * X
    Wy = w * y

    # scale WX to improve condition number and solve
    scale = np.sqrt(np.sum(WX*WX, axis=0))
    scale[scale == 0.0] = 1.0
    b = np.linalg.lstsq(WX / scale, Wy, rcond=-1)[0]
    return b / scale


@njit('int64[:](float64,float64[::1],float64[::1],float64,float64)')
def _eval0(f, src, src_exposure, bkg, bkg_exposure):
    net = src - src_exposure * (bkg / bkg_exposure)
    cumsum_net = np.cumsum(net)
    f1 = (1.0 - f) / 2.0
    f2 = 1.0 - f1
    L1 = cumsum_net[-1]*f1
    L2 = cumsum_net[-1]*f2
    idx = np.empty(2, dtype=np.int64)
    idx0 = np.argwhere(cumsum_net < L1)
    idx[0] = idx0[-1,0] + 1 if idx0.size > 0 else 0
    idx[1] = np.argwhere(cumsum_net >= L2)[0,0]
    return idx


@njit('int64[:](float64,float64[::1],float64[::1],float64[:,::1],float64[:,::1])')
def _eval(f, src, bkg, basis_src, basis_bkg):
    w = _get_weight(bkg)
    b = _wlstsq(basis_bkg, bkg, w)
    w = _get_weight(basis_bkg @ b)
    b = _wlstsq(basis_bkg, bkg, w)
    net = src - basis_src @ b
    cumsum_net = np.cumsum(net)
    f1 = (1.0 - f) / 2.0
    f2 = 1.0 - f1
    L1 = cumsum_net[-1]*f1
    L2 = cumsum_net[-1]*f2
    idx = np.empty(2, dtype=np.int64)
    idx0 = np.argwhere(cumsum_net < L1)
    idx[0] = idx0[-1,0] + 1 if idx0.size > 0 else 0
    idx[1] = np.argwhere(cumsum_net >= L2)[0,0]
    return idx
