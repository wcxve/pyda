# -*- coding: utf-8 -*-
"""
@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import sys

import numpy as np

from numba import njit
from tqdm import trange

__all__ = ['calc_duration']


def calc_duration(
    counts, tstart, tstop, exposure, srange, brange, order, q, nboot=100
):
    """
    Calculate signal duration.

    Parameters
    ----------
    counts : array_like of shape (t,) or (n, t)
        Counts of light curves.
    tstart : array_like of shape (t,)
        Left time bin of light curves.
    tstop : array_like of shape (t,)
        Right time bin of light curves.
    exposure : array_like of shape (t,) or (n, t)
        Exposure of time bins.
    srange : array_like of shape (2,)
        Time range of signal.
    brange : array_like of shape (2,) or (b, 2)
        Time range of background.
    order : int
        Polynomial order used in background estimation.
    q : float
        Quantile of signal, 0 < q <= 1.
    nboot : int, optional
        Number of bootstrap. The default is 100.

    Returns
    -------
    duration : float
        Duration of q-quantile.
    uncertainty : float
        Uncertainty of duration.
    duration_boots : array of shape (nboot,)
        Bootstrap samples of duration.
    trange_of_q : array of shape (2,)
        Time range of q-quantile.
    trange_of_q_boots : array of shape (nboot, 2)
        Bootstrap samples of `trange_of_q`.

    """
    c_array = np.ascontiguousarray

    counts = np.atleast_2d(np.array(counts, dtype=np.float64))
    tstart = np.array(tstart, dtype=np.float64)
    tstop = np.array(tstop, dtype=np.float64)
    exposure = np.atleast_2d(np.array(exposure, dtype=np.float64))
    factor = exposure / (tstop - tstart)
    srange = np.array(srange, dtype=np.float64)
    brange = np.atleast_2d(np.array(brange, dtype=np.float64))

    smask = (srange[0] <= tstart) & (tstop <= srange[1])
    scounts = c_array(counts[:, smask])  # shape=(n, ts)
    ststart = tstart[smask]
    ststop = tstop[smask]
    sfactor = c_array(factor[:, smask])  # shape=(n, ts)
    sbasis = [  # shape=(order + 1, ts), no exposure correction
        ststop ** (i + 1.0) - ststart ** (i + 1.0)
        for i in range(order + 1)
    ]
    sbasis = c_array(np.transpose(sbasis)) # shape=(ts, order + 1)

    bmask = (brange[:, :1] <= tstart) & (tstop <= brange[:, 1:])
    bmask = np.any(bmask, axis=0)
    bcounts = c_array(counts[:, bmask])  # shape=(n, tb)
    btstart = tstart[bmask]
    btstop = tstop[bmask]
    bfactor = factor[:, bmask]  # shape=(n, tb)
    bbasis = [  # shape=(order + 1, tb)
        btstop ** (i + 1.0) - btstart ** (i + 1.0)
        for i in range(order + 1)
    ]
    bbasis = np.array(bbasis)
    # apply exposure correction when fit background, shape=(n, order + 1, tb)
    bbasis = bbasis[None, :, :] * bfactor[:, None, :]
    bbasis = np.transpose(bbasis, axes=(0, 2, 1))  # shape=(n, tb, order + 1)
    bbasis = c_array(bbasis)

    trange_of_q = _get_trange_of_q(q, ststart, ststop, scounts, bcounts,
                                   sfactor, sbasis, bbasis)
    duration = trange_of_q[1] - trange_of_q[0]

    rng = np.random.default_rng(seed=42)
    ssim = rng.poisson(scounts, size=(nboot, *scounts.shape))
    ssim = np.array(ssim, dtype=np.float64)
    bsim = rng.poisson(bcounts, size=(nboot, *bcounts.shape))
    bsim = np.array(bsim, dtype=np.float64)

    trange_of_q_boot = np.empty((nboot, 2))
    for i in trange(nboot, desc='Duration Bootstrap', file=sys.stdout):
        trange_of_q_boot[i] = _get_trange_of_q(
            q, ststart, ststop, ssim[i], bsim[i], sfactor, sbasis, bbasis
        )

    duration_boot = trange_of_q_boot[:, 1] - trange_of_q_boot[:, 0]
    uncertainty = np.std(duration_boot, ddof=1)

    return duration, uncertainty, duration_boot, trange_of_q, trange_of_q_boot


@njit('float64[::1](float64[::1])', cache=True)
def _get_weight(y):
    w = np.empty(y.size)
    for i in range(y.size):
        yi = y[i]
        if yi > 0.0:
            w[i] = 1.0 / np.sqrt(yi)
        else:
            w[i] = 0.0

    return w


@njit('float64[::1](float64[:,::1], float64[::1], float64[::1])', cache=True)
def _wlstsq(X, y, w):
    if np.all(y == 0.0):
        return np.full(X.shape[1], 0.0)

    # set up least squares equation with weight
    WX = np.expand_dims(w, axis=1) * X
    Wy = w * y

    # scale WX to improve condition number and solve
    scale = np.sqrt(np.sum(WX * WX, axis=0))
    scale[scale == 0.0] = 1.0
    b = np.linalg.lstsq(WX / scale, Wy, rcond=-1)[0]
    return b / scale


@njit('float64[::1](float64[::1], float64[:,::1], float64[:,::1])', cache=True)
def _estimate_back(bcounts, bbasis, sbasis):
    """Fit the background via 2-pass procedure, then interpolate."""
    w0 = _get_weight(bcounts)
    b0 = _wlstsq(bbasis, bcounts, w0)
    w1 = _get_weight(bbasis @ b0)
    b1 = _wlstsq(bbasis, bcounts, w1)
    return sbasis @ b1


@njit(
    'float64[::1](float64, float64[::1], float64[::1], float64[:,::1], '
    'float64[:,::1], float64[:,::1], float64[:,::1], float64[:,:,::1])',
    cache=True
)
def _get_trange_of_q(
    q, ststart, ststop, scounts, bcounts, sfactor, sbasis, bbasis
):
    signal = np.zeros(scounts.shape[1])

    for i in range(scounts.shape[0]):
        back_i = _estimate_back(bcounts[i], bbasis[i], sbasis)
        signal += scounts[i] / sfactor[i] - back_i

    signal_cumsum = signal.cumsum()
    q1 = (1.0 - q) / 2.0
    q2 = 1.0 - q1
    L1 = q1 * signal_cumsum[-1]
    L2 = q2 * signal_cumsum[-1]

    trange_of_q = np.empty(2)

    idx_max = len(ststart) - 1
    indices = np.flatnonzero(signal_cumsum < L1)
    if indices.size > 0:
        idx = min(indices[-1] + 1, idx_max)
        trange_of_q[0] = ststop[idx]
    else:  # TODO: how can this be possible?
        trange_of_q[0] = ststart[0]

    indices = np.flatnonzero(signal_cumsum >= L2)
    if indices.size > 0:
        trange_of_q[1] = ststop[indices[0]]
    else:  # TODO: how can this be possible?
        trange_of_q[1] = ststop[-1]

    return trange_of_q
