# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 00:57:23 2023

@author: Wang-Chen Xue < https://orcid.org/0000-0001-8664-5085 >
"""

import numpy as np
from numba import guvectorize, njit
from numpy.linalg import norm

from numba import prange
from math import lgamma, log
@njit('float64[::1](float64[::1],float64,float64,int64[::1])',
      cache=True, parallel=True)
def knuth_bin_logp(data, left, right, nbins):
    r"""Return the log posterior of histogram bin number using Knuth's rule.

    Parameters
    ----------
    data : ndarray of shape (n,)
        Observed (one-dimensional) data.
    left : float
        Left edge of histogram.
    right : float
        Right edge of histogram.
    nbins : ndarray of shape (m,)
        The bin numbers of histogram to be considered.

    Returns
    -------
    logp : ndarray of shape (m,)
        The log posterior of bin number.

    Notes
    -----
    The log posterior of bins is given by

    .. math::
        F(M|x,I) = n\log(M) + \log\Gamma(\frac{M}{2})
        - M\log\Gamma(\frac{1}{2})
        - \log\Gamma(\frac{2n+M}{2})
        + \sum_{k=1}^M \log\Gamma(n_k + \frac{1}{2})

    where :math:`\Gamma` is the Gamma function, :math:`n` is the number of
    data points, :math:`n_k` is the number of measurements in bin :math:`k`
    [1]_.

    References
    ----------
    .. [1] Knuth, K.H. "Optimal Data-Based Binning for Histograms".
       arXiv:0605197v2, 2013
    """
    data = data[(left <= data) & (data <= right)]
    N = data.size
    logp = np.empty(nbins.size)

    for m in prange(nbins.size):
        M = nbins[m]
        logp_M = N*log(M) + lgamma(M/2) - lgamma(N + M/2) - M*lgamma(0.5)

        n = np.histogram(data, bins=np.linspace(left, right, M + 1))[0]
        for k in range(M):
            logp_M += lgamma(n[k] + 0.5)

        logp[m] = logp_M

    return logp


@njit('float64(int64[::1], int64[::1])', cache=True)
def CCF(u, v):
    u = u - u.mean()
    u /=norm(u)
    v = v - v.mean()
    v /= norm(v)
    return min(max(np.dot(u, v), -1.0), 1.0)


@njit('int64[::1](int64[::1], int64)', cache=True)
def sum_m(arr, m):
    N = arr.size//m
    s = np.empty(N, np.int64)
    n = 0
    for i in range(N):
        n_ = n+m
        s[i] = arr[n:n_].sum()
        n = n_
    return s


@njit('float64[::1](int64[::1], int64[::1], int64)', cache=True, parallel=True)
def MCCF(X, Y, M):
    # assuming the time of y[0] is that of x[0] plus lstart[0], then
    #     len(X) = N*M + (M - 1),
    #     len(Y) = N*M + (M - 1) + (K - 1),
    # where K = nlags, including zero lag, so the last term in len(Y) is K-1,
    # and N is the number of CCF bins
    N = (X.size - M + 1) // M
    K = Y.size - X.size + 1

    XM = []
    YM = []
    for m in range(M):
        XM.append(sum_m(X[m : m + N*M], M))
        YM.append(sum_m(Y[m : m + (Y.size - m)//M*M], M))

    mccf = np.empty(K, dtype=np.float64)
    ccf_k = np.empty(M, dtype=np.float64)
    for k in range(K):
        for m in range(M):
            n = m + k
            ccf_k[m] = CCF(XM[m], YM[n%M][n//M : n//M + N])
        mccf[k] = ccf_k.mean()
    return mccf


@guvectorize(
    ['void(int64[::1], int64[::1], int64, int64[::1])'],
    '(x),(y),()->()',
    cache=True,
    nopython=True,
)
def MCCF_kmax(X, Y, M, Kmax):
    # assuming the time of y[0] is that of x[0] plus lstart[0], then
    #     len(X) = N*M + (M - 1),
    #     len(Y) = N*M + (M - 1) + (K - 1),
    # where K = nlags, including zero lag, so the last term in len(Y) is K-1,
    # and N is the number of CCF bins
    N = (X.size - M + 1) // M
    K = Y.size - X.size + 1

    XM = []
    YM = []
    for m in range(M):
        XM.append(sum_m(X[m : m + N*M], M))
        YM.append(sum_m(Y[m : m + (Y.size - m)//M*M], M))

    mccf_max = -1.0
    kmax = -1
    ccf_k = np.empty(M, dtype=np.float64)
    for k in range(K):
        for m in range(M):
            n = m + k
            ccf_k[m] = CCF(XM[m], YM[n%M][n//M : n//M + N])
        mccf_k = ccf_k.mean()
        if mccf_k > mccf_max:
            mccf_max = mccf_k
            kmax = k

    Kmax[0] = kmax


@njit('int64[::1](int64)', cache=True)
def prime_factors(n):
    i = 2
    res = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            res.append(i)
    if n > 1:
        res.append(n)

    nfactor = len(res)
    factors = np.empty(nfactor, dtype=np.int64)
    for i in range(nfactor):
        factors[i] = res[i]

    return factors


@njit('int64[::1](int64)', cache=True)
def dividable_factors(n):
    i = 1
    res = []
    while i * 2 <= n:
        if not n % i:
            res.append(i)
        i += 1

    nfactor = len(res)
    factors = np.empty(nfactor, dtype=np.int64)
    for i in range(nfactor):
        factors[i] = res[i]

    return factors


def event_lag(x, y, tstart, tstop, lstart, lstop, dt, Dt=None, nboots=1000):
    # 假设
    # 1. x, y 输入为事例数据
    # 2. x, y 时间零点已对齐
    # 3. x, y 的时间精度好于 dt
    # 4. 以 x 为基准求 y 的 lag
    tmp = (tstop - tstart)/dt
    n = round(tmp)
    if abs(tmp - n) > 1e-6:
        raise ValueError('the length of `time_range` must be multiple of `dt`')

    lstart = np.sign(lstart) * np.ceil(np.abs(lstart/dt)) * dt
    lstop = np.sign(lstop) * np.ceil(np.abs(lstop/dt)) * dt
    nlags = round((lstop - lstart)/dt)

    if Dt is None:
        nbins = n//dividable_factors(n)
        logp_xbin = knuth_bin_logp(x, tstart, tstop, nbins)
        logp_ybin = knuth_bin_logp(y, tstart, tstop, nbins)
        M = nbins[(logp_xbin + logp_ybin).argmax()]
        print(f"MCCF: Dt set to {M}*dt={M*dt} according to Knuth's rule.")
    else:
        tmp = Dt/dt
        M = round(tmp)
        if abs(tmp - M) > 1e-6:
            raise ValueError('``Dt`` must be multiple of ``dt``')

        tmp = (tstop - tstart) / Dt
        if abs(tmp - round(tmp)) > 1e-6:
            raise ValueError('length of `time_range` must be multiple of `Dt`')

    xbins = np.linspace(tstart,
                        tstop + (M - 1)*dt,
                        n + M)
    X = np.histogram(x, xbins)[0]

    ybins = np.linspace(tstart + lstart,
                        tstop + (M - 1)*dt + lstop,
                        n + M + nlags)
    Y = np.histogram(y, ybins)[0]

    lags = np.linspace(lstart, lstop, nlags + 1)
    # mccf = MCCF(X, Y, M)
    lag_data = lags[MCCF_kmax(X, Y, M)]
    XB = np.random.poisson(X, size=(nboots, X.size))
    YB = np.random.poisson(Y, size=(nboots, Y.size))
    lag_boots = lags[MCCF_kmax(XB, YB, M)]
    lag_uncert = lag_boots.std(ddof=1)
    return lag_data, lag_uncert, lag_boots


def event_lags(x, y, tstart, tstop, lstart, lstop, dt, Dt=None):
    # 假设
    # 1. x, y 输入为事例数据
    # 2. x, y 时间零点已对齐
    # 3. x, y 的时间精度好于 dt
    # 4. 以 x 为基准求 y 的 lag
    tmp = (tstop - tstart)/dt
    n = round(tmp)
    if abs(tmp - n) > 1e-6:
        raise ValueError('length of `time_range` must be multiple of `dt`')

    if Dt is None:
        nbins = n//dividable_factors(n)
        logp_xbin = knuth_bin_logp(x, tstart, tstop, nbins)
        logp_ybin = knuth_bin_logp(y, tstart, tstop, nbins)
        M = nbins[(logp_xbin + logp_ybin).argmax()]
        print(f"MCCF: Dt set to {M}*dt={M*dt} according to Knuth's rule.")
    else:
        tmp = Dt/dt
        M = round(tmp)
        if abs(tmp - M) > 1e-6:
            raise ValueError('``Dt`` must be multiple of ``dt``')

        tmp = (tstop - tstart) / Dt
        if abs(tmp - round(tmp)) > 1e-6:
            raise ValueError('length of `time_range` must be multiple of `Dt`')

    lstart = np.sign(lstart) * np.ceil(np.abs(lstart/dt)) * dt
    lstop = np.sign(lstop) * np.ceil(np.abs(lstop/dt)) * dt
    nlags = round((lstop - lstart)/dt)

    x_bins = np.linspace(tstart,
                         tstop + (M - 1)*dt,
                         n + M)
    X = np.histogram(x, x_bins)[0]

    y_bins = np.linspace(tstart + lstart,
                         tstop + (M - 1)*dt + lstop,
                         n + M + nlags)
    Y = np.histogram(y, y_bins)[0]

    lags = np.linspace(lstart, lstop, nlags + 1)
    mccf = MCCF(X, Y, M)
    return lags, mccf

#%%
if __name__ ==  '__main__':
    # CCF运行时间 t ~ N=(tstop-tstart)/Dt
    # MCCF运行时间 T ~ M*K*t, M=Dt/dt, K=nlags
    import time
    import matplotlib.pyplot as plt
    import scienceplots
    x = np.random.normal(0,1,10000)
    y = np.random.normal(1,1,10000)
    t0 = time.time()
    tstart = -1.5
    tstop = 2.5
    lstart = -1.5
    lstop = 1.5
    dt = 0.01
    nboots = 200
    lags = []
    boots = []
    Dt_list = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, None]
    # plt.figure()
    for Dt in Dt_list:
        res = event_lag(x, y, tstart, tstop, lstart, lstop, dt, Dt, nboots)
        # plt.plot(*event_lags(x, y, tstart, tstop, Dt, lstart, lstop, dt),label=Dt)
        lags.append(res[:2])
        boots.append(res[-1])
        print(Dt)
    # plt.legend()
    lags = np.array(lags)
    boots = np.array(boots)
#%%
    with plt.style.context(['science', 'nature', 'no-latex']):
        plt.figure(figsize=(4,3), dpi=150)
        vp=plt.violinplot(boots.T, #Dt_list, widths=Dt_list/2,
                          showmedians=0, showextrema=0,
                          quantiles=[[0.15865, 0.5, 0.8413] for i in Dt_list], points=500)
        eb=plt.errorbar(range(1,len(Dt_list)+1), lags[:,0], lags[:,1],
                     fmt='o ', ms=1, c='tab:blue')
        plt.legend([vp['bodies'][0],eb], ['bootstrap distribution', r'Lag$_{\rm obs}$ w/ 1-$\sigma$ error'],
                   frameon=1, loc='upper left')

        plt.axhline(1, c='k', ls=':')
        plt.xlabel('$\Delta t$')
        plt.ylabel('Lag [s]')
        # plt.semilogx()
