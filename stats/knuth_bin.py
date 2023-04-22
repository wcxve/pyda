# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 01:54:51 2023

@author: Wang-Chen Xue < https://orcid.org/0000-0001-8664-5085 >
"""

from math import log, lgamma

import numpy as np
from numba import njit, prange

__all__ = ['knuth_bin_number', 'knuth_bin_logp']


@njit('int64(float64[::1],float64,float64,float64)', cache=True, parallel=True)
def knuth_bin_number(data, left, right, min_width):
    r"""Return the optimal histogram bin number using Knuth's rule.

    Knuth's rule is a fixed-width, Bayesian approach to determining
    the optimal bin number of a histogram.

    Parameters
    ----------
    data : ndarray of shape (n,)
        Observed (one-dimensional) data.
    left : float
        Left edge of histogram.
    right : float
        Right edge of histogram.
    min_width : float
        The minimum width of histogram to be considered.

    Returns
    -------
    nbin : int
        The optimal bin number.

    Notes
    -----
    The optimal number of bins is the value M which maximizes the function

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
    maxM = int((right - left) / min_width)
    logp = np.empty(maxM)

    for M in prange(1, maxM + 1):
        logp_M = N*log(M) + lgamma(M/2) - lgamma(N + M/2) - M*lgamma(0.5)

        n = np.histogram(data, bins=np.linspace(left, right, M + 1))[0]
        for k in range(M):
            logp_M += lgamma(n[k] + 0.5)

        logp[M-1] = logp_M

    nbin = logp.argmax() + 1

    return nbin


@njit('float64[::1](float64[::1],float64,float64,float64)',
      cache=True, parallel=True)
def knuth_bin_logp(data, left, right, min_width):
    r"""Return the log posterior of histogram bin number using Knuth's rule.

    Parameters
    ----------
    data : ndarray of shape (n,)
        Observed (one-dimensional) data.
    left : float
        Left edge of histogram.
    right : float
        Right edge of histogram.
    min_width : float
        The minimum width of histogram to be considered.

    Returns
    -------
    logp : ndarray of shape (M,)
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
    maxM = int((right - left) / min_width)
    logp = np.empty(maxM)

    for M in prange(1, maxM + 1):
        logp_M = N*log(M) + lgamma(M/2) - lgamma(N + M/2) - M*lgamma(0.5)

        n = np.histogram(data, bins=np.linspace(left, right, M + 1))[0]
        for k in range(M):
            logp_M += lgamma(n[k] + 0.5)

        logp[M-1] = logp_M

    return logp


if __name__ == '__main__':
    nbin = knuth_bin_number(np.random.randn(10000), -5, 5, 0.01)
    import matplotlib.pyplot as plt
    plt.plot(knuth_bin_logp(np.random.randn(10000), -5, 5, 0.01))
    plt.semilogy()
