# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 23:59:31 2023

@author: xuewc
"""

from sys import stdout

import matplotlib.pyplot as plt
import numpy as np
import numexpr as ne
from tqdm import trange

__all__ = ['blocks_tte', 'plot_blocks']


def plot_blocks(t, edges, dt, unit='s'):
    tmin = edges.min()
    tmax = edges.max()
    nbins = round((tmax - tmin)/dt)
    tbins = np.linspace(tmin, tmax, nbins + 1)
    tmid = (tbins[:-1] + tbins[1:])/2.0
    rate_cc = np.histogram(t, tbins)[0]/np.diff(tbins)
    rate_bb = np.histogram(t, edges)[0]/np.diff(edges)
    plt.figure()
    plt.step(tbins, np.append(rate_cc, rate_cc[-1]), c='k', where='post')
    plt.errorbar(tmid, rate_cc, np.sqrt(rate_cc/np.diff(tbins)), fmt='k ', alpha=0.5)
    plt.hist(edges[:-1], edges, weights=rate_bb,
             histtype='barstacked', fill=True, edgecolor='tab:blue', zorder=0, alpha=0.5, lw=0.5, ls=':')
    plt.xlabel('time [%s]' % unit)
    plt.ylabel('Rate [%s$^{-1}$]' % unit)
    plt.xlim(tmin, tmax)


def block_binned(tbins, counts, p0=0.05, niter=0):
    if len(counts) != len(tbins) - 1:
        raise ValueError('``counts`` must be matched to ``tbins``')

    exposure = np.diff(tbins)

    if np.any(exposure <= 0.0):
        raise ValueError('``tbins`` must be ordered')

    ndata = len(counts)
    N = ndata

    N_cumsum = np.hstack((0, counts)).cumsum()
    N_remainders = N_cumsum[-1] - N_cumsum

    edges = tbins
    T_remainders = edges[-1] - edges

    print(T_remainders, N_remainders)

    # Codes below is copied from blocks tte
    len8_rjust = lambda s: str(s).rjust(2).rjust(5).ljust(8)
    len8_ljust = lambda s: str(s).ljust(2).rjust(5).ljust(8)

    tstr = _estimate_run_time(ndata)
    print(f'\nBayesian Blocks: about {tstr} to go')

    ncp_prior = 4 - np.log(73.53 * p0 * (N ** -0.478))

    cp0 = _loop_events(N_remainders, T_remainders, ncp_prior, '  EVT ')
    ncp0 = cp0.size
    fpr0 = 1 - (1 - p0) ** ncp0
    print(f'  NCP : {ncp0}\n  FPR : {fpr0:.2e}\n')

    # -----------------------------------------------------------------
    # Iterate if desired
    # -----------------------------------------------------------------
    if niter > 0:
        print(
            f'Bayesian Blocks: iteration starts, each step takes about {tstr}'
        )

        ncp_hist = [ncp0]
        cp_hist = [cp0]
        fpr_hist = [fpr0]

        nit = 0
        converge = False
        while nit < niter:
            nit += 1

            fpr_single = 1 - (1 - p0) ** (1 / ncp_hist[-1])
            ncp_prior = 4 - np.log(73.53 * fpr_single * (N ** -0.478))

            cp = _loop_events(
                N_remainders, T_remainders, ncp_prior, f'{nit} '.rjust(6)
            )
            cp_hist.append(cp)

            ncp = cp.size
            ncp_hist.append(ncp)

            fpr = 1 - (1 - fpr_single) ** ncp
            fpr_hist.append(fpr)

            print(
                f'  NCP : {len8_rjust(ncp_hist[-2])} -> {len8_ljust(ncp)}\n'
                f'  FPR : {fpr_hist[-2]:.2e} -> {fpr:.2e}\n'
            )

            if ncp == ncp_hist[-2]:
                converge = 1
                break

            if ncp in ncp_hist[:-2]:
                converge = 2

                fpr_hist = np.array(fpr_hist)

                mask = fpr_hist <= p0
                if any(mask):
                    idx = fpr_hist[mask].argmax()
                    idx = np.where(mask)[0][idx]
                else:
                    idx = fpr_hist.argmin()

                fpr = fpr_hist[idx]
                cp = cp_hist[idx]
                ncp = ncp_hist[idx]

                break

        cstr = 'converged' if converge else 'not converged'
        print(
            f'Bayesian Blocks: iteration {cstr} within {nit} step(s)\n'
            f'  NCP : {len8_rjust(ncp_hist[0])} -> {len8_ljust(ncp)}\n'
            f'  FPR : {fpr_hist[0]:.2e} -> {fpr:.2e}\n'
        )

        edges = (edges[cp0], edges[cp])

    else:
        edges = edges[cp0]

    return edges


def blocks_tte(t, p0=0.05, niter=0, return_mvts=False):
    """
    Bayesian blocks analysis for time tagged events.

    Parameters
    ----------
    t : array
        Arrival time of events.
    p0 : float, optional
        False positive rate of an individual change point. The default is 0.05.
    niter : int, optional
        Iterate ``niter`` times such that the overall false positive rate
        approaches ``p0``. The default is 0.
    return_mvts : bool, optional
        Whether to return minimum variability time scale. The default is False.

    Returns
    -------
    edges
        The edges of Bayesian blocks.

    """
    if p0 <= 0.0 or p0 >= 1.0:
        raise ValueError('``p0`` must be in (0,1)')

    if niter < 0 or type(niter) is not int:
        raise ValueError('``niter`` must be non-negative integer')

    N_remainders, T_remainders, edges, N, ndata = _get_data(t)

    len8_rjust = lambda s: str(s).rjust(2).rjust(5).ljust(8)
    len8_ljust = lambda s: str(s).ljust(2).rjust(5).ljust(8)

    tstr = _estimate_run_time(ndata)
    print(f'\nBayesian Blocks: about {tstr} to go')

    ncp_prior = 4 - np.log(73.53 * p0 * (N ** -0.478))

    cp0 = _loop_events(N_remainders, T_remainders, ncp_prior, '  EVT ')
    ncp0 = cp0.size
    fpr0 = 1 - (1 - p0) ** ncp0
    print(f'  NCP : {ncp0}\n  FPR : {fpr0:.2e}\n')

    # -----------------------------------------------------------------
    # Iterate if desired
    # -----------------------------------------------------------------
    if niter > 0:
        print(
            f'Bayesian Blocks: iteration starts, each step takes about {tstr}'
        )

        ncp_hist = [ncp0]
        cp_hist = [cp0]
        fpr_hist = [fpr0]

        nit = 0
        converge = False
        while nit < niter:
            nit += 1

            fpr_single = 1 - (1 - p0) ** (1 / ncp_hist[-1])
            ncp_prior = 4 - np.log(73.53 * fpr_single * (N ** -0.478))

            cp = _loop_events(
                N_remainders, T_remainders, ncp_prior, f'{nit} '.rjust(6)
            )
            cp_hist.append(cp)

            ncp = cp.size
            ncp_hist.append(ncp)

            fpr = 1 - (1 - fpr_single) ** ncp
            fpr_hist.append(fpr)

            print(
                f'  NCP : {len8_rjust(ncp_hist[-2])} -> {len8_ljust(ncp)}\n'
                f'  FPR : {fpr_hist[-2]:.2e} -> {fpr:.2e}\n'
            )

            if ncp == ncp_hist[-2]:
                converge = 1
                break

            if ncp in ncp_hist[:-2]:
                converge = 2

                fpr_hist = np.array(fpr_hist)

                mask = fpr_hist <= p0
                if any(mask):
                    idx = fpr_hist[mask].argmax()
                    idx = np.where(mask)[0][idx]
                else:
                    idx = fpr_hist.argmin()

                fpr = fpr_hist[idx]
                cp = cp_hist[idx]
                ncp = ncp_hist[idx]

                break

        cstr = 'converged' if converge else 'not converged'
        print(
            f'Bayesian Blocks: iteration {cstr} within {nit} step(s)\n'
            f'  NCP : {len8_rjust(ncp_hist[0])} -> {len8_ljust(ncp)}\n'
            f'  FPR : {fpr_hist[0]:.2e} -> {fpr:.2e}\n'
        )

        edges = (edges[cp0], edges[cp])

    else:
        edges = edges[cp0]

    return edges


def _get_data(t):
    if np.any(np.diff(t) < 0.0):
        raise ValueError('``t`` must be ordered')

    unq, counts = np.unique(t, return_counts=True)

    N = t.size
    ndata = unq.size

    N_cumsum = np.hstack((0, counts)).cumsum()
    N_remainders = N_cumsum[-1] - N_cumsum

    edges = np.hstack((unq[0], (unq[1:] + unq[:-1])/2.0, unq[-1]))
    T_remainders = edges[-1] - edges

    return N_remainders, T_remainders, edges, N, ndata


def _estimate_run_time(ndata):
    # run time estimated based on 8 performance-cores of M1 Max and vecLib
    s = 300 * (ndata/660000) ** 2
    h = int(s/3600)
    m = int((s - h*3600) / 60)
    s = round(s - h*3600 - m*60)
    tstr = ''
    if h:
        tstr += f'{h} hr '
    if m or h:
        tstr += f'{m} min '
    tstr += f'{s} sec'

    return tstr


# -----------------------------------------------------------------
# Speed tricks: resolve once for fitness function used in the loop
# -----------------------------------------------------------------
# N - n: number of elements in each block
# T - t: width or duration of each block
_fitness = ne.NumExpr(
    ex='(N - n) * log((N - n) / (T - t)) + best - ncp_prior',
    signature=(
        ('N', np.float64),
        ('n', np.float64),
        ('T', np.float64),
        ('t', np.float64),
        ('ncp_prior', np.float64),
        ('best', np.float64),
    ),
    optimization='aggressive',
    truediv=True
)


def _loop_events(N_r, T_r, ncp_prior, desc):
    ne.set_num_threads(ne._init_num_threads())

    T_r = np.asarray(T_r, dtype=np.float64)
    N_r = np.asarray(N_r, dtype=np.float64)
    N = len(N_r) - 1

    # arrays to store the best configuration
    tmp = np.empty(N, dtype=np.float64)
    best = np.zeros(N + 1, dtype=np.float64)
    last = np.zeros(N + 1, dtype=np.int64)

    # -----------------------------------------------------------------
    # Start core loop, add one cell at each iteration
    # -----------------------------------------------------------------
    for R in trange(1, N + 1, desc=desc, file=stdout):
        AR = tmp[:R]
        _fitness(
            N_r[:R], N_r[R], T_r[:R], T_r[R], ncp_prior, best[:R],
            out=AR, order='K', casting='no', ex_uses_vml=False
        )

        imax = AR.argmax()
        last[R] = imax
        best[R] = AR[imax]

    # -----------------------------------------------------------------
    # Now find changepoints by iteratively peeling off the last block
    # -----------------------------------------------------------------
    idx = N
    cp = [idx]
    while True:
        idx = last[idx]
        cp.append(idx)
        if idx == 0:
            break

    return np.flip(cp)
