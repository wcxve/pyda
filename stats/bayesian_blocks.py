# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 21:11:59 2022

@author: xuewc
"""

from sys import stdout

import numpy as np
import numexpr as ne
from tqdm import trange

__all__ = ['blocks_tte']


def blocks_tte(*t, p0=0.05, iteration=0, return_cp=False):
    len8_rjust = lambda s: str(s).rjust(2).rjust(5).ljust(8)
    len8_ljust = lambda s: str(s).ljust(2).rjust(5).ljust(8)

    N_r, T_r, edges, cp_shift, N, nunique = _data_matrix_tte(*t)

    N = np.full_like(N, N.sum())
    ncp_prior = 4 - np.log(73.53 * p0 * (N ** -0.478))
    ncp_prior /= len(N)

    tstr = _estimate_run_time(nunique, len(t))
    print(f'\nBayesian Blocks: about {tstr} to go')

    loop_events = _loop_events1 if len(t) == 1 else _loop_events2
    cp = loop_events(N_r, T_r, ncp_prior, desc='  EVT ')
    ncp = len(cp)
    fpr = 1 - (1 - p0) ** ncp
    print(f'  NCP : {ncp}\n  FPR : {fpr:.2e}\n')

    # -----------------------------------------------------------------
    # Iterate if desired
    # -----------------------------------------------------------------
    if iteration > 0:
        print(
            f'Bayesian Blocks: iteration starts, each step takes about '
            f'{tstr} to go'
        )

        ncp_hist = [ncp]
        cp_hist = [cp]
        fpr_hist = [fpr]

        nit = 0
        converge = False
        while nit < iteration:
            nit += 1

            fpr_single = 1 - (1 - p0) ** (1 / ncp_hist[-1])
            ncp_prior = 4 - np.log(73.53 * fpr_single * (N ** -0.478))
            ncp_prior /= len(N)

            cp = loop_events(N_r, T_r, ncp_prior, f'{nit} '.rjust(6))
            cp_hist.append(cp)

            ncp = len(cp)
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

    if not return_cp:
        return edges[cp]
    else:
        return cp + cp_shift[cp]


def _data_matrix_tte(*t):
    t = [np.asarray(ti, dtype=np.float64) for ti in t]

    if np.any([np.any(np.diff(ti) < 0.0) for ti in t]):
        raise ValueError('all elements in ``t`` must be ordered')

    unq_t = []
    counts = []
    for ti in t:
        unq_ti, counts_i = np.unique(ti, return_counts=True)
        unq_t.append(unq_ti)
        counts.append(counts_i)

    if len(t) == 1:

        t = t[0]
        unq_t = unq_t[0]
        counts = counts[0]

        N = t.size
        nunique = unq_t.size

        N_cumsum = np.hstack((0, counts)).cumsum()
        N_remainders = N_cumsum[-1] - N_cumsum

        edges = np.hstack((unq_t[0], (unq_t[1:] + unq_t[:-1])/2.0, unq_t[-1]))
        T_remainders = edges[-1] - edges

        cp_shift = np.hstack((0, np.cumsum(counts - 1)))

    else:

        N = np.array([ti.size for ti in t])
        N_unique = np.array([unq_ti.size for unq_ti in unq_t])
        nseries = len(t)
        nunique = np.sum(N_unique)

        argsort = np.hstack(unq_t).argsort()

        zeros = np.zeros((nseries, 1), dtype=np.int64)

        n_idx = np.hstack((0, N_unique)).cumsum()
        n = np.zeros((nseries, nunique), dtype=np.int64)
        for i in range(nseries):
            n[i][n_idx[i] : n_idx[i+1]] = counts[i]
        n = n[:, argsort]
        N_cumsum = np.hstack((zeros, n)).cumsum(axis=1)
        N_remainders = N_cumsum[:, -1:] - N_cumsum

        t_idx = np.hstack((zeros, np.where(n, 1, 0))).cumsum(axis=1)
        T_cumsum = np.zeros_like(N_cumsum, dtype=np.float64)
        for i in range(nseries):
            unqi = unq_t[i]
            t_cs = np.hstack((unqi[0], (unqi[1:] + unqi[:-1])/2.0, unqi[-1]))
            T_cumsum[i] = t_cs[t_idx[i]]
        T_remainders = T_cumsum[:, -1:] - T_cumsum

        unq = np.sort(np.hstack(unq_t))
        edges = np.hstack((unq[0], (unq[1:] + unq[:-1])/2.0, unq[-1]))
        cp_shift = np.hstack((0, np.cumsum(n.sum(0) - 1)))

    return N_remainders, T_remainders, edges, cp_shift, N, nunique


def _estimate_run_time(nunique, nseries):
    # run time estimated based on 8 performance-cores of M1 Max and vecLib
    s = 22 * (nunique/110000) ** 2
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
_fitness1 = ne.NumExpr(
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
def _loop_events1(N_r, T_r, ncp_prior, desc=None):
    ne.set_num_threads(ne._init_num_threads())

    T_r = np.asarray(T_r, dtype=np.float64)
    N_r = np.asarray(N_r, dtype=np.float64)
    N = len(N_r) - 1

    # arrays to store the best configuration
    tmp = np.empty(N, dtype=np.float64)
    best = np.zeros(N + 1, dtype=np.float64)
    last = np.zeros(N, dtype=np.int64)
    cp = np.zeros(N, int)

    # -----------------------------------------------------------------
    # Start core loop, add one cell at each iteration
    # -----------------------------------------------------------------
    for R in trange(1, N + 1, desc=desc, file=stdout):
        AR = tmp[:R]
        _fitness1(
            N_r[:R],
            N_r[R],
            T_r[:R],
            T_r[R],
            ncp_prior,
            best[:R],
            out=AR,
            order='K',
            casting='no',
            ex_uses_vml=False
        )

        imax = AR.argmax()
        last[R - 1] = imax
        best[R] = AR[imax]

    # -----------------------------------------------------------------
    # Now find changepoints by iteratively peeling off the last block
    # -----------------------------------------------------------------
    i_cp = N
    idx = N
    while True:
        i_cp -= 1
        cp[i_cp] = idx
        if idx == 0:
            break
        idx = last[idx - 1]

    return cp[i_cp:]


# -----------------------------------------------------------------
# Speed tricks: resolve once for fitness function used in the loop
# -----------------------------------------------------------------
# N - n: number of elements in each block
# T - t: width or duration of each block
_fitness2 = ne.NumExpr(
    ex='(N - n) * log((N - n) / (T - t)) - ncp_prior',
    signature=(
        ('N', np.float64),
        ('n', np.float64),
        ('T', np.float64),
        ('t', np.float64),
        ('ncp_prior', np.float64),
    ),
    optimization='aggressive',
    truediv=True
)
def _loop_events2(N_r, T_r, ncp_prior, desc=None):
    ne.set_num_threads(ne._init_num_threads())

    N_r = np.asarray(N_r, dtype=np.float64, order='C')
    T_r = np.asarray(T_r, dtype=np.float64, order='C')
    n = N_r.shape[0]
    N = N_r.shape[1] - 1

    # arrays to store the intermediate results and the best configuration
    idx = np.where(np.transpose(N_r[:, :-1] > N_r[:, 1:]))[1]
    f = np.full((n, N), -ncp_prior[:, None], dtype=np.float64)
    fsum = np.full(N, -ncp_prior.sum(), dtype=np.float64)
    tmp = np.empty(N, dtype=np.float64)
    best = np.zeros(N + 1, dtype=np.float64)
    last = np.zeros(N, dtype=np.int64)
    cp = np.zeros(N, int)

    # -----------------------------------------------------------------
    # Start core loop, add one cell at each iteration
    # -----------------------------------------------------------------
    for R in trange(1, N + 1, desc=desc, file=stdout):
        i = idx[R-1]
        fsumR = fsum[:R]
        tmpR = tmp[:R]

        _fitness2(
            N_r[i, :R],
            N_r[i, R],
            T_r[i, :R],
            T_r[i, R],
            ncp_prior[i],
            out=tmpR,
            order='K',
            casting='no',
            ex_uses_vml=False
        )

        np.add(fsumR, tmpR - f[i, :R], out=fsumR)
        f[i, :R] = tmpR
        np.add(fsumR, best[:R], out=tmpR)

        imax = tmpR.argmax()
        last[R - 1] = imax
        best[R] = tmpR[imax]

    # -----------------------------------------------------------------
    # Now find change points by iteratively peeling off the last block
    # -----------------------------------------------------------------
    i_cp = N
    idx = N
    while True:
        i_cp -= 1
        cp[i_cp] = idx
        if idx == 0:
            break
        idx = last[idx - 1]

    return cp[i_cp:]

# def _loop_events2(N_r, T_r, ncp_prior, desc=None):
#     ne.set_num_threads(ne._init_num_threads())

#     N_r = np.asarray(N_r, dtype=np.float64)
#     T_r = np.asarray(T_r, dtype=np.float64)
#     n = N_r.shape[0]
#     N = N_r.shape[1] - 1

#     # arrays to store the intermediate results and the best configuration
#     idx = np.where(np.transpose(N_r[:, :-1] > N_r[:, 1:]))[1]
#     tmp1 = np.full((n, N), -ncp_prior[:, None], dtype=np.float64)
#     tmp2 = np.empty(N, dtype=np.float64)
#     best = np.zeros(N + 1, dtype=np.float64)
#     last = np.zeros(N, dtype=np.int64)
#     cp = np.zeros(N, int)

#     # -----------------------------------------------------------------
#     # Start core loop, add one cell at each iteration
#     # -----------------------------------------------------------------
#     for R in trange(1, N + 1, desc=desc, file=stdout):
#         f = tmp1[:, :R]
#         i = idx[R-1]
#         _fitness2(
#             N_r[i, :R],
#             N_r[i, R],
#             T_r[i, :R],
#             T_r[i, R],
#             ncp_prior[i],
#             out=f[i],
#             order='K',
#             casting='no',
#             ex_uses_vml=False
#         )
#         AR = tmp2[:R]
#         np.sum(f, axis=0, out=AR)
#         np.add(AR, best[:R], out=AR)
#         imax = AR.argmax()
#         last[R - 1] = imax
#         best[R] = AR[imax]

#     # -----------------------------------------------------------------
#     # Now find changepoints by iteratively peeling off the last block
#     # -----------------------------------------------------------------
#     i_cp = N
#     idx = N
#     while True:
#         i_cp -= 1
#         cp[i_cp] = idx
#         if idx == 0:
#             break
#         idx = last[idx - 1]

#     return cp[i_cp:]
