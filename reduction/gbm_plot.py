"""
Created at 00:06:23 on 2023-05-15

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from pyda.reduction.gbm import gbm_ehist, gbm_tehist, gbm_thist
from pyda.utils.time import met_to_utc, utc_to_met
# from pyda.utils.time import get_YMDh

__all__ = ['plot_gbm_thist']

def plot_gbm_thist(
    data_path, utc0, trange, dt,
    erange_n=(8.0, 900.0), erange_b=(250.0, 40000.0),
    bkg_t=None, bkg_path=None, vlines=None
):
    t0 = utc_to_met(utc0, 'Fermi')
    # Y, M, D, h = get_YMDh(utc0)
    # YMD = Y[-2:] + M + D

    dets = [
        'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb',
        'b0', 'b1'
    ]

    tte_files = [
        # sorted(glob(f'{data_path}/glg_tte_{d}_{YMD}_{h}z_v??.fit*'))
        sorted(glob(f'{data_path}/glg_tte_{d}_*_v??.fit*'))
        for d in dets
    ]
    tte_files = [f[-1] if f else None for f in tte_files]

    if bkg_t is not None:
        t0_bkg = utc_to_met(utc0, 'Fermi') + bkg_t
        # Y, M, D, h = get_YMDh(met_to_utc(t0_bkg, 'Fermi'))
        # YMD = Y[-2:] + M + D
        bkg_path = data_path if bkg_path is None else bkg_path
        bkg_files = [
            # sorted(glob(f'{bkg_path}/glg_tte_{d}_{YMD}_{h}z_v??.fit*'))
            sorted(glob(f'{bkg_path}/glg_tte_{d}_*_v??.fit.gz'))
            for d in dets
        ]
        bkg_files = [f[-1] if f else None for f in bkg_files]
    else:
        bkg_files = [None for f in tte_files]

    # with plt.style.context(['science', 'nature', 'no-latex']):
    fig, axes = plt.subplots(
        nrows=4, ncols=4, sharex=True, figsize=(10, 6.18)
    )
    fig.subplots_adjust(
        top=0.94, bottom=0.08, left=0.069, right=0.93,
        hspace=0.06, wspace=0.25
    )
    fig.align_ylabels(axes)
    title = '$Fermi$/GBM Light Curves, $T_0$={}$\,$UTC, $\Delta t={}\,$s'
    fig.suptitle(title.format(utc0, dt))

    axes_flat = axes.flatten()
    axes_flat[0].set_xlim(np.min(trange), np.max(trange))

    # with plt.style.context(['science', 'nature', 'no-latex']):
    for ax, d, src_f, bkg_f in zip(axes_flat, dets, tte_files, bkg_files):
        ax.annotate(
            text=d, xy=(0.96, 0.95), xycoords='axes fraction',
            ha='right', va='top'
        )
        if src_f is None:
            continue
        erange = erange_n if d.startswith('n') else erange_b
        lc = gbm_thist(src_f, erange, trange, dt, t0)
        ax.step(lc.time, lc.counts / lc.exposure, where='mid')
        ax.grid(ls=':')
        #ax.set_yscale('log')

        if vlines is not None:
            vlines = np.atleast_1d(vlines)
            for vline in vlines:
                ax.axvline(vline, c='r', ls='--', lw=0.618)

        if bkg_f is not None:
            bkg = gbm_thist(bkg_f, trange, erange, dt, t0_bkg)
            ax.step(
                bkg.time, bkg.counts / bkg.exposure,
                c='orange', alpha=0.618, where='mid', zorder=0
            )

    for ax in axes_flat[-2:]:
        ax.set_axis_off()

    for i in [10, 11]:
        axes_flat[i].tick_params(labelbottom=True)

    for i in range(4):
        axes[i, 0].set_ylabel('Rate [s$^{-1}$]')

    for i in range(10, 14):
        axes_flat[i].set_xlabel('$t-T_0$ [s]')

    # axes_flat[14].annotate(
    #     text=f'NaI: {emin_n:.1f}–{emax_n:.1f}$\,$keV', xy=(0.05, 0.6),
    #     xycoords='axes fraction', ha='left', va='top'
    # )
    # axes_flat[14].annotate(
    #     text=f'BGO: {emin_b:.1f}–{emax_b:.1f}$\,$keV', xy=(0.05, 0.3),
    #     xycoords='axes fraction', ha='left', va='top'
    # )
    axes_flat[14].annotate(
        text=f'NaI: {erange_n}$\,$keV', xy=(0.05, 0.6),
        xycoords='axes fraction', ha='left', va='top'
    )
    axes_flat[14].annotate(
        text=f'BGO: {erange_b}$\,$keV', xy=(0.05, 0.3),
        xycoords='axes fraction', ha='left', va='top'
    )

    return fig, axes


if __name__ == '__main__':
    path = '/Users/xuewc/BurstData/GRB230511A/GBM'
    utc0 = '2023-05-11T13:08:30.718'
    trange = [-50,150]
    dt = 0.5
    fig, axes = plot_gbm_thist(path, utc0, trange, dt)