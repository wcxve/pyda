#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 04:20:05 2022

@author: xuewc
"""

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from utils.time import utc_to_met, met_to_utc, get_YMDh


def get_exposure(events, t0, tbins):
    chbins = (0, 127, 128)
    hist = np.histogram2d(
        events['TIME'],
        events['PHA'],
        bins=(t0 + tbins, chbins)
    )[0]

    deadtime = np.sum(hist * (2.6, 10.0), axis=1)
    exposure = tbins[1:] - tbins[:-1] - deadtime * 1e-6

    return exposure


def get_lightcurve(tte_file, t0, tstart, tstop, tbins, emin, emax):
    with fits.open(tte_file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        events = hdul['EVENTS'].data

    emask = (emin <= ebounds['E_MIN']) & (ebounds['E_MAX'] <= emax)
    chmax = ebounds['CHANNEL'][emask].max()
    chmin = ebounds['CHANNEL'][emask].min()

    tmask = (t0 + tstart <= events['TIME']) & (events['TIME'] <= t0 + tstop)
    events = events[tmask]

    exposure = get_exposure(events, t0, tbins)

    chmask = (chmin <= events['PHA']) & (events['PHA'] <= chmax)
    events = events[chmask]

    counts = np.histogram(events['TIME'], bins=t0 + tbins)[0]

    return counts, exposure


def tte_lc_by_dets(
    data_path, utc0, tstart, tstop, bin_width,
    nai_emin=8.0, nai_emax=900.0, bgo_emin=250.0, bgo_emax=40000.0,
    bkg_shift=None, vlines=None
):
    t0 = utc_to_met(utc0, 'Fermi')
    Y, M, D, h = get_YMDh(utc0)
    YMD = Y[-2:] + M + D

    dets = [
        'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb',
        'b0', 'b1'
    ]

    tte_files = [
        sorted(glob(f'{data_path}/glg_tte_{d}_{YMD}_{h}z_v??.fit.gz'))[-1]
        for d in dets
    ]
    if bkg_shift is not None:
        t0_ = utc_to_met(utc0, 'Fermi') + bkg_shift
        Y, M, D, h = get_YMDh(met_to_utc(t0_, 'Fermi'))
        YMD = Y[-2:] + M + D
        bkg_files = [
            sorted(glob(f'{data_path}/glg_tte_{d}_{YMD}_{h}z_v??.fit.gz'))[-1]
            for d in dets
        ]
    else:
        bkg_files = [None for d in dets]

    tbins = np.arange(tstart, tstop + bin_width, bin_width)
    ebounds = {
        'n': {
            'emin': nai_emin,
            'emax': nai_emax
        },
        'b': {
            'emin': bgo_emin,
            'emax': bgo_emax
        }
    }

    with plt.style.context(['science', 'nature', 'no-latex']):
        fig, axes = plt.subplots(
            nrows=4, ncols=4, sharex=True, figsize=(10, 6.18)
        )
        fig.subplots_adjust(
            top=0.94, bottom=0.08, left=0.069, right=0.93,
            hspace=0.06, wspace=0.25
        )
        fig.align_ylabels(axes)
        fig.suptitle(
            f'$Fermi$/GBM light curve, $T_0=${utc0}$\,$UTC, '
            f'bin width$=${bin_width}$\,$s',
            y=0.97
        )

    axes_flat = axes.flatten()
    axes_flat[0].set_xlim(tstart, tstop)

    with plt.style.context(['science', 'nature', 'no-latex']):
        for ax, d, f, f_ in zip(axes_flat, dets, tte_files, bkg_files):
            det_type = d[:1]
            emin = ebounds[det_type]['emin']
            emax = ebounds[det_type]['emax']
            cnt, expo = get_lightcurve(f, t0, tstart, tstop, tbins, emin, emax)
            rate = cnt / expo

            ax.step(tbins, np.append(rate, rate[-1]), where='post')
            ax.grid(ls=':')
            ax.annotate(
                text=d, xy=(0.96, 0.95), xycoords='axes fraction',
                ha='right', va='top'
            )
            #ax.set_yscale('log')

            if vlines is not None:
                vlines = np.atleast_1d(vlines)
                for vline in vlines:
                    ax.axvline(vline, c='r', ls='--', lw=0.618)

            if f_ is not None:
                c,e = get_lightcurve(f_, t0_, tstart, tstop, tbins, emin, emax)
                bkg = c / e
                ax.step(
                    tbins, np.append(bkg, bkg[-1]),
                    c='orange', alpha=0.618, where='post', zorder=0
                )

    for ax in axes_flat[-2:]:
        ax.set_axis_off()

    for i in [10, 11]:
        axes_flat[i].tick_params(labelbottom=True)

    for i in range(4):
        axes[i, 0].set_ylabel('Count Rate [s$^{-1}$]')

    for i in range(10, 14):
        axes_flat[i].set_xlabel('$T-T_0$ [s]')

    axes_flat[14].annotate(
        text=f'NaI: {nai_emin:.1f}–{nai_emax:.1f}$\,$keV', xy=(0.05, 0.6),
        xycoords='axes fraction', ha='left', va='top'
    )
    axes_flat[14].annotate(
        text=f'BGO: {bgo_emin:.1f}–{bgo_emax:.1f}$\,$keV', xy=(0.05, 0.3),
        xycoords='axes fraction', ha='left', va='top'
    )

    return fig, axes


if __name__ == '__main__':
    # data_path = '/Users/xuewc/Downloads/FRB221128A'
    # utc0 = '2022-11-28T17:02:22.72'
    # tstart = -20
    # tstop = 15
    # bin_width = 0.1
    # tte_lc_by_dets(data_path, utc0, tstart, tstop, bin_width, vline=-3.0)

    data_path = '/Users/xuewc/BurstData/GRB221009A/Fermi_GBM/'
    utc0 = '2022-10-09T13:17:00.050'
    tstart = 500
    tstop = 1800
    bin_width = 5
    tte_lc_by_dets(data_path, utc0, tstart, tstop, bin_width, vline=None)
