# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:58:10 2023

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from pyda.reduction.gecam import gecam_ehist, gecam_tehist, gecam_thist
from pyda.utils.time import met_to_utc, utc_to_met

__all__ = ['plot_gecam_tehist', 'plot_gecam_thist', 'plot_gecam_ehist']


DET = {
    'GECAM-A': [
        21, 12, 22, 14, 23,
        11,  4, 13,  5, 15,
        20,  3,  1,  6, 24,
        10,  2,  8,  7, 16,
        19,  9, 18, 17, 25
    ],
    'GECAM-B': [
        21, 12, 22, 14, 23,
        11,  4, 13,  5, 15,
        20,  3,  1,  6, 24,
        10,  2,  8,  7, 16,
        19,  9, 18, 17, 25
    ],
    'GECAM-C': [
        1, 2, 3,  4,  5,  6,
        7, 8, 9, 10, 11, 12
    ]
}

DAQ = {
    'GECAM-A': [
        5, 3, 4, 1, 3,
        1, 4, 2, 5, 4,
        2, 3, 1, 2, 5,
        5, 2, 5, 3, 1,
        4, 1, 3, 4, 2
    ],
    'GECAM-B': [
        5, 3, 4, 1, 3,
        1, 4, 2, 5, 4,
        2, 3, 1, 2, 5,
        5, 2, 5, 3, 1,
        4, 1, 3, 4, 2
    ],
    'GECAM-C': [
        1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2
    ]
}


def _create_figure(sat, sharey=True):
    if sat.upper() != 'GECAM-C':
        axes_shape = (5, 5)
        figsize = (12, 9)
        l, r, t, b, h, w = (0.06, 0.94, 0.95, 0.06, 0.00, 0.00)
    else:
        axes_shape = (4, 3)
        figsize = (4.5 * 1.7, 4 * 1.7)
        l, r, t, b, h, w = (0.10, 0.90, 0.95, 0.06, 0.00, 0.00)
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(*axes_shape, sharex=True, sharey=sharey)
    fig.subplots_adjust(left=l, right=r, top=t, bottom=b, hspace=h, wspace=w)
    fig.align_ylabels(axes)
    return axes, fig


def _get_met_and_utc(t0, sat):
    if type(t0) is str:
        utc0 = t0
        met0 = utc_to_met(utc0, sat)
    else:
        met0 = t0
        utc0 = met_to_utc(met0, sat)

    return met0, utc0


def _get_sat_name(evt_file):
    with fits.open(evt_file) as hdul:
        telescope = hdul['PRIMARY'].header['TELESCOP']
    sat = telescope if telescope != 'HEBS' else 'GECAM-C'

    return sat


def plot_gecam_ehist(evt_file, t0, trange, emin=8.0, emax=8000.0):
    r"""Plot spectra of GECAM detectors.

    Parameters
    ----------
    evt_file : str
        File path of GECAM EVT data.
    t0 : float
        Reference time for `trange`.
    trange : tuple or list of tuples
        Time range(s) of events.
    emin : float
        Minimum energy of spectra.
    emax : float
        Maximum energy of spectra.

    Returns
    -------
    fig : Figure
        `matplotlib.pyplot.figure`.
    axes : array of Axes
        `matplotlib.pyplot.axes.Axes`.

    """
    erange = [emin, emax]
    sat = _get_sat_name(evt_file)
    met0, utc0 = _get_met_and_utc(t0, sat)

    axes, fig = _create_figure(sat, sharey=True)
    title = r'{} Spectra, $T_0$={}$\,$UTC, Time Interval: $T_0$ + {}$\,$s'
    fig.suptitle(title.format(sat, utc0, trange))

    ncols = axes.shape[1]
    dets = DET[sat]
    daqs = DAQ[sat]
    for i in range(len(dets)):
        ax = axes[i//ncols, i%ncols]
        det = dets[i]
        daq = daqs[i]

        hg = gecam_ehist(evt_file, det, 0, erange, trange, met0)
        ebins_mid = hg['ebins'].mean('edge')
        ebins_width = np.squeeze(hg['ebins'].diff('edge'))
        rate = hg['counts'] / hg['exposure'] / ebins_width
        ax.scatter(ebins_mid, rate, c='tab:blue', s=3)

        lg = gecam_ehist(evt_file, det, 1, erange, trange, met0)
        ebins_mid = lg['ebins'].mean('edge')
        ebins_width = np.squeeze(lg['ebins'].diff('edge'))
        rate = lg['counts'] / lg['exposure'] / ebins_width
        ax.scatter(ebins_mid, rate, c='tab:orange', s=3)
        ax.annotate(f'GRD{str(det).zfill(2)}\nDAQ{daq} ',
                    xy=(0.96, 0.95), xycoords='axes fraction',
                    ha='right', va='top')
        ax.grid(which='both', axis='x', ls=':')
        ax.grid(which='major', axis='y', ls=':')
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, right=True)

    ax.set_xlim(emin, emax)
    ax.set_xscale('log')
    ax.set_yscale('log')

    for i in range(axes.shape[0]):
        axes[i, 0].set_ylabel('Rate [s$^{-1}$ keV$^{-1}$]')
    for i in range(axes.shape[1]):
        axes[-1, i].set_xlabel('Energy [keV]')

    return fig, axes


def plot_gecam_thist(evt_file, t0, tstart, tstop, dt, emin=8.0, emax=8000.0):
    trange = [tstart, tstop]
    erange = [emin, emax]
    sat = _get_sat_name(evt_file)
    met0, utc0 = _get_met_and_utc(t0, sat)

    axes, fig = _create_figure(sat, sharey=True)
    title = r'{} Light Curves, $T_0$={}$\,$UTC, $\Delta t={}\,$s'
    fig.suptitle(title.format(sat, utc0, dt))

    tbins = np.linspace(tstart, tstop, round((tstop - tstart) / dt) + 1)

    rates_hg = []
    for det in DET[sat]:
        lc = gecam_thist(evt_file, det, 0, erange, trange, dt, t0)
        rate = lc['counts'] / lc['exposure']
        rates_hg.append(np.append(rate, rate[-1]))

    rates_lg = []
    for det in DET[sat]:
        lc = gecam_thist(evt_file, det, 1, erange, trange, dt, t0)
        rate = lc['counts'] / lc['exposure']
        rates_lg.append(np.append(rate, rate[-1]))

    max_rate = np.max(rates_hg)
    n_hg = int(np.floor(np.log10(max_rate))) if max_rate > 0.0 else 0
    max_rate = np.max(rates_lg)
    n_lg = int(np.floor(np.log10(max_rate))) if max_rate > 0.0 else 0

    axes_ = axes.flatten()
    twinxes_ = [ax.twinx() for ax in axes_]
    twinxes_[0].get_shared_y_axes().join(*twinxes_)
    twinxes = np.reshape(twinxes_, axes.shape)

    dets = DET[sat]
    daqs = DAQ[sat]
    for i in range(len(dets)):
        ax = axes_[i]
        twinx = twinxes_[i]
        det = dets[i]
        daq = daqs[i]
        rate_hg = rates_hg[i]
        rate_lg = rates_lg[i]

        twinx.step(tbins, rate_lg / 10 ** n_lg,
                   where='post', color='tab:orange', alpha=0.6, zorder=10)
        # twinx.grid(axis='y', ls=':', c='tab:orange')
        twinx.tick_params(axis='both', which='both', direction='in', left=True)
        twinx.tick_params(axis='y', colors='tab:orange', which='both')
        twinx.yaxis.label.set_color('tab:orange')

        ax.annotate(f'GRD{str(det).zfill(2)}\nDAQ{daq} ',
                    xy=(0.96, 0.95), xycoords='axes fraction',
                    ha='right', va='top')
        ax.step(tbins, rate_hg / 10 ** n_hg,
                where='post', color='tab:blue', alpha=0.6, zorder=10)
        # ax.grid(axis='y', ls=':', c='tab:blue')
        ax.tick_params(axis='both', which='both', direction='in', right=True,
                       top=True)
        ax.tick_params(axis='y', colors='tab:blue', which='both')
        ax.spines['left'].set_color('tab:blue')
        ax.spines['right'].set_color('tab:orange')
        ax.yaxis.label.set_color('tab:blue')

    ax.set_xlim(tstart, tstop)
    ax.set_ylim(bottom=0.0)
    twinx.set_ylim(bottom=0.0)

    for i in range(axes.shape[0]):
        axes[i, 0].set_ylabel('HG Rate [10$^{%s}$ s$^{-1}$]' % n_hg)
        twinxes[i, -1].set_ylabel('LG Rate [10$^{%s}$ s$^{-1}$]' % n_lg)
        for j in range(axes.shape[1] - 1):
            twinxes[i, j].tick_params(labelright=False)

    for i in range(axes.shape[1]):
        axes[-1, i].set_xlabel('$t-T_0$ [s]')

    return fig, axes


def plot_gecam_tehist(evt_file, t0, tstart, tstop, dt, emin=8.0, emax=8000.0):
    trange = [tstart, tstop]
    erange = [emin, emax]
    sat = _get_sat_name(evt_file)
    met0, utc0 = _get_met_and_utc(t0, sat)

    if sat.upper() != 'GECAM-C':
        axes_shape = (5, 5)
        figsize = (12, 9)
        l, r, t, b = (0.05, 0.92, 0.95, 0.06)
    else:
        axes_shape = (4, 3)
        figsize = (4.5 * 1.7, 4 * 1.7)
        l, r, t, b = (0.08, 0.88, 0.95, 0.06)

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=l, right=r, top=t, bottom=b)
    gs = fig.add_gridspec(*axes_shape, hspace=0.12)

    title = r'{} Events Distribution, $T_0$={}$\,$UTC, $\Delta t={}\,$s'
    fig.suptitle(title.format(sat, utc0, dt))

    tbins = np.linspace(tstart, tstop, round((tstop - tstart) / dt) + 1)

    Rmin = []
    Rmax = []

    rates_hg = []
    ebins_hg = []
    for det in DET[sat]:
        hist = gecam_tehist(evt_file, det, 0, erange, trange, dt, t0)
        ebins_width = np.squeeze(hist['ebins'].diff('edge'))
        rate = hist['counts'] / hist['exposure'] / ebins_width
        indices = np.flatnonzero(np.any(rate > 0.0, axis=1))
        i1 = indices.min()
        i2 = indices.max() + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            log10_rate = np.nan_to_num(np.log10(rate[i1:i2]), neginf=np.nan)
        rates_hg.append(log10_rate)
        Rmin.append(np.nanmin(log10_rate))
        Rmax.append(np.nanmax(log10_rate))
        ebins = np.append(hist['ebins'].sel(edge='start')[i1:i2],
                          hist['ebins'].sel(edge='stop')[i1:i2][-1])
        ebins_hg.append(ebins)

    rates_lg = []
    ebins_lg = []
    for det in DET[sat]:
        if sat == 'GECAM-C' and det in [6, 12]:
            rates_lg.append([])
            ebins_lg.append([])
            continue
        hist = gecam_tehist(evt_file, det, 1, erange, trange, dt, t0)
        ebins_width = np.squeeze(hist['ebins'].diff('edge'))
        rate = hist['counts'] / hist['exposure'] / ebins_width
        indices = np.flatnonzero(np.any(rate > 0.0, axis=1))
        i1 = indices.min()
        i2 = indices.max() + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            log10_rate = np.nan_to_num(np.log10(rate[i1:i2]), neginf=np.nan)
        rates_lg.append(log10_rate)
        Rmin.append(np.nanmin(log10_rate))
        Rmax.append(np.nanmax(log10_rate))
        ebins = np.append(hist['ebins'].sel(edge='start')[i1:i2],
                          hist['ebins'].sel(edge='stop')[i1:i2][-1])
        ebins_lg.append(ebins)

    Norm = plt.Normalize(vmin=np.min(Rmin), vmax=np.max(Rmax))
    if sat != 'GECAM-C':
        emin_hg = np.min(np.hstack(ebins_hg))
        emax_hg = np.max(np.hstack(ebins_hg))
    else:
        emin_hg = np.min(np.hstack(ebins_hg[:5] + ebins_hg[6:-1]))
        emax_hg = np.max(np.hstack(ebins_hg[:5] + ebins_hg[6:-1]))
    emin_lg = np.min(np.hstack(ebins_lg))
    emax_lg = np.max(np.hstack(ebins_lg))

    dets = DET[sat]
    daqs = DAQ[sat]
    ncols = axes_shape[1]
    for i in range(len(dets)):
        det = dets[i]
        daq = daqs[i]

        gs_i = gs[i//ncols, i%ncols]

        if sat == 'GECAM-C' and det in [6, 12]:
            ax_hg = fig.add_subplot(gs_i)
            ebin_hg = ebins_hg[i]
            rate_hg = rates_hg[i]
            T, E = np.meshgrid(tbins, ebin_hg)
            _ = ax_hg.pcolormesh(T, E, rate_hg, cmap='jet', norm=Norm)
            ax_hg.annotate(f'GRD{str(det).zfill(2)}\nDAQ{daq} ',
                           xy=(0.96, 0.95), xycoords='axes fraction',
                           ha='right', va='top')
            ax_hg.tick_params(axis='both', which='both', direction='in',
                              top=True, right=True)
            ax_hg.set_yscale('log')
            if det == 12:
                ax_hg.set_xlabel('$t-T_0$ [s]')
            else:
                ax_hg.tick_params(labelbottom=False)

            continue

        axes = gs_i.subgridspec(2, 1, hspace=0.05).subplots(sharex=True)
        if gs_i.is_first_col():
            axes[1].set_ylabel('              Energy [keV]')
        if gs_i.is_last_row():
            axes[1].set_xlabel('$t-T_0$ [s]')
        else:
            axes[1].tick_params(labelbottom=False)

        ax_hg = axes[1]
        ebin_hg = ebins_hg[i]
        rate_hg = rates_hg[i]
        T, E = np.meshgrid(tbins, ebin_hg)
        _ = ax_hg.pcolormesh(T, E, rate_hg, cmap='jet', norm=Norm)
        # ax_hg.annotate(f'GRD{str(det).zfill(2)}\nDAQ{daq} ',
        #                xy=(0.96, 0.95), xycoords='axes fraction',
        #                ha='right', va='top')
        ax_hg.tick_params(axis='both', which='both', direction='in',
                          top=True, right=True)
        ax_hg.set_ylim(emin_hg, emax_hg)
        ax_hg.set_yscale('log')

        ax_lg = axes[0]
        ebin_lg = ebins_lg[i]
        rate_lg = rates_lg[i]
        T, E = np.meshgrid(tbins, ebin_lg)
        _ = ax_lg.pcolormesh(T, E, rate_lg, cmap='jet', norm=Norm)
        ax_lg.annotate(f'GRD{str(det).zfill(2)}\nDAQ{daq} ',
                       xy=(0.96, 0.95), xycoords='axes fraction',
                       ha='right', va='top')
        ax_lg.tick_params(axis='both', which='both', direction='in',
                          top=True, right=True)
        ax_lg.set_ylim(emin_lg, emax_lg)
        ax_lg.set_yscale('log')

    if sat != 'GECAM-C':
        cax = fig.add_axes([0.933, 0.2, 0.01, 1 - 0.2 * 2])
    else:
        cax = fig.add_axes([0.895, 0.2, 0.01, 1 - 0.2 * 2])
    cb = fig.colorbar(_, cax=cax, label=r'$\log_{10}({\rm Rate/s/keV})$')
    cb.ax.tick_params(which='both', length=0)
    for i in cb.ax.get_yticks():
        cb.ax.axhline(i, c='k', lw=0.8)

    return fig, axes


if __name__ == '__main__':
    # evt_file = '/Users/xuewc/BurstData/GRB221009A/gcg_evt_221009_12_v05.fits'
    # t0 = utc_to_met('2022-10-09T12:00:20', 'GECAM-C')
    evt_file = '/Users/xuewc/BurstData/GRB230307A/GECAM-B/gbg_evt_230307_15_v01.fits'
    t0 = utc_to_met('2023-03-07T15:44:06.670', 'GECAM-B')
    fig, axes = plot_gecam_thist(evt_file, t0, -1, 70, 1)
    fig, axes = plot_gecam_tehist(evt_file, t0, -1, 70, 0.1)
    fig, axes = plot_gecam_ehist(evt_file, t0, trange=[-1, 70])
