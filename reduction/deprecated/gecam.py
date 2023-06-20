# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 03:13:15 2023

@author: Wang-Chen Xue < https://orcid.org/0000-0001-8664-5085 >
"""

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from astropy.io import fits
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from utils.time import met_to_utc, utc_to_met


DET = {
    'GECAM-A': [
        21, 12, 22, 14, 23,
        11,  4, 13,  5, 15,
        20,  3,  1,  6, 24,
        10,  2,  8,  7, 16,
        19,  9, 18, 17, 25
    ],
    'HEBS': np.arange(1, 13)
}
DET['GECAM-B'] = DET['GECAM-A']


DAQ = {
    'GECAM-A': [
        5, 3, 4, 1, 3,
        1, 4, 2, 5, 4,
        2, 3, 1, 2, 5,
        5, 2, 5, 3, 1,
        4, 1, 3, 4, 2
    ],
    'HEBS': [
        1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2
    ]
}
DAQ['GECAM-B'] = DAQ['GECAM-A']


def evt_exposure_by_gain(data, t0, tbins):
    # exposure.shape = (2, ntbin)
    time_mask = (t0 + tbins[0] <= data['TIME']) \
                & (data['TIME'] <= t0 + tbins[-1])
    data = data[time_mask]
    dvals = np.unique(data['DEAD_TIME'])
    dbins = np.append(dvals, dvals[-1] + 1)
    tmp = np.column_stack(
        (data['GAIN_TYPE'], data['TIME'] - t0, data['DEAD_TIME'])
    )
    gbins = np.arange(3)
    hist = np.histogramdd(tmp, (gbins, tbins, dbins))[0]
    deadtime = np.sum(hist*dvals, axis=2) * 1e-6
    exposure = np.diff(tbins) - deadtime
    return exposure


def evt_total_rate(data, t0, tbins):
    time_mask = (t0 + tbins[0] <= data['TIME']) \
                & (data['TIME'] <= t0 + tbins[-1])
    data = data[time_mask]
    gbins = (0, 1, 2)
    flag = (data['PI'] < 448) & (data['FLAG'] <= 1) & (data['EVT_TYPE'] == 1)
    counts = np.histogram2d(
        data[flag]['GAIN_TYPE'], data[flag]['TIME'] - t0,
        bins=[gbins, tbins]
    )[0]
    exposure = evt_exposure_by_gain(data, t0, tbins)
    rate = np.sum(counts / exposure, axis=0)
    return rate


def evt_channel_rate(data, t0, tbins):
    # rate.shape = (448, ntbin)
    time_mask = (t0 + tbins[0] <= data['TIME']) \
                & (data['TIME'] <= t0 + tbins[-1])
    data = data[time_mask]
    flag = (data['PI'] < 448) & (data['FLAG'] <= 1) & (data['EVT_TYPE'] == 1)
    tmp = np.column_stack(
        (data[flag]['PI'], data[flag]['TIME'] - t0, data[flag]['GAIN_TYPE'])
    )
    gbins = (0, 1, 2)
    cbins = np.arange(449)
    counts = np.histogramdd(tmp, bins=[cbins, tbins, gbins])[0]
    exposure = evt_exposure_by_gain(data, t0, tbins)
    rate = np.sum(counts / exposure.T, axis=2)
    return rate


def evt_lc_by_dets(evt_file, t0, tstart, tstop, bin_width, gain, vline=0.0,
                   dpi=100.0):
    tbins = np.linspace(tstart, tstop, round((tstop - tstart)/bin_width) + 1)
    with fits.open(evt_file) as hdul:
        telescope = hdul['PRIMARY'].header['TELESCOP']
        sat = 'GECAM-B' if telescope != 'HEBS' else 'GECAM-C'
        ax_shape = (5, 5) if sat != 'GECAM-C' else (4, 3)
        figsize = (12, 9) if sat != 'GECAM-C' else (4.5*1.5, 4*1.5)
        utc0 = met_to_utc(t0, sat)
        with plt.style.context(['science', 'nature', 'no-latex']):
            fig, axes = plt.subplots(
                *ax_shape, sharex=True, sharey=True, figsize=figsize, dpi=dpi
            )
            if sat != 'GECAM-C':
                fig.subplots_adjust(
                    left=0.07, right=0.93, top=0.97, bottom=0.04,
                    hspace=0, wspace=0
                )
            else:
                fig.subplots_adjust(
                    left=0.1, right=0.9, top=0.95, bottom=0.06,
                    hspace=0, wspace=0
                )
            fig.align_ylabels(axes)
            fig.suptitle(f'{telescope} light curve, '
                         f'$T_0=${utc0}$\,$UTC, '
                         f'bin width$=${bin_width}$\,$s',
                         y=0.98)
            for ax, det, daq in zip(axes.reshape(-1), DET[sat], DAQ[sat]):
                if sat != 'HEBS' or det not in [6 ,12] or gain != 1:
                    data = hdul[f'EVENTS{str(det).zfill(2)}'].data
                    data = data[data['GAIN_TYPE'] == gain]
                    rate = evt_total_rate(data, t0, tbins)
                    ax.step(tbins, np.append(rate, rate[-1]), where='post')
                    ax.axvline(vline, c='grey', ls='--', lw=0.618)
                    # ax.step(tbins, np.append(rate, rate[-1]), c='k', where='post')
                    # ax.axvline(0, c='r', ls='--', lw=0.618)
                ax.grid(ls=':')
                ax.annotate(f'GRD{str(det).zfill(2)}\nDAQ{daq} ',
                            xy=(0.96, 0.95), xycoords='axes fraction',
                            ha='right', va='top')

            axes[0, 0].set_xlim(tstart, tstop)
            for ax in axes[-1]:
                ax.set_xlabel('$T-T_0$ [s]')
            for axes_i in axes:
                axes_i[0].set_ylabel('Count Rate [s$^{-1}$]')


def evt_lc_by_energies(evt_file, t0, tstart, tstop, bin_width, dets=-1,
                       gain=0, emin=None, emax=None, cblog=False, sigma=1.5,
                       divide_ebin=False):
    tbins = np.linspace(tstart, tstop, round((tstop - tstart)/bin_width) + 1)
    rate = np.zeros((448, tbins.size - 1))
    with fits.open(evt_file) as hdul:
        ebounds = hdul['EBOUNDS'].data[:448]
        ebins = np.append(ebounds['E_MIN'], ebounds['E_MAX'][-1])
        telescope = hdul['PRIMARY'].header['TELESCOP']
        sat = 'GECAM-B' if telescope != 'HEBS' else 'GECAM-C'
        if dets == -1:
            det_list = np.arange(1, 26 if sat != 'GECAM-C' else 13)
        else:
            det_list = np.sort(dets)
        utc0 = met_to_utc(t0, sat)
        for det in det_list:
            data = hdul[f'EVENTS{str(det).zfill(2)}'].data
            data = data[data['GAIN_TYPE'] == gain]
            rate += evt_channel_rate(data, t0, tbins)

    emin = emin if emin is not None else ebins[0]
    emax = emax if emax is not None else ebins[-1]
    mask = (emin <= ebins[:-1]) & (ebins[:-1] <= emax)
    ebins = np.append(ebins[:-1][mask], ebins[1:][mask][-1])
    rate = rate[mask] / (np.diff(ebins)[:, np.newaxis] if divide_ebin else 1.0)

    X, Y = np.meshgrid(tbins, ebins)
    img = gaussian_filter(rate, sigma)
    if dets == -1:
        det_str = '1-25' if sat != 'GECAM-C' else '1-12'
    else:
        det_str = str(list(dets))[1:-1].replace(' ', '')
    #with plt.style.context(['science', 'nature', 'no-latex']):
    #    fig = plt.figure(figsize=(4, 3))
    plt.figure()
    norm = LogNorm() if cblog else None
    pcm = plt.pcolormesh(X, Y, img, norm=norm, cmap='jet')
    plt.semilogy()
    ebin_label = ' keV$^{-1}$' if divide_ebin else ''
    cb = plt.colorbar(pcm, pad=0, aspect=40,
                      label='Rate [s$^{-1}$%s]' % ebin_label)
    cb.ax.tick_params(which='both', length=0)
    for i in cb.ax.get_yticks():
        cb.ax.axhline(i, c='k', lw=0.8)
    plt.xlabel('$T-T_0$ [s]')
    plt.ylabel('Energy [keV]')
    plt.title(f'{telescope}/GRD {det_str} light curve\n'
              f'$T_0=${utc0}$\,$UTC')


def bin_total_rate(data, t0, tstart, tstop):
    time_mask = (t0 + tstart <= data['STARTTIME']) \
                & (data['ENDTIME'] <= t0 + tstop)
    data = data[time_mask]
    tbins = np.append(data['STARTTIME'], data['ENDTIME'][-1]) - t0
    nchan = data['COUNTS'].shape[1] // 2
    hcounts = data['COUNTS'][:, :nchan-1].sum(1)
    lcounts = data['COUNTS'][:, nchan:-1].sum(1)
    counts = np.column_stack((hcounts, lcounts))
    exposure = data['EXPOSURE']
    rate = counts / exposure
    return tbins, rate.transpose().copy()


def bin_channel_rate(data, t0, tstart, tstop):
    time_mask = (t0 + tstart <= data['STARTTIME']) \
                & (data['ENDTIME'] <= t0 + tstop)
    data = data[time_mask]
    tbins = np.append(data['STARTTIME'], data['ENDTIME'][-1]) - t0
    nchan = data['COUNTS'].shape[1] // 2
    hcounts = data['COUNTS'][:, :nchan-1]
    lcounts = data['COUNTS'][:, nchan:-1]
    counts = np.transpose([hcounts, lcounts], axes=(0, 2, 1))
    exposure = data['EXPOSURE'].T[:, None, :]
    rate = counts / exposure
    return tbins, rate.copy()


def bin_lc_by_dets(bin_file, t0, tstart, tstop, gain=0, dpi=100.0):
    g = 'H' if gain == 0 else 'L'
    with fits.open(bin_file) as hdul:
        telescope = hdul['PRIMARY'].header['TELESCOP']
        sat = 'GECAM' if telescope != 'HEBS' else 'HEBS'
        ax_shape = (5, 5) if sat == 'GECAM' else (4, 3)
        figsize = (12, 9) if sat == 'GECAM' else (4.5, 4)
        utc0 = met_to_utc(t0, sat)
        with plt.style.context(['science', 'nature', 'no-latex']):
            fig, axes = plt.subplots(
                *ax_shape, sharex=True, sharey=True, figsize=figsize, dpi=dpi
            )
            if sat == 'GECAM':
                fig.subplots_adjust(
                    left=0.07, right=0.93, top=0.97, bottom=0.04,
                    hspace=0, wspace=0
                )
            else:
                fig.subplots_adjust(
                    left=0.11, right=0.89, top=0.95, bottom=0.08,
                    hspace=0, wspace=0
                )
            fig.align_ylabels(axes)
            bin_width = 0.05 if 'bspec' in bin_file else 1
            fig.suptitle(f'{telescope} light curve ({g}), '
                         f'$T_0=${utc0}$\,$UTC, '
                         f'bin width$=${bin_width}$\,$s',
                         y=0.99)
            for ax, det, daq in zip(axes.reshape(-1), DET[sat], DAQ[sat]):
                data = hdul[f'SPECTRUM{str(det).zfill(2)}'].data
                tbins, rate = bin_total_rate(data, t0, tstart, tstop)
                ax.step(tbins, np.append(rate[gain], rate[gain][-1]), where='post')
                ax.axvline(0, c='grey', ls='--', lw=0.618)
                # ax.step(tbins, np.append(rate, rate[-1]), c='k', where='post')
                # ax.axvline(0, c='r', ls='--', lw=0.618)
                ax.grid(ls=':')
                ax.annotate(f'GRD{str(det).zfill(2)}\nDAQ{daq} ',
                            xy=(0.96, 0.95), xycoords='axes fraction',
                            ha='right', va='top')

            axes[0, 0].set_xlim(tstart, tstop)
            for ax in axes[-1]:
                ax.set_xlabel('$T-T_0$ [s]')
            for axes_i in axes:
                axes_i[0].set_ylabel('Count Rate [s$^{-1}$]')


def bin_lc_by_energies2(bin_file, t0, tstart, tstop, dets=-1, gain=0, dpi=100,
                        emin=None, emax=None, cblog=False, sigma=1.5):
    g = 'H' if gain == 0 else 'L'
    norm = LogNorm() if cblog else None
    with fits.open(bin_file) as hdul:
        telescope = hdul['PRIMARY'].header['TELESCOP']
        sat = 'GECAM' if telescope != 'HEBS' else 'HEBS'
        ax_shape = (5, 5) if sat == 'GECAM' else (4, 3)
        figsize = (12, 9) if sat == 'GECAM' else (4.5*1.2, 4*1.2)
        utc0 = met_to_utc(t0, sat)
        # with plt.style.context(['science', 'nature', 'no-latex']):
        fig, axes = plt.subplots(
            *ax_shape, sharex=True, sharey=True, figsize=figsize, dpi=dpi
        )
        if sat == 'GECAM':
            fig.subplots_adjust(
                left=0.07, right=0.93, top=0.97, bottom=0.04,
                hspace=0, wspace=0
            )
        else:
            fig.subplots_adjust(
                left=0.11, right=0.89, top=0.95, bottom=0.1,
                hspace=0.04, wspace=0.04
            )
        fig.align_ylabels(axes)
        fig.suptitle(f'{telescope} light curve ({g}), '
                     f'$T_0=${utc0}$\,$UTC',
                     y=0.99)
        for ax, det, daq in zip(axes.flat, DET[sat], DAQ[sat]):
            data = hdul[f'SPECTRUM{str(det).zfill(2)}'].data
            ebounds = hdul['EBOUNDS'].data[f'GRD{str(det).zfill(2)}'][:,1:]
            nbin = len(ebounds) // 2
            ebounds = ebounds[gain*nbin : (gain+1)*nbin-1]
            tbins, rate = bin_channel_rate(data, t0, tstart, tstop)

            emin = emin if emin is not None else ebounds[:,0][0]
            emax = emax if emax is not None else ebounds[:,1][-1]
            mask = (emin <= ebounds[:,0]) & (ebounds[:,1] <= emax)
            ebins = np.append(ebounds[:,0][mask],
                              ebounds[:,1][mask][-1])
            rate = rate[gain][mask]
            X, Y = np.meshgrid(tbins, ebins)
            img = gaussian_filter(rate, sigma)

            pcm = ax.pcolormesh(X, Y, img, norm=norm, cmap='jet')
            ax.semilogy()
            # cb = fig.colorbar(pad=0, aspect=40,
            #                   label='Rate [s$^{-1}$]')
        fig.subplots_adjust(right=1.0)
        cb = fig.colorbar(pcm, ax=axes.ravel().tolist(),
                          pad=0.01, aspect=40,
                          label='Rate [s$^{-1}$]')
        cb.ax.tick_params(which='both', length=0)
        for i in cb.ax.get_yticks():
            cb.ax.axhline(i, c='k', lw=0.8)
        axes[0, 0].set_xlim(tstart, tstop)
        for ax in axes[-1]:
            ax.set_xlabel('$T-T_0$ [s]')
        for axes_i in axes:
            axes_i[0].set_ylabel('Energy [keV]')


def bin_lc_by_energies(evt_file, t0, tstart, tstop, dets=-1,
                       emin=None, emax=None, cblog=False, sigma=1.5):
    rates = []
    with fits.open(evt_file) as hdul:
        ebounds = hdul['EBOUNDS'].data[:448]
        ebins = np.append(ebounds['E_MIN'], ebounds['E_MAX'][-1])
        telescope = hdul['PRIMARY'].header['TELESCOP']
        sat = 'GECAM' if telescope != 'HEBS' else 'HEBS'
        if dets == -1:
            det_list = np.arange(1, 26 if sat == 'GECAM' else 13)
        else:
            det_list = np.sort(dets)
        utc0 = met_to_utc(t0, sat)
        for det in det_list:
            data = hdul[f'SPECTRUM{str(det).zfill(2)}'].data
            tbins, rate = bin_channel_rate(data, t0, tstart, tstop)
            rates.append(rate)

    emin = emin if emin is not None else ebins[0]
    emax = emax if emax is not None else ebins[-1]
    mask = (emin <= ebins[:-1]) & (ebins[:-1] <= emax)
    ebins = np.append(ebins[:-1][mask], ebins[1:][mask][-1])
    rate = rate[mask]

    X, Y = np.meshgrid(tbins, ebins)
    img = gaussian_filter(rate, sigma)
    if dets == -1:
        det_str = '1-25' if sat == 'GECAM' else '1-12'
    else:
        det_str = str(list(dets))[1:-1].replace(' ', '')
    #with plt.style.context(['science', 'nature', 'no-latex']):
    #    fig = plt.figure(figsize=(4, 3))
    plt.figure()
    norm = LogNorm() if cblog else None
    pcm = plt.pcolormesh(X, Y, img, norm=norm, cmap='jet')
    plt.semilogy()
    cb = plt.colorbar(pcm, pad=0, aspect=40,
                      label='Rate [s$^{-1}$]')
    cb.ax.tick_params(which='both', length=0)
    for i in cb.ax.get_yticks():
        cb.ax.axhline(i, c='k', lw=0.8)
    plt.xlabel('$T-T_0$ [s]')
    plt.ylabel('Energy [keV]')
    plt.title(f'{telescope}/GRD {det_str} light curve\n'
              f'$T_0=${utc0}$\,$UTC')
