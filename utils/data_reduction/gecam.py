# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 03:06:59 2023

@author: Wang-Chen Xue < https://orcid.org/0000-0001-8664-5085 >
"""

import numpy as np
import xarray as xr

from astropy.io import fits
from numba import njit, prange

__all__ = ['tehist_gecam', 'thist_gecam', 'ehist_gecam', 'events_gecam']


def tehist_gecam(file, det, gain, trange, erange, dt, t0=0.0, return_ds=True):
    """
    Discretize raw events of GECAM in dimensions of time and energy.

    Parameters
    ----------
    file : str
        File path of GECAM EVT data.
    det : int
        Serial number of GECAM/GRD.
    gain : int
        Detector gain type. 0 for high gain and 1 for low gain.
    trange : tuple or list of tuples
        The time range(s) of events to be discretized.
    erange : tuple or list of tuples
        The energy range(s) of events to be discretized.
    dt : float
        Sampling period in time dimension.
    t0 : float, optional
        Reference time for `trange`. If `trange` is in MET scale, then `t0`
        must be 0.0 (the default).
    return_ds : bool, optional
        Return ``xarray.DataSet`` object if True (the default).

    Returns
    -------
    counts : ndarray of shape (C, T)
        Time-Energy Histogram.
    exposure : ndarray of shape (T,)
        Exposure of each time bin.
    channels : ndarray of shape (C,)
        Channels corresponding to `erange` and `ebins`.
    tbins : ndarray of shape (T, 2)
        Bins used for discretizing events in time dimension. Reference time for
        `tbins` is `t0`.
    ebins : ndarray of shape (C, 2)
        Bins used for discretizing events in energy dimension.

    """
    # alias for transforming an array to C-contiguous array
    c_array = np.ascontiguousarray

    # detector string
    det_str = str(det).zfill(2)

    # initialize time bins
    trange = np.atleast_2d(trange)
    ntbin = np.squeeze(np.diff(trange, axis=1) / dt, axis=1)
    ntbin = np.around(ntbin).astype(int)
    tbins = [np.linspace(*tr, n + 1) for tr, n in zip(trange, ntbin)]
    tstart = np.hstack([tb[:-1] for tb in tbins])
    tstop = np.hstack([tb[1:] for tb in tbins])
    tbins = np.column_stack((tstart, tstop))
    _tbins = t0 + np.append(tstart, tstop[-1])

    # read in EBOUNDS and EVENTS
    with fits.open(file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        overflow = ebounds[447 + det + gain*25]
        mask = ebounds['E_MAX'] <= overflow['E_MIN']
        cmax = ebounds[mask]['CHANNEL'].max()
        ebounds = hdul['EBOUNDS'].data[:cmax+2]
        ebounds[-1] = overflow

        evts = hdul[f'EVENTS{det_str}'].data

    # reduce data according to trange
    TIME = c_array(evts['TIME'], np.float64)
    GAIN = c_array(evts['GAIN_TYPE'])
    evts = evts[_mask_tg(TIME, GAIN, t0 + trange, gain)]

    # calculate exposure of each time bin
    T = c_array(evts['TIME'])
    DT = c_array(evts['DEAD_TIME'])
    exposure = tstop - tstart - 1e-6*np.histogram(T, _tbins, weights=DT)[0]

    # initialize energy bins
    erange = np.atleast_2d(erange)
    estart = c_array(erange[:, 0])
    estop = c_array(erange[:, 1])
    EMIN = c_array(ebounds['E_MIN'])
    EMAX = c_array(ebounds['E_MAX'])
    emask = (estart[:, None] <= EMIN) & (EMAX <= estop[:, None])
    ebounds = ebounds[emask.any(axis=0)]
    ebins = np.column_stack((ebounds['E_MIN'], ebounds['E_MAX']))
    channels = c_array(ebounds['CHANNEL'], np.int16)
    cbins = np.append(channels, channels[-1]+0.5)

    # now reduce data according to erange
    PI = c_array(evts['PI'], np.int16)
    evts = evts[_mask_pi(PI, channels)]
    # evts = evts[c_array(evts['EVT_TYPE']) <= 2] # 3 is for over-width evts

    # Discretize reduced event data
    PI = c_array(evts['PI'])
    T = c_array(evts['TIME'])
    bins = (cbins, _tbins)
    counts = np.histogram2d(PI, T, bins)[0]

    if return_ds:
        return xr.Dataset(
            data_vars={
                'counts': (['channel', 'time'], counts),
                'exposure': (['time'], exposure),
                'channels': (['channel'], channels),
                'tbins': (['time', 'edge'], tbins),
                'ebins': (['channel', 'edge'], ebins),
            },
            coords={
                'channel': channels,
                'time': np.mean(tbins, axis=1),
                'edge': ['left', 'right']
            }
        )
    else:
        return counts, exposure, channels, tbins, ebins


def thist_gecam(file, det, gain, trange, erange, dt, t0=0.0, return_ds=True):
    """
    Discretize raw events of GECAM in dimension of time.

    Parameters
    ----------
    file : str
        File path of GECAM EVT data.
    det : int
        Serial number of GECAM/GRD.
    gain : int
        Detector gain type. 0 for high gain and 1 for low gain.
    trange : tuple or list of tuples
        The time range(s) of events to be discretized.
    erange : tuple or list of tuples
        The energy range(s) of events to be discretized.
    dt : float
        Sampling period in time dimension.
    t0 : float, optional
        Reference time for `trange`. If `trange` is in MET scale, then `t0`
        must be 0.0 (the default).
    return_ds : bool, optional
        Return ``xarray.DataSet`` object if True (the default).

    Returns
    -------
    counts : ndarray of shape (T,)
        Time Histogram.
    exposure : ndarray of shape (T,)
        Exposure of each time bin.
    tbins : ndarray of shape (T, 2)
        Bins used for discretizing events in time dimension. Reference time for
        `tbins` is `t0`.

    """
    # alias for transforming an array to C-contiguous array
    c_array = np.ascontiguousarray

    # detector string
    det_str = str(det).zfill(2)

    # initialize time bins
    trange = np.atleast_2d(trange)
    ntbin = np.squeeze(np.diff(trange, axis=1) / dt, axis=1)
    ntbin = np.around(ntbin).astype(int)
    tbins = [np.linspace(*tr, n + 1) for tr, n in zip(trange, ntbin)]
    tstart = np.hstack([tb[:-1] for tb in tbins])
    tstop = np.hstack([tb[1:] for tb in tbins])
    tbins = np.column_stack((tstart, tstop))
    _tbins = t0 + np.append(tstart, tstop[-1])

    # read in EBOUNDS and EVENTS
    with fits.open(file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        overflow = ebounds[447 + det + gain*25]
        mask = ebounds['E_MAX'] <= overflow['E_MIN']
        cmax = ebounds[mask]['CHANNEL'].max()
        ebounds = hdul['EBOUNDS'].data[:cmax+2]
        ebounds[-1] = overflow

        evts = hdul[f'EVENTS{det_str}'].data

    # reduce data according to trange
    TIME = c_array(evts['TIME'], np.float64)
    GAIN = c_array(evts['GAIN_TYPE'])
    evts = evts[_mask_tg(TIME, GAIN, t0 + trange, gain)]

    # calculate exposure of each time bin
    T = c_array(evts['TIME'])
    DT = c_array(evts['DEAD_TIME'])
    exposure = tstop - tstart - 1e-6*np.histogram(T, _tbins, weights=DT)[0]

    # initialize energy bins
    erange = np.atleast_2d(erange)
    estart = c_array(erange[:, 0])
    estop = c_array(erange[:, 1])
    EMIN = c_array(ebounds['E_MIN'])
    EMAX = c_array(ebounds['E_MAX'])
    emask = (estart[:, None] <= EMIN) & (EMAX <= estop[:, None])
    ebounds = ebounds[emask.any(axis=0)]
    channels = c_array(ebounds['CHANNEL'], np.int16)

    # now reduce data according to erange
    PI = c_array(evts['PI'], np.int16)
    evts = evts[_mask_pi(PI, channels)]
    # evts = evts[c_array(evts['EVT_TYPE']) <= 2] # 3 is for over-width evts

    # Discretize reduced event data
    T = c_array(evts['TIME'])
    counts = np.histogram(T, _tbins)[0]

    if return_ds:
        return xr.Dataset(
            data_vars={
                'counts': (['time'], counts),
                'exposure': (['time'], exposure),
                'tbins': (['time', 'edge'], tbins),
            },
            coords={
                'time': np.mean(tbins, axis=1),
                'edge': ['left', 'right']
            }
        )
    else:
        return counts, exposure, tbins


def ehist_gecam(file, det, gain, trange, erange, t0=0.0, return_ds=True):
    """
    Discretize raw events of GECAM in dimension of energy.

    Parameters
    ----------
    file : str
        File path of GECAM EVT data.
    det : int
        Serial number of GECAM/GRD.
    gain : int
        Detector gain type. 0 for high gain and 1 for low gain.
    trange : tuple or list of tuples
        The time range(s) of events to be discretized.
    erange : tuple or list of tuples
        The energy range(s) of events to be discretized.
    t0 : float, optional
        Reference time for `trange`. If `trange` is in MET scale, then `t0`
        must be 0.0 (the default).
    return_ds : bool, optional
        Return ``xarray.DataSet`` object if True (the default).

    Returns
    -------
    counts : ndarray of shape (C,)
        Time-Energy Histogram.
    exposure : float
        Exposure of overall histogram.
    channels : ndarray of shape (C,)
        Channels corresponding to `erange` and `ebins`.
    ebins : ndarray of shape (C, 2)
        Bins used for discretizing events in energy dimension.

    """
    # alias for transforming an array to C-contiguous array
    c_array = np.ascontiguousarray

    # detector string
    det_str = str(det).zfill(2)

    # initialize time bins
    trange = np.atleast_2d(trange)

    # read in EBOUNDS and EVENTS
    with fits.open(file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        overflow = ebounds[447 + det + gain*25]
        mask = ebounds['E_MAX'] <= overflow['E_MIN']
        cmax = ebounds[mask]['CHANNEL'].max()
        ebounds = hdul['EBOUNDS'].data[:cmax+2]
        ebounds[-1] = overflow

        evts = hdul[f'EVENTS{det_str}'].data

    # reduce data according to trange
    TIME = c_array(evts['TIME'], np.float64)
    GAIN = c_array(evts['GAIN_TYPE'])
    evts = evts[_mask_tg(TIME, GAIN, t0 + trange, gain)]

    # calculate exposure of each time bin
    DT = c_array(evts['DEAD_TIME'])
    exposure = np.diff(trange, axis=1).sum() - 1e-6*DT.sum()

    # initialize energy bins
    erange = np.atleast_2d(erange)
    estart = c_array(erange[:, 0])
    estop = c_array(erange[:, 1])
    EMIN = c_array(ebounds['E_MIN'])
    EMAX = c_array(ebounds['E_MAX'])
    emask = (estart[:, None] <= EMIN) & (EMAX <= estop[:, None])
    ebounds = ebounds[emask.any(axis=0)]
    ebins = np.column_stack((ebounds['E_MIN'], ebounds['E_MAX']))
    channels = c_array(ebounds['CHANNEL'], np.int16)
    cbins = np.append(channels, channels[-1]+0.5)

    # now reduce data according to erange
    PI = c_array(evts['PI'], np.int16)
    evts = evts[_mask_pi(PI, channels)]
    # evts = evts[c_array(evts['EVT_TYPE']) <= 2] # 3 is for over-width evts

    # Discretize reduced event data
    PI = c_array(evts['PI'])
    counts = np.histogram(PI, cbins)[0]

    if return_ds:
        return xr.Dataset(
            data_vars={
                'counts': (['channel'], counts),
                'exposure': exposure,
                'channels': (['channel'], channels),
                'ebins': (['channel', 'edge'], ebins),
            },
            coords={
                'channel': channels,
                'edge': ['left', 'right']
            }
        )
    else:
        return counts, exposure, channels, ebins


def _events(file, det, gain, trange, erange, t0):
    """
    Get raw event list of GECAM.

    Parameters
    ----------
    file : str
        File path of GECAM EVT data.
    det : int
        Serial number of GECAM/GRD.
    gain : int
        Detector gain type. 0 for high gain and 1 for low gain.
    trange : tuple or list of tuples
        The time range(s) of events.
    erange : tuple or list of tuples
        The energy range(s) of events.
    t0 : float, optional
        Reference time for `trange`. If `trange` is in MET scale, then `t0`
        must be 0.0 (the default).

    Returns
    -------
    events : list of ndarray
        Events of given `erange`.

    """
    # alias for transforming an array to C-contiguous array
    c_array = np.ascontiguousarray

    # detector string
    det_str = str(det).zfill(2)

    # initialize time bins
    trange = np.atleast_2d(trange)

    # read in EBOUNDS and EVENTS
    with fits.open(file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        overflow = ebounds[447 + det + gain*25]
        mask = ebounds['E_MAX'] <= overflow['E_MIN']
        cmax = ebounds[mask]['CHANNEL'].max()
        ebounds = hdul['EBOUNDS'].data[:cmax+2]
        ebounds[-1] = overflow

        evts = hdul[f'EVENTS{det_str}'].data

    # reduce data according to trange
    TIME = c_array(evts['TIME'], np.float64)
    GAIN = c_array(evts['GAIN_TYPE'])
    evts = evts[_mask_tg(TIME, GAIN, t0 + trange, gain)]

    # initialize energy bins
    erange = np.atleast_2d(erange)
    estart = c_array(erange[:, 0])
    estop = c_array(erange[:, 1])
    EMIN = c_array(ebounds['E_MIN'])
    EMAX = c_array(ebounds['E_MAX'])
    emask = (estart[:, None] <= EMIN) & (EMAX <= estop[:, None])
    channels_list = [
        c_array(ebounds['CHANNEL'][mask], np.int16) for mask in emask
    ]

    PI = c_array(evts['PI'], np.int16)
    evts_list = [
        evts[_mask_pi(PI, channels)]['TIME'] - t0
        for channels in channels_list
    ]

    return evts_list


def events_gecam(file, dets, gains, trange, erange, t0):
    """
    Get raw event list of GECAM.

    Parameters
    ----------
    file : str
        File path of GECAM EVT data.
    dets : list of int
        Serial numbers of GECAM/GRD.
    gains : list of int
        Detector gain type. 0 for high gain and 1 for low gain.
    trange : tuple or list of tuples
        The time range(s) of events.
    erange : tuple or list of tuples
        The energy range(s) of events.
    t0 : float, optional
        Reference time for `trange`. If `trange` is in MET scale, then `t0`
        must be 0.0 (the default).

    Returns
    -------
    events : list of ndarray
        Events of given `erange`.

    """
    erange = np.atleast_2d(erange)
    nenergy = len(erange)
    ndet = len(dets)
    evts = [[] for i in range(nenergy)]
    for i in range(ndet):
        res = _events(file, dets[i], gains[i], trange, erange, t0)
        for j in range(nenergy):
            evts[j].append(res[j])
    evts = [np.sort(np.hstack(i)) for i in evts]
    return evts


@njit(
    'boolean[:](float64[::1], uint8[::1], float64[:,::1], int64)',
    cache=True, parallel=True
)
def _mask_tg(t, g, trange, gain):
    n = t.size
    tmin = trange.min()
    tmax = trange.max()
    mask = np.full(n, False, dtype=np.bool_)
    ninterval = len(trange)
    for i in prange(n):
        ti = t[i]
        if ti < tmin or ti > tmax:
            continue
        if g[i] != gain:
            continue
        if ninterval == 1:
            mask[i] = True
        else:
            for j in range(ninterval):
                trange_j = trange[j]
                if trange_j[0] <= ti and ti <= trange_j[1]:
                    mask[i] = True
                    break
    return mask


@njit(
    'boolean[:](int16[::1], int16[::1])',
    cache=True, parallel=True
)
def _mask_pi(pi, channels):
    n = pi.size
    cmin = channels.min()
    cmax = channels.max()
    mask = np.full(n, False, dtype=np.bool_)
    for i in prange(n):
        pii = pi[i]
        if pii < cmin or pii > cmax:
            continue
        else:
            if pii in channels:
                mask[i] = True
    return mask


if __name__ == '__main__':
    file = '/Users/xuewc/BurstData/GRB230307A/gbg_evt_230307_15_v01.fits'
    det = 1
    gain = 0
    trange = np.array([(-1,40),(50,70)])
    erange = [(30,100),(200, 99999)]
    # erange = [(30, 99999)]
    dt = 0.5
    t0 = 131903046.67

    evts = events_gecam(file, [det], [gain], trange, erange, t0)

    res = tehist_gecam(file, det, gain, trange, erange, dt, t0)
    res_t = thist_gecam(file, det, gain, trange, erange, dt, t0)
    res_e = ehist_gecam(file, det, gain, trange, erange, t0)
    # import matplotlib.pyplot as plt
    # plt.errorbar(np.mean(res_e[-1][:-1],axis=1),
    #              res_e[0][:-1],
    #              yerr=res_e[0][:-1]**0.5,
    #              xerr=np.squeeze(np.diff(res_e[-1][:-1], axis=1))/2,
    #              fmt=' ')
    # plt.errorbar(res_e[-1][-1,0],
    #              res_e[0][-1],
    #              yerr=np.sqrt(res_e[0][-1]),
    #              xerr=np.diff(res_e[-1][-1]),
    #              xlolims=True,
    #              )
    # plt.loglog()
