# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 03:06:59 2023

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import numpy as np
import xarray as xr

from astropy.io import fits
from numba import njit, prange

__all__ = ['gecam_tehist', 'gecam_thist', 'gecam_ehist', 'gecam_events']


def gecam_tehist(
    file, det, gain, erange, trange, dt, t0=0.0,
    return_evt=False, return_ds=True
):
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
    erange : tuple or list of tuples
        Energy range(s) of events to be discretized.
    trange : tuple or list of tuples
        Time range(s) of events to be discretized.
    dt : float or None
        Sampling period in time dimension. No bins within `trange` if None.
    t0 : float, optional
        Reference time for `trange`. If `trange` is in MET scale, then `t0`
        must be 0.0 (the default).
    return_evt : bool, optional
        Return raw events (in dict) if True, the default is False.
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
    if dt is not None:
        trange = np.array(np.atleast_2d(trange), dtype=np.float64)
        ntbin = np.squeeze(np.diff(trange, axis=1) / dt, axis=1)
        ntbin = np.around(ntbin).astype(int)
        tbins = [np.linspace(*tr, n + 1) for tr, n in zip(trange, ntbin)]
        tstart = np.hstack([tb[:-1] for tb in tbins])
        tstop = np.hstack([tb[1:] for tb in tbins])
        tbins = np.column_stack((tstart, tstop))
        _tbins = t0 + np.append(tstart, tstop[-1])
    else:
        tbins = np.array(trange, dtype=np.float64)
        if len(tbins.shape) == 1:
            tstart = tbins[:-1]
            tstop = tbins[1:]
        elif len(tbins.shape) == 2:
            tstart = tbins[:, 0]
            tstop = tbins[:, 1]
        else:
            raise ValueError('shape of `trange` must be (t+1,) or (t, 2)')
        trange = np.column_stack((tstart, tstop))
        tbins = trange
        _tbins = t0 + np.append(tstart, tstop[-1])

    # read in EBOUNDS and EVENTS
    with fits.open(file) as hdul:
        telescope = hdul['PRIMARY'].header['TELESCOP']
        telescope = telescope if telescope != 'HEBS' else 'GECAM-C'
        ebounds = hdul['EBOUNDS'].data[:447]
        # ebounds = hdul['EBOUNDS'].data[:448]
        # overflow = hdul['EBOUNDS'].data[447 + det + gain*25]
        # mask = ebounds['E_MAX'] <= overflow['E_MIN']
        # cmax = ebounds[mask]['CHANNEL'].max()
        # ebounds = hdul['EBOUNDS'].data[:cmax+2]
        # ebounds[-1] = overflow

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
    # FIXME: should FLAG=2/12/245/255 be filtered before exposure calculation?
    # FLAG 0: time is good, 1: time is ok, 2: time is bad
    # FLAG +10: non-recommended evt given EVT_PAIR info
    # FLAG +243: wrong gain bias
    # EVT_PAIR: 2 bits, the first is for flight and the second is for ground
    mask = evts['FLAG'] < 2
    evts = evts[mask]
    print(f'FLAG: filtered out {np.sum(~mask)} events with FLAG>=2')
    # evts = evts[c_array(evts['EVT_TYPE']) <= 2] # 3 is for over-width evts

    # Discretize reduced event data
    PI = c_array(evts['PI'])
    T = c_array(evts['TIME'])
    bins = (cbins, _tbins)
    counts = np.histogram2d(PI, T, bins)[0]
    rate = counts/exposure
    ebins_width = ebins[:, 1] - ebins[:, 0]
    flux = rate/ebins_width[:, None]
    counts_error = np.sqrt(counts)
    rate_error = counts_error / exposure
    flux_error = rate_error/ebins_width[:, None]

    if return_evt:
        evt = {
            'time': c_array(evts['TIME'] - t0),
            'channel': c_array(evts['PI'])
        }

    if return_ds:
        ds = xr.Dataset(
            data_vars={
                'counts': (['channel', 'time'], counts),
                'rate': (['channel', 'time'], rate),
                'flux': (['channel', 'time'], flux),
                'counts_error': (['channel', 'time'], counts_error),
                'rate_error': (['channel', 'time'], rate_error),
                'flux_error': (['channel', 'time'], flux_error),
                'exposure': (['time'], exposure),
                'tbins': (['time', 'edge'], tbins),
                'tbins_width': (['time'], tbins[:, 1] - tbins[:, 0]),
                'ebins': (['channel', 'edge'], ebins),
                'ebins_width': (['channel'], ebins_width),
                'telescope': telescope,
                'instrument': 'GRD',
                'detname': f'G{telescope[-1]}G{det_str}{"L" if gain else "H"}'
            },
            coords={
                'channel': channels,
                'time': np.mean(tbins, axis=1),
                'edge': ['start', 'stop']
            }
        )
        if return_evt:
            return ds, evt
        else:
            return ds
    else:
        if return_evt:
            return counts, exposure, channels, tbins, ebins, evt
        else:
            return counts, exposure, channels, tbins, ebins


def to_gecam_tehist(
    counts, error, ebins, channel, tbins, exposure, sat, det, gain
):
    r"""

    Parameters
    ----------
    counts : (c, t)
    error : (c, t)
    ebins : (2,) or (c, 2)
    channel : (c,)
    tbins : (2,) or (t, 2)
    exposure : (t,)
    sat : str
    det : int
    gain : int

    """
    counts = np.atleast_2d(np.array(counts))
    error = np.atleast_2d(np.array(error))
    ebins = np.atleast_2d(np.array(ebins))
    channel = np.atleast_1d(np.array(channel))
    tbins = np.atleast_2d(np.array(tbins))
    exposure = np.atleast_1d(np.array(exposure))

    if not len(channel.shape) == len(exposure.shape) == 1:
        raise ValueError(
            'dimension of channel and exposure must be <= 1'
        )

    if not len(counts.shape) == len(error.shape) == 2:
        raise ValueError(
            'dimension of counts and error must be <= 2'
        )

    if ebins.shape[1] != 2:
        raise ValueError('shape of ebins must be (2,) or (c, 2)')

    if tbins.shape[1] != 2:
        raise ValueError('shape of tbins must be (2,) or (t, 2)')

    if not counts.shape == error.shape == (len(channel), len(exposure))\
            == (len(ebins), len(tbins)):
        raise ValueError('the time and channel dims of input are inconsistent')

    rate = counts/exposure
    ebins_width = ebins[:, 1] - ebins[:, 0]
    flux = rate/ebins_width[:, None]
    counts_error = np.sqrt(counts)
    rate_error = counts_error / exposure
    flux_error = rate_error/ebins_width[:, None]

    return xr.Dataset(
        data_vars={
            'counts': (['channel', 'time'], counts),
            'rate': (['channel', 'time'], rate),
            'flux': (['channel', 'time'], flux),
            'counts_error': (['channel', 'time'], counts_error),
            'rate_error': (['channel', 'time'], rate_error),
            'flux_error': (['channel', 'time'], flux_error),
            'exposure': (['time'], exposure),
            'tbins': (['time', 'edge'], tbins),
            'tbins_width': (['time'], tbins[:, 1] - tbins[:, 0]),
            'ebins': (['channel', 'edge'], ebins),
            'ebins_width': (['channel'], ebins_width),
            'telescope': sat,
            'instrument': 'GRD',
            'detname': f'G{sat[-1]}G{det:02d}{"L" if gain else "H"}'
        },
        coords={
            'channel': channel,
            'time': np.mean(tbins, axis=1),
            'edge': ['start', 'stop']
        }
    )


def gecam_thist(file, det, gain, erange, trange, dt, t0=0.0, return_ds=True):
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
    erange : tuple or list of tuples
        Energy range(s) of events to be discretized.
    trange : tuple or list of tuples
        Time range(s) of events to be discretized.
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
        telescope = hdul['PRIMARY'].header['TELESCOP']
        telescope = telescope if telescope != 'HEBS' else 'GECAM-C'
        ebounds = hdul['EBOUNDS'].data[:447]
        # ebounds = hdul['EBOUNDS'].data[:448]
        # overflow = hdul['EBOUNDS'].data[447 + det + gain*25]
        # mask = ebounds['E_MAX'] <= overflow['E_MIN']
        # cmax = ebounds[mask]['CHANNEL'].max()
        # ebounds = hdul['EBOUNDS'].data[:cmax+2]
        # ebounds[-1] = overflow

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
    # FIXME: should FLAG=2/12/245/255 be filtered before exposure calculation?
    # FLAG 0: time is good, 1: time is ok, 2: time is bad
    # FLAG +10: non-recommended evt given EVT_PAIR info
    # FLAG +243: wrong gain bias
    # EVT_PAIR: 2 bits, the first is for flight and the second is for ground
    mask = evts['FLAG'] < 2
    evts = evts[mask]
    print(f'FLAG: filtered out {np.sum(~mask)} events with FLAG>=2')
    # evts = evts[c_array(evts['EVT_TYPE']) <= 2] # 3 is for over-width evts

    # Discretize reduced event data
    T = c_array(evts['TIME'])
    counts = np.histogram(T, _tbins)[0]
    rate = counts/exposure
    counts_error = np.sqrt(counts)
    rate_error = counts_error/exposure

    if return_ds:
        return xr.Dataset(
            data_vars={
                'counts': (['time'], counts),
                'rate': (['time'], rate),
                'counts_error': (['time'], counts_error),
                'rate_error': (['time'], rate_error),
                'exposure': (['time'], exposure),
                'tbins': (['time', 'edge'], tbins),
                'tbins_width': (['time'], tbins[:, 1] - tbins[:, 0]),
                'ebins': (['emid', 'edge'], erange),
                'telescope': telescope,
                'instrument': 'GRD',
                'detname': f'G{telescope[-1]}G{det_str}{"L" if gain else "H"}'
            },
            coords={
                'time': np.mean(tbins, axis=1),
                'emid': (erange[:, 0] + erange[:, 1])/2.0,
                'edge': ['start', 'stop']
            }
        )
    else:
        return counts, exposure, tbins


def gecam_ehist(file, det, gain, erange, trange, t0=0.0, return_ds=True):
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
    erange : tuple or list of tuples
        Energy range(s) of events to be discretized.
    trange : tuple or list of tuples
        Time range(s) of events to be discretized.
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
        telescope = hdul['PRIMARY'].header['TELESCOP']
        telescope = telescope if telescope != 'HEBS' else 'GECAM-C'
        ebounds = hdul['EBOUNDS'].data[:447]
        # ebounds = hdul['EBOUNDS'].data[:448]
        # overflow = hdul['EBOUNDS'].data[447 + det + gain*25]
        # mask = ebounds['E_MAX'] <= overflow['E_MIN']
        # cmax = ebounds[mask]['CHANNEL'].max()
        # ebounds = hdul['EBOUNDS'].data[:cmax+2]
        # ebounds[-1] = overflow

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
    # FIXME: should FLAG=2/12/245/255 be filtered before exposure calculation?
    # NOTE: time will not shift too much, so these are ok to not be filtered
    # FLAG 0: time is good, 1: time is ok, 2: time is bad
    # FLAG +10: non-recommended evt given EVT_PAIR info
    # FLAG +243: wrong gain bias
    # EVT_PAIR: 2 bits, the first is for flight and the second is for ground
    mask = evts['FLAG'] < 2
    evts = evts[mask]
    print(f'FLAG: filtered out {np.sum(~mask)} events with FLAG>=2')
    # evts = evts[c_array(evts['EVT_TYPE']) <= 2] # 3 is for over-width evts

    # Discretize reduced event data
    PI = c_array(evts['PI'])
    counts = np.histogram(PI, cbins)[0]
    rate = counts/exposure
    ebins_width = ebins[:, 1] - ebins[:, 0]
    flux = rate/ebins_width
    counts_error = np.sqrt(counts)
    rate_error = counts_error/exposure
    flux_error = rate_error/ebins_width

    if return_ds:
        return xr.Dataset(
            data_vars={
                'counts': (['channel'], counts),
                'rate': (['channel'], rate),
                'flux': (['channel'], flux),
                'counts_error': (['channel'], counts_error),
                'rate_error': (['channel'], rate_error),
                'flux_error': (['channel'], flux_error),
                'exposure': exposure,
                'gti': (['tmid', 'edge'], trange),
                'ebins': (['channel', 'edge'], ebins),
                'ebins_width': (['channel'], ebins_width),
                'telescope': telescope,
                'instrument': 'GRD',
                'detname': f'G{telescope[-1]}G{det_str}{"L" if gain else "H"}'
            },
            coords={
                'channel': channels,
                'tmid': (trange[:, 1] + trange[:, 0])/2.0,
                'edge': ['start', 'stop']
            }
        )
    else:
        return counts, exposure, channels, ebins


def _events(file, det, gain, erange, trange, t0):
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
    erange : tuple or list of tuples
        Energy range(s) of events.
    trange : tuple or list of tuples
        Time range(s) of events.
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
        ebounds = hdul['EBOUNDS'].data[:447]
        # ebounds = hdul['EBOUNDS'].data[:448]
        # overflow = hdul['EBOUNDS'].data[447 + det + gain*25]
        # mask = ebounds['E_MAX'] <= overflow['E_MIN']
        # cmax = ebounds[mask]['CHANNEL'].max()
        # ebounds = hdul['EBOUNDS'].data[:cmax+2]
        # ebounds[-1] = overflow

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


def gecam_events(file, dets, gains, erange, trange, t0):
    """
    Get raw event list of GECAM.

    Parameters
    ----------
    file : str
        File path of GECAM EVT data.
    dets : int or list of int
        Serial numbers of GECAM/GRD.
    gains : int or list of int
        Detector gain type. 0 for high gain and 1 for low gain.
    erange : tuple or list of tuples
        Energy range(s) of events.
    trange : tuple or list of tuples
        Time range(s) of events.
    t0 : float, optional
        Reference time for `trange`. If `trange` is in MET scale, then `t0`
        must be 0.0 (the default).

    Returns
    -------
    events : list of ndarray
        Events of given `erange`.

    """
    dets = np.atleast_1d(dets)
    gains = np.atleast_1d(gains)
    if gains.size == 1 and dets.size != 1:
        gains = gains.repeat(dets.size)
    erange = np.atleast_2d(erange)
    nenergy = len(erange)
    ndet = len(dets)
    evts = [[] for i in range(nenergy)]
    for i in range(ndet):
        res = _events(file, dets[i], gains[i], erange, trange, t0)
        for j in range(nenergy):
            evts[j].append(res[j])
    evts = [np.sort(np.hstack(i)) for i in evts]
    evts = evts[0] if len(evts) == 1 else evts
    return evts


@njit('boolean[::1](float64[::1], uint8[::1], float64[:,::1], int64)',
      cache=True)
def _mask_tg(t, g, trange, gain):
    n = t.size
    tmin = trange.min()
    tmax = trange.max()
    mask = np.full(n, False, dtype=np.bool_)
    ninterval = len(trange)
    single_interval = ninterval == 1
    for i in prange(n):
        ti = t[i]
        if tmin <= ti <= tmax and g[i] == gain:
            if single_interval:
                mask[i] = True
            else:
                for j in range(ninterval):
                    trange_j = trange[j]
                    if trange_j[0] <= ti <= trange_j[1]:
                        mask[i] = True
                        break
    return mask


@njit('boolean[::1](int16[::1], int16[::1])', cache=True)
def _mask_pi(pi, channels):
    n = pi.size
    cmin = channels.min()
    cmax = channels.max()
    mask = np.full(n, False, dtype=np.bool_)
    for i in prange(n):
        pii = pi[i]
        if cmin <= pii <= cmax:
            if pii in channels:
                mask[i] = True
    return mask


if __name__ == '__main__':
    file = '/Users/xuewc/BurstData/GRB230307A/GECAM-B/gbg_evt_230307_15_v01.fits'
    det = 1
    gain = 0
    trange = [-1,70]
    erange = [30,350]
    dt = 0.5
    t0 = 131903046.67

    evts = gecam_events(file, [det], [gain], trange, erange, t0)

    res = gecam_tehist(file, det, gain, erange, trange, dt, t0)
    res_t = gecam_thist(file, det, gain, trange, erange, dt, t0)
    res_e = gecam_ehist(file, det, gain, erange, trange, t0)
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
