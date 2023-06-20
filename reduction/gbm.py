"""
Created at 15:55:20 on 2023-05-12

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import numpy as np
import xarray as xr

from astropy.io import fits
from numba import njit, prange

__all__ = ['gbm_tehist', 'gbm_thist', 'gbm_ehist', 'gbm_events']


def gbm_tehist(file, erange, trange, dt, t0=0.0, return_ds=True):
    """
    Discretize raw events of Fermi/GBM in dimensions of time and energy.

    Parameters
    ----------
    file : str
        File path of Fermi/GBM TTE data.
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
        evts = hdul[f'EVENTS'].data

    # reduce data according to trange
    TIME = c_array(evts['TIME'], np.float64)
    evts = evts[_mask_t(TIME, t0 + trange)]

    # calculate exposure of each time bin
    T = c_array(evts['TIME'])
    DT = np.where(evts['PHA'] < 127, 2.6e-6, 1e-5)
    exposure = tstop - tstart - np.histogram(T, _tbins, weights=DT)[0]

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
    PHA = c_array(evts['PHA'], np.int16)
    evts = evts[_mask_pha(PHA, channels)]

    # Discretize reduced event data
    PHA = c_array(evts['PHA'])
    T = c_array(evts['TIME'])
    bins = (cbins, _tbins)
    counts = np.histogram2d(PHA, T, bins)[0]

    if return_ds:
        return xr.Dataset(
            data_vars={
                'counts': (['channel', 'time'], counts),
                'exposure': (['time'], exposure),
                'tbins': (['time', 'edge'], tbins),
                'ebins': (['channel', 'edge'], ebins),
            },
            coords={
                'channel': channels,
                'time': np.mean(tbins, axis=1),
                'edge': ['start', 'stop']
            }
        )
    else:
        return counts, exposure, channels, tbins, ebins


def gbm_thist(file, erange, trange, dt, t0=0.0, return_ds=True):
    """
    Discretize raw events of Fermi/GBM in dimension of time.

    Parameters
    ----------
    file : str
        File path of Fermi/GBM TTE data.
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
        evts = hdul[f'EVENTS'].data

    # reduce data according to trange
    TIME = c_array(evts['TIME'], np.float64)
    evts = evts[_mask_t(TIME, t0 + trange)]

    # calculate exposure of each time bin
    T = c_array(evts['TIME'])
    DT = np.where(evts['PHA'] < 127, 2.6e-6, 1e-5)
    exposure = tstop - tstart - np.histogram(T, _tbins, weights=DT)[0]

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
    PHA = c_array(evts['PHA'], np.int16)
    evts = evts[_mask_pha(PHA, channels)]

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
                'edge': ['start', 'stop']
            }
        )
    else:
        return counts, exposure, tbins


def gbm_ehist(file, erange, trange, t0=0.0, return_ds=True):
    """
    Discretize raw events of Fermi/GBM in dimension of energy.

    Parameters
    ----------
    file : str
        File path of Fermi/GBM TTE data.
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

    # initialize time bins
    trange = np.atleast_2d(trange)

    # read in EBOUNDS and EVENTS
    with fits.open(file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        evts = hdul[f'EVENTS'].data

    # reduce data according to trange
    TIME = c_array(evts['TIME'], np.float64)
    evts = evts[_mask_t(TIME, t0 + trange)]

    # calculate exposure of each time bin
    DT = np.where(evts['PHA'] < 127, 2.6e-6, 1e-5)
    exposure = np.diff(trange, axis=1).sum() - DT.sum()

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
    PHA = c_array(evts['PHA'], np.int16)
    evts = evts[_mask_pha(PHA, channels)]

    # Discretize reduced event data
    PHA = c_array(evts['PHA'])
    counts = np.histogram(PHA, cbins)[0]

    if return_ds:
        return xr.Dataset(
            data_vars={
                'counts': (['channel'], counts),
                'exposure': exposure,
                'ebins': (['channel', 'edge'], ebins),
            },
            coords={
                'channel': channels,
                'edge': ['start', 'stop']
            }
        )
    else:
        return counts, exposure, channels, ebins


def _events(file, erange, trange, t0):
    """
    Get raw event list of Fermi/GBM.

    Parameters
    ----------
    file : str
        File path of Fermi/GBM TTE data.
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

    # initialize time bins
    trange = np.atleast_2d(trange)

    # read in EBOUNDS and EVENTS
    with fits.open(file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        evts = hdul[f'EVENTS'].data

    # reduce data according to trange
    TIME = c_array(evts['TIME'], np.float64)
    evts = evts[_mask_t(TIME, t0 + trange)]

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

    PHA = c_array(evts['PHA'], np.int16)
    evts_list = [
        evts[_mask_pha(PHA, channels)]['TIME'] - t0
        for channels in channels_list
    ]

    return evts_list


def gbm_events(files, erange, trange, t0):
    """
    Get raw event list of Fermi/GBM.

    Parameters
    ----------
    files : str
        Files path of Fermi/GBM TTE data.
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
    erange = np.atleast_2d(erange)
    nenergy = len(erange)
    evts = [[] for i in range(nenergy)]
    for file in files:
        res = _events(file, trange, erange, t0)
        for i in range(nenergy):
            evts[i].append(res[i])
    evts = [np.sort(np.hstack(i)) for i in evts]
    evts = evts[0] if len(evts) == 1 else evts
    return evts


@njit('boolean[::1](float64[::1], float64[:,::1])', cache=True)
def _mask_t(t, trange):
    n = t.size
    tmin = trange.min()
    tmax = trange.max()
    mask = np.full(n, False, dtype=np.bool_)
    ninterval = len(trange)
    single_interval = ninterval == 1
    for i in prange(n):
        ti = t[i]
        if tmin <= ti <= tmax:
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
def _mask_pha(pha, channels):
    n = pha.size
    cmin = channels.min()
    cmax = channels.max()
    mask = np.full(n, False, dtype=np.bool_)
    for i in prange(n):
        phai = pha[i]
        if cmin <= phai <= cmax:
            if phai in channels:
                mask[i] = True
    return mask


if __name__ == '__main__':
    file = '/Users/xuewc/BurstData/GRB230511A/GBM/glg_tte_n6_bn230511548_v00.fit'
    trange = [-50,150]
    erange = [8,900]
    dt = 0.5
    t0 = 705503315.718
    res = gbm_tehist(file, erange, trange, dt, t0)
    res_t = gbm_thist(file, erange, trange, dt, t0)
    res_e = gbm_ehist(file, erange, trange, t0)

    import matplotlib.pyplot as plt
    plt.step(res.time, res.counts.sum('channel')/res.exposure, where='mid')