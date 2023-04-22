# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:42:35 2022

@author: xuewc
"""

import numpy as np
from astropy.io import fits
from astropy.time import Time


def time_diff(src_ra, src_dec, loc0_j2000, loc1_j2000):
    """
    Calculate the time difference of the arrival of the signal in loc1 respect
    to loc0.

    Parameters
    ----------
    src_ra : float
        DESCRIPTION.
    src_dec : float
        DESCRIPTION.
    loc1_j2000 : array_like
        DESCRIPTION.
    loc2_j2000 : array_like
        DESCRIPTION.

    Returns
    -------
    dt : array
        DESCRIPTION.

    """
    src_ra = np.radians(src_ra)
    src_dec = np.radians(src_dec)
    src_x = np.cos(src_dec)*np.cos(src_ra)
    src_y = np.cos(src_dec)*np.sin(src_ra)
    src_z = np.sin(src_dec)
    p_src = np.array([src_x, src_y, src_z])
    
    p0 = np.atleast_2d(loc0_j2000)
    p1 = np.atleast_2d(loc1_j2000)
    dp = p0 - p1
    
    dt = np.einsum('c,tc->t', p_src, dp) / 299792458.0
    return dt


def utc_to_met(utc, sat):
    if sat.upper() == 'GECAM':
        UTC0 = '2019-01-01T00:00:00'
    elif sat.upper() == 'HEBS':
        UTC0 = '2021-01-01T00:00:00'
    else:
        raise ValueError('Available values for `sat` are `GECAM` and `HEBS`')
    return (Time(utc, scale='utc') - Time(UTC0, scale='utc')).sec


def get_sat_j2000(met, orbit_file, sat, ext=1):
    """
    Compute J2000 coordinates of satellite given the time and orbit file.

    Parameters
    ----------
    met : array_like
        MET of the satellite.
    orbit_file : str
        File path of the orbit file.
    sat : str
        Satellite label, available values are `HXMT` and `GECAM`.
    ext : int or str, optional
        Extension index or name of orbit file. The default is 1.

    Returns
    -------
    array
        J2000 coordinates of satellite at given the time point.

    """
    with fits.open(orbit_file) as hdul:
        orbit = hdul[ext].data
    
    if sat == 'HXMT':
        x, y, z, vx, vy, vz = (
            'X', 'Y', 'Z',
            'VX', 'VY', 'VZ'
        )
    elif sat == 'GECAM':
        x, y, z, vx, vy, vz = (
            'X_J2000', 'Y_J2000', 'Z_J2000',
            'VX_J2000', 'VY_J2000', 'VZ_J2000'
        )
    else:
        raise ValueError('Available values for `sat` are `HXMT` and `GECAM`')
    time_mask = (
        int(met.min()) <= orbit['TIME']) & (orbit['TIME'] <= int(met.max())
    )
    orbit = orbit[time_mask]

    # get corresponding index of MET in orbit info
    indices = (met.astype(int) - orbit['TIME'][0]).astype(int)
    # get the decimal part of MET
    met_decimal = met - met.astype(int) 
    # use velocities to compute coordinate
    sat_x = orbit[x][indices] + met_decimal*orbit[vx][indices]
    sat_y = orbit[y][indices] + met_decimal*orbit[vy][indices]
    sat_z = orbit[z][indices] + met_decimal*orbit[vz][indices]
    
    return np.c_[sat_x, sat_y, sat_z]


if __name__ == '__main__':
    ra = 293.743
    dec = 21.896
    utc0 = '2022-10-14T19:21:39.100'
    met_b = utc_to_met(utc0, 'GECAM')
    met_c = utc_to_met(utc0, 'HEBS')
    gb_posatt = '/Users/xuewc/BurstData/FRB 221014/gb_posatt_221014_19_v00.fits'
    gc_posatt = '/Users/xuewc/BurstData/FRB 221014/gc_posatt_221014_19_v00.fits'
    locb = get_sat_j2000(met_b, gb_posatt, 'GECAM')
    locc = get_sat_j2000(met_c, gc_posatt, 'GECAM')
    dt = time_diff(ra, dec, locb, locc)