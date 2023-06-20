# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 14:42:35 2022

@author: xuewc
"""

import numpy as np
from astropy.io import fits
from astropy.time import Time
from pyda.utils import utc_to_met

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


def get_sat_j2000(utc, orbit_file, ext=1):
    """
    Compute J2000 coordinates of satellite given the time and orbit file.

    Parameters
    ----------
    utc : array_like
        Dates and times, in ISO-8601 format.
    orbit_file : str
        File path of the orbit file.
    ext : int or str, optional
        Extension index or name of `orbit_file`. The default is 1.

    Returns
    -------
    array
        J2000 coordinates of satellite at given the time point.

    """
    with fits.open(orbit_file) as hdul:
        telescope = hdul[0].header['TELESCOP']
        sat = telescope if telescope != 'HEBS' else 'GECAM-C'
        orbit = hdul[ext].data

    if sat == 'HXMT':
        t, x, y, z, vx, vy, vz = (
            'TIME',
            'X', 'Y', 'Z',
            'VX', 'VY', 'VZ'
        )
    elif sat.startswith('GECAM'):
        t, x, y, z, vx, vy, vz = (
            'TIME',
            'X_J2000', 'Y_J2000', 'Z_J2000',
            'VX_J2000', 'VY_J2000', 'VZ_J2000'
        )

    elif sat == 'GLAST':
        t, x, y, z, vx, vy, vz = (
            'SCLK_UTC',
            'POS_X', 'POS_Y', 'POS_Z',
            'VEL_X', 'VEL_Y', 'VEL_Z'
        )
        sat = 'Fermi'
    else:
        raise ValueError(
            '`orbit_file` is only supported for "Fermi", "HXMT", and "GECAM"'
        )

    met = utc_to_met(utc, sat)
    indices = np.abs(orbit[t]-met).argmin()
    delta = met - orbit[t][indices]
    # use velocities to interpolate coordinate
    sat_x = orbit[x][indices] + delta*orbit[vx][indices]
    sat_y = orbit[y][indices] + delta*orbit[vy][indices]
    sat_z = orbit[z][indices] + delta*orbit[vz][indices]

    return np.column_stack((sat_x, sat_y, sat_z))


if __name__ == '__main__':
    ra, dec = 60.819, -75.379
    utc0 = '2023-03-07T15:44:06.650'
    gb_posatt = '/Users/xuewc/BurstData/GRB230307A/GECAM-B/gb_posatt_230307_15_v00.fits'
    gc_posatt = '/Users/xuewc/BurstData/GRB230307A/GECAM-C/gc_posatt_230307_15_v00.fits'
    f_poshist = '/Users/xuewc/BurstData/GRB230307A/Fermi/glg_poshist_all_230307_v01.fit'
    locb = get_sat_j2000(utc0, gb_posatt)
    locc = get_sat_j2000(utc0, gc_posatt)
    locf = get_sat_j2000(utc0, f_poshist)
    dt1 = time_diff(ra, dec, locb, locc)
    dt2 = time_diff(ra, dec, locb, locf)

    from astropy.time import Time
    from astropy.units import s
    utc = Time('2023-03-07T15:44:06.650', scale='utc') + [0, 6, 9, 13, 17, 21,
                                                          25, 29, 34, 39, 45,
                                                          50, 55, 61, 68, 76,
                                                          84, 92, 103, 116,
                                                          131, 153, 187] * s
    delta1 = []
    for t in utc.isot:
        locb = get_sat_j2000(t, gb_posatt)
        locc = get_sat_j2000(t, gc_posatt)
        dt = time_diff(ra, dec, locb, locc)
        delta1.append(dt[0])

    delta2 = []
    for t in utc.isot:
        locb = get_sat_j2000(t, gb_posatt)
        locf = get_sat_j2000(t, f_poshist)
        dt = time_diff(ra, dec, locb, locf)
        delta2.append(dt[0])