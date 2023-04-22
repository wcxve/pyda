# -*- coding: utf-8 -*-
"""
@author: xuewc<xuewc@ihep.ac.cn>
"""

from typing import Optional, Union

import astropy.units as u
import numpy as np
from numpy.typing import ArrayLike, NDArray
from astropy.coordinates import GCRS, ITRS, WGS84GeodeticRepresentation
from astropy.constants import R_earth
from astropy.io import fits
from astropy.time import Time

from ..utils.time import utc_to_met
from .msis.density import column_density
from .xcom import calculate_cross_section


def get_sat_j2000(
    utc: ArrayLike,
    orbit_file: str,
    ext: Optional[Union[int, str]] = 1
) -> NDArray:
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
        sat = hdul[0].header['TELESCOP']
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
    time_mask = (
        int(met.min()) <= orbit[t]) & (orbit[t] <= int(met.max())
    )
    orbit = orbit[time_mask]

    # get corresponding index of MET in orbit info
    indices = (met.astype(int) - orbit[t][0]).astype(int)
    # get the decimal part of MET
    met_decimal = met - met.astype(int)
    # use velocities to interpolate coordinate
    sat_x = orbit[x][indices] + met_decimal*orbit[vx][indices]
    sat_y = orbit[y][indices] + met_decimal*orbit[vy][indices]
    sat_z = orbit[z][indices] + met_decimal*orbit[vz][indices]

    return np.column_stack((sat_x, sat_y, sat_z))


def calc_tangent_height(src_j2000, utc, orbit_file):
    """
    Calculate tangent height of the line of sight.

    Parameters
    ----------
    src_j2000 : array_like
        J2000 coordinate (R.A., Dec.) of source.
    loc_j2000 : array_like
        Array of J2000 coordinate (x, y, z), in unit m.
    orbit_file : str
        File path of the orbit file.

    Returns
    -------
    height : array
        Tangent height array.

    """
    f = 1/298.257
    z_factor = (1 - f)**(-2)

    utc = Time(np.atleast_1d(utc), scale='utc')

    src_j2000 = np.asarray(src_j2000)
    src = GCRS(
        ra=src_j2000[0].repeat(utc.size) * u.deg,
        dec=src_j2000[1].repeat(utc.size) * u.deg,
        obstime=utc
    ).transform_to(
        ITRS(obstime=utc, representation_type=WGS84GeodeticRepresentation)
    )

    loc_j2000 = get_sat_j2000(utc.value, orbit_file)
    loc = GCRS(
        x=loc_j2000[:, 0] * u.m,
        y=loc_j2000[:, 1] * u.m,
        z=loc_j2000[:, 2] * u.m,
        representation_type='cartesian',
        obstime=utc
    ).transform_to(
        ITRS(obstime=utc, representation_type=WGS84GeodeticRepresentation)
    )

    src_x, src_y, src_z = src.x.value, src.y.value, src.z.value
    loc_x, loc_y, loc_z = loc.x.value, loc.y.value, loc.z.value # unit: m
    a = src_x**2 + src_y**2 + z_factor * src_z**2
    b = 2*(loc_x*src_x + loc_y*src_y + z_factor*loc_z*src_z)
    c = loc_x**2 + loc_y**2 + z_factor*loc_z**2 - R_earth.value**2
    hmin = np.sqrt(c + R_earth.value**2 - b*b/4.0/a) - R_earth.value
    mask = (a*b > 0.0)
    hmin[mask] = np.linalg.norm(
        [loc_x[mask], loc_y[mask], loc_z[mask]/(1 - f)], axis=0
    ) - R_earth.value

    return hmin / 1000.0


def calc_transmis_coeff(
    src_j2000: ArrayLike,
    utc: ArrayLike,
    energy: ArrayLike,
    orbit_file: str,
    step_size: Optional[float] = 0.5,
    name: Optional[str] = None
) -> NDArray:
    src = np.asarray(src_j2000)
    energy = np.asarray(energy, dtype=np.float64) * 1000  # keV to eV

    loc = get_sat_j2000(utc, orbit_file)

    cd = column_density(src, loc, utc, step_size=step_size, name=name) # (t, 5)
    element = (1, 2, 7, 8, 18)
    cs = np.column_stack(
        [calculate_cross_section(z, energy)['total'] * 1e-24 for z in element]
    )  # (e, 5)

    trans_coeff = np.exp(-np.einsum('tE,eE->te', cd ,cs, optimize='optimal'))

    return trans_coeff
