# -*- coding: utf-8 -*-
"""
@author: xuewc<xuewc@ihep.ac.cn>
"""

import multiprocessing as mp
import platform
import time
from sys import stdout
from typing import Optional

import astropy.units as u
import numpy as np
from numpy.typing import ArrayLike, NDArray
from astropy.coordinates import GCRS, ITRS, WGS84GeodeticRepresentation
from astropy.constants.iau2015 import R_earth
from astropy.time import Time
from pymsis.utils import get_f107_ap
from tqdm import tqdm

from .msis import element_density


def column_density(
    src_j2000: ArrayLike,
    loc_j2000: ArrayLike,
    utc: ArrayLike,
    step_size: Optional[float] = 0.5,
    lower_alt: Optional[float] = 5.0,
    upper_alt: Optional[float] = 550.0,
    f: Optional[float] = 1/298.257,
    num: Optional[bool] = True,
    summation: Optional[bool] = True,
    name: Optional[str] = '',
    **kwargs: dict
) -> NDArray:
    """
    Compute column density of atmosphere between a celestial source and given
    locations at given times.

    Parameters
    ----------
    src_j2000 : array_like
        J2000 coordinate (R.A., Dec.) of source.
    loc_j2000 : array_like
        Array of J2000 coordinate (X, Y, Z), in unit m.
    utc : array_like
        Array of time string in UTC format.
    step_size : float, optional
        The sampling step size (in unit km). The default is 0.5.
    lower_alt : float, optional
        Below `lower_alt` (in unit km), the air density is assume to be large
        enough so that no X-ray photons will pass through. The default is 5.
    upper_alt : float, optional
        Above `upper_alt` (in unit km), the air density is assume to be small
        enough so that all X-ray photons will pass through. The default is 500.
    f : float, optional
        The flattening factor of Earth. The default is 1/298.257.
    num : bool, optional
        Return column density in unit atom/cm^2 (True) or g/cm^2 (False). The
        default is False.
    summation : bool, optional
        Return column density (True) or column profile (False).
    name : str, optional
        `name` must be given as `__name__` when call on Windows.
    **kwargs : dict, optional
        Kwargs for `element_density`. `f107`, `f107a`, `ap`, `options` and
        `version` are available. Note that `f107`, `f107a` and `ap` must be
        given together, otherwise they will be ignored.

    Returns
    -------
    col_atoms : ndarray
        The column density of H, He, N, O and Ar atoms, in shape (ntimes, 5).
    path_loc : ndarray
        Sampling location, only returned when ``summation=False``.

    """
    if name != '__main__' and platform.system() == 'Windows':
        raise RuntimeError(
            "When call on Windows, function `column_density` or codes using it"
            " must be called inside ``if __name__ == '__main__':``, and the "
            "keyword argument `name` must be given as `__name__`"
        )
    utc = Time(np.atleast_1d(utc), format='isot', scale='utc')
    loc_j2000 = np.atleast_2d(loc_j2000)
    step_size = step_size*1000 # unit: m
    lower_alt = lower_alt*1000 + R_earth.value # unit: m
    upper_alt = upper_alt*1000 + R_earth.value # unit: m

    src_j2000 = np.asarray(src_j2000)
    src = GCRS(
        ra=src_j2000[0].repeat(utc.size) * u.deg,
        dec=src_j2000[1].repeat(utc.size) * u.deg,
        obstime=utc
    ).transform_to(
        ITRS(obstime=utc, representation_type=WGS84GeodeticRepresentation)
    )

    loc_j2000 = np.atleast_2d(loc_j2000)
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

    # note that this is not exact the start and end point, but a good enough
    # approximation based on Harmon et al. (2002), and hence `lower_alt` and
    # `upper_alt` should be small and large enough, repectively.
    z_factor = (1-f)**(-2)
    a = src_x**2 + src_y**2 + z_factor*src_z**2
    b = 2*(loc_x*src_x + loc_y*src_y + z_factor*loc_z*src_z)
    c_lower = loc_x**2 + loc_y**2 + z_factor*loc_z**2 - lower_alt**2
    c_upper = loc_x**2 + loc_y**2 + z_factor*loc_z**2 - upper_alt**2
    d_lower = b**2 - 4*a*c_lower
    d_upper = b**2 - 4*a*c_upper

    lmask = d_lower < 0
    umask = d_upper > 0
    mask = (lmask) & (umask)

    path_start = (-b[mask] - np.sqrt(d_upper[mask]))/(2*a[mask])
    path_end = (-b[mask] + np.sqrt(d_upper[mask]))/(2*a[mask])
    length = [
        np.arange(start, end + step_size, step_size)
        for start, end in zip(path_start, path_end)
    ]

    path = [
        ITRS(
            loc[mask][i].cartesian + src[mask][i].cartesian * length[i] * u.m,
            obstime=utc[mask][i],
            representation_type=WGS84GeodeticRepresentation
        ).earth_location
        for i in range(utc[mask].size)
    ]

    if not ('f107' in kwargs and 'f107a' in kwargs and 'ap' in kwargs):
        f107, f107a, ap = get_f107_ap(utc.datetime)
        kwargs.pop('f107', None)
        kwargs.pop('f107a', None)
        kwargs.pop('aps', None)
    else:
        f107 = np.atleast_1d(kwargs.pop('f107'))
        f107a = np.atleast_1d(kwargs.pop('f107a'))
        ap = np.atleast_2d(kwargs.pop('ap'))
        if not (len(f107) == len(f107a) == len(ap) == len(utc)):
            raise ValueError(
                f'The length of time ({len(utc)}), f107 ({len(f107)}), f107a '
                f'({len(f107a)}), and ap ({len(ap)}) must all be equal'
            )

    print('Start computation of column density...')
    t0 = time.time()
    cpus = mp.cpu_count() - 1 or 1
    with mp.Pool(cpus) as pool:
        results = [
            pool.apply_async(
                func=element_density,
                args=(
                    utc[mask][i].value,
                    path[i].lon.value,
                    path[i].lat.value,
                    path[i].height.to(u.km).value,
                    f107[mask][i],
                    f107a[mask][i],
                    ap[mask][i],
                    summation
                ),
                kwds=kwargs
            )
            for i in range(utc[mask].size)
        ]

        results = [
            r.get()
            for r in tqdm(results, total=len(results), file=stdout)
        ]


    if summation:
        # shape = (t, elements)
        density = np.vstack(results)

        # column density of H, He, N, O and Ar, in shape (t, , 5)
        col_atoms = np.empty((utc.size, 5), dtype=np.float32)
        col_atoms[~lmask] = np.inf
        col_atoms[~umask] = 0

        # atom/m^3 -> atom/m^2 -> atom/cm^2
        col_atoms[mask] = density * step_size/10000
    else:
        # shape = (t, locs, elements)
        density = _to_density_array(results)

        # column profile of H, He, N, O and Ar, in shape (t, locs, 5)
        col_atoms = np.zeros((utc.size, density.shape[1], 5), dtype=np.float32)
        col_atoms[~lmask] = np.inf
        col_atoms[~umask] = 0.0

        # atom/m^3 -> atom/m^2 -> atom/cm^2
        col_atoms[mask] = density * step_size/10000

        path_loc = np.empty((utc.size, density.shape[1], 3), dtype=np.float32)
        path_loc[~lmask] = [
            -181., -91., (lower_alt - R_earth.value) / 1000.0
        ]
        path_loc[~umask] = [
            -181., -91., (upper_alt - R_earth.value) / 1000.0
        ]
        path_loc[mask] = _to_loc_array([
            np.column_stack(
                (i.lon.value, i.lat.value, i.height.to(u.km).value)
            ) for i in path
        ])

    if not num: # atom/cm^2 to g/cm^2
        atom_mass = np.array([1.00794, 4.002602, 14.0067, 15.9994, 39.948])
        col_atoms *= atom_mass * 1.6605390666e-24

    t = time.time() - t0
    print(f'Computation ended with {cpus} CPU(s) used and {t:.2f} s cost')

    if summation:
        return col_atoms
    else:
        return col_atoms, path_loc


def _to_density_array(v):
    # adapted from https://stackoverflow.com/a/38619350
    lens = np.array([len(item) for item in v])
    mask = lens[:, None, None] > np.tile(np.arange(lens.max()), (5, 1)).T
    out = np.zeros(mask.shape, dtype=np.float32)
    out[mask] = np.concatenate(v, None)
    return out


def _to_loc_array(v):
    # adapted from https://stackoverflow.com/a/38619350
    lens = np.array([len(item) for item in v])
    mask = lens[:, None, None] > np.tile(np.arange(lens.max()), (3, 1)).T
    out = np.empty(mask.shape, dtype=np.float32)
    out[:, :, 0] = -181.0
    out[:, :, 1] = -91.0
    out[:, :, 2] = -1.0
    out[mask] = np.concatenate(v, None)
    return out
