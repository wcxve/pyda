import astropy.units as u
import numpy as np

from astropy.coordinates import get_body, SkyCoord, GCRS, ITRS, \
                                WGS84GeodeticRepresentation
from astropy.constants import R_earth
from astropy.io import fits
from astropy.time import Time

from pyda.occult.msis.density import column_density
from pyda.occult.xcom import calculate_cross_section
from pyda.utils.coordinate import radec_to_cart
from pyda.utils.time import met_to_utc
from pyda.utils.misc import get_sat_j2000, telescope_to_sat, ORBIT_CONFIG

__all__ = ['calc_tangent_height', 'calc_transmis_coeff', 'get_oti']


def calc_tangent_height(src_j2000, utc, file):
    """
    Calculate tangent height of the line of sight.

    Parameters
    ----------
    src_j2000 : array_like
        J2000 coordinate (R.A., Dec.) of source.
    utc : array_like
        Dates and times, in ISO-8601 format.
    file : str
        Path of HXMT's orbit, GECAM's posatt or Fermi's poshist file.

    Returns
    -------
    height : array
        Tangent height array.

    """
    f = 1/298.257
    z_factor = (1 - f)**(-2)

    utc = Time(np.atleast_1d(utc), scale='utc')

    src_j2000 = np.array(src_j2000, dtype=np.float64, order='C')
    src = GCRS(
        ra=src_j2000[0].repeat(utc.size) * u.deg,
        dec=src_j2000[1].repeat(utc.size) * u.deg,
        obstime=utc
    ).transform_to(
        ITRS(obstime=utc, representation_type=WGS84GeodeticRepresentation)
    )

    loc_j2000 = get_sat_j2000(utc.value, file)
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
    Rearth = R_earth.value
    Rearth2 = Rearth*Rearth
    a = src_x*src_x + src_y*src_y + z_factor * src_z*src_z
    b = 2*(loc_x*src_x + loc_y*src_y + z_factor*loc_z*src_z)
    c = loc_x*loc_x + loc_y*loc_y + z_factor*loc_z*loc_z - Rearth2
    hmin = np.sqrt(c + Rearth2 - b*b/4.0/a) - Rearth
    mask = (a*b > 0.0)
    hmin[mask] = np.linalg.norm(
        [loc_x[mask], loc_y[mask], loc_z[mask]/(1 - f)], axis=0
    ) - Rearth

    return hmin / 1000.0


def calc_transmis_coeff(
    src_j2000,
    utc,
    energy,
    orbit_file,
    step_size=0.5,
    name=None
):
    src = np.array(src_j2000, dtype=np.float64, order='C')

    energy = np.array(energy, dtype=np.float64, order='C')
    energy *= 1000.0  # keV to eV

    loc = get_sat_j2000(utc, orbit_file)

    cd = column_density(src, loc, utc, step_size=step_size, name=name) # (t, 5)
    element = (1, 2, 7, 8, 18)
    cs = np.column_stack(
        [calculate_cross_section(z, energy)['total'] * 1e-24 for z in element]
    )  # (e, 5)

    trans_coeff = np.exp(-np.einsum('tE,eE->te', cd ,cs, optimize='optimal'))

    return trans_coeff


def get_oti(obj, file, alt_range):
    r"""
    Get Occultation Time Intervals given the range of line-of-sight altitude.

    Parameters
    ----------
    obj : str, or array_like of shape (2,)
        Celestial object name, or J2000 coordinate (R.A., Dec.).
    file : str
        Path of HXMT's orbit, GECAM's posatt or Fermi's poshist file.
    alt_range : array_like of shape (2,)
        Range of line-of-sight altitude, in unit of km.

    Returns
    -------
    oti : ndarray of shape (n, 2)
        Occultation Time Intervals.

    """
    with fits.open(file) as hdu_list:
        telescope = hdu_list['PRIMARY'].header['TELESCOP']
        sat = telescope_to_sat(telescope)
        orbit_ext, t, *_ = ORBIT_CONFIG[sat]
        met = hdu_list[orbit_ext].data[t]

    utc = met_to_utc(met, sat)

    if type(obj) in [list, tuple] and len(obj) == 2:
        src_j2000 = radec_to_cart(obj)
    elif type(obj) == str and obj.lower() in (
            'sun', 'moon', 'mercury', 'venus', 'earth-moon-barycenter', 'mars',
            'jupiter', 'saturn', 'uranus', 'neptune'
    ):
        src = get_body(obj, met_to_utc(met, sat, True))  # in GCRS frame
        src_j2000 = src.cartesian.xyz.value.T
    elif type(obj) == str:
        src = SkyCoord.from_name(obj, frame='gcrs')
        src_j2000 = src.cartesian.xyz.value.T
    else:
        raise ValueError(f'wrong input {obj=}')

    h = calc_tangent_height(src_j2000, utc, file)

    hmask = (alt_range[0] <= h) & (h <= alt_range[1])

    met = met[hmask]
    h = h[hmask]

    diff = met[1:] - met[:-1]
    idx = np.flatnonzero(diff >= 2.0)
    idx = np.sort([0, *idx, *(idx+1), len(met) - 1]).reshape(-1, 2)

    return met[idx], h[idx]