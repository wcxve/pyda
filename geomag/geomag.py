import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.io import fits
from astropy import coordinates as coord
from astropy.coordinates.representation import CartesianRepresentation, SphericalRepresentation

from pyda.geomag.ppigrf import igrf
from pyda.utils import utc_to_met, met_to_utc
from pyda.utils.coordinate import radec_to_cart, cart_to_sph

def geomag_angle(aux_file, posatt_file, utc0):
    # TODO: igrf should not be broadcast over coordinates
    # raise ValueError('B/C sat must be predefined !!! see Line 17 & 20')
    with fits.open(aux_file) as hdul:
        aux = hdul[1].data
    utc = Time(met_to_utc(aux['TIME'], 'GECAM-C'), scale='utc')
    idx = np.abs(((utc - Time(utc0, scale='utc')).sec)).argmin()
    aux = aux[idx:idx+1]
    utc = Time(met_to_utc(aux['TIME'], 'GECAM-C'))

    Be, Bn, Bu =  igrf(
        aux['LONLAT'][:,0], aux['LONLAT'][:,1], aux['ALT'], utc.datetime
    )
    print('position', aux['LONLAT'][:,0], aux['LONLAT'][:,1], aux['ALT'])

    decl = np.degrees(np.arctan2(Be, Bn))[0]
    incl = np.degrees(np.arctan2(Bu, np.hypot(Bn, Be)))[0]
    intensity = np.sqrt(Be * Be + Bn * Bn + Bu * Bu)[0]

    earth_loc = coord.EarthLocation(lon=aux['LONLAT'][:, 0], lat=aux['LONLAT'][:, 1], height=aux['ALT']*u.km)
    mag_altaz=coord.AltAz(az=decl*u.deg, alt=incl*u.deg, location=earth_loc, obstime=utc)
    mag_gcrs = mag_altaz.transform_to(coord.GCRS(obstime=utc))
    mag_radec = np.column_stack([mag_gcrs.ra.value, mag_gcrs.dec.value])
    from pyda.utils.coordinate import object_angle
    object_angle(mag_radec[0], aux['TIME'][0], -0.1, 0.1, posatt_file)
    print('mag_radec', mag_radec)
    mag_gcrs_xyz = radec_to_cart(mag_radec)
    # z transfromed from sat to payload
    z_axis_radec = np.column_stack([aux['Z_J2000'][:, 0]+180, -aux['Z_J2000'][:, 1]])
    z_axis_gcrs = radec_to_cart(z_axis_radec)
    x_axis_radec = np.column_stack([aux['X_J2000'][:, 0], aux['X_J2000'][:, 1]])
    x_axis_gcrs = radec_to_cart(x_axis_radec)
    y_axis_gcrs = np.cross(z_axis_gcrs, x_axis_gcrs)
    # the "-" is for the definition of incidence
    mags = np.column_stack([
        -np.sum(mag_gcrs_xyz*x_axis_gcrs, axis=1),
        -np.sum(mag_gcrs_xyz*y_axis_gcrs, axis=1),
        -np.sum(mag_gcrs_xyz*z_axis_gcrs, axis=1)
    ])
    theta, phi = cart_to_sph(mags).T
    mags = np.array([
        utc.value,
        theta,
        phi,
        intensity
    ]).T
    return mags

if __name__ == '__main__':
    # aux_file = '/Users/xuewc/Downloads/aux/gb_aux_210710_01_v01.fits'
    # mags = geomag_angle(aux_file, '2021-07-10T01:46:36.709997')
    aux_file = '/Users/xuewc/BurstData/busrt_candidate/gc_aux_230715_07_v00.fits'
    posatt_file = '/Users/xuewc/BurstData/busrt_candidate/gc_posatt_230715_07_v00.fits'
    mags = geomag_angle(aux_file, posatt_file, '2023-07-15T07:11:02.400')
    # aux_file = '/Users/xuewc/BurstData/busrt_candidate/gc_aux_230816_13_v00.fits'
    # posatt_file = '/Users/xuewc/BurstData/busrt_candidate/gc_posatt_230816_13_v00.fits'
    # mags = geomag_angle(aux_file, posatt_file, '2023-08-16T13:54:51.100')