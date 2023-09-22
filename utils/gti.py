import numpy as np
import numba as nb

from astropy.coordinates import EarthLocation
from astropy.units import m


@nb.njit('int64(float64[::1], float64[:, ::1])')
def is_inside(point, polygon):
    # adapt from https://stackoverflow.com/a/66189882
    x = point[0]
    y = point[1]
    length = len(polygon) - 1
    dy2 = y - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii<length:
        dy = dy2
        dy2 = y - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (x >= polygon[ii][0] or x >= polygon[jj][0]):

            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if x > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif x == F: # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2==0 and (x==polygon[jj][0] or (dy==0 and (x-polygon[ii][0])*(x-polygon[jj][0])<=0)):
                return 2

        ii = jj
        jj += 1

    return intersections & 1


@nb.njit('boolean[::1](float64[:, ::1], float64[:, ::1])', parallel=True)
def is_inside_parallel(points, polygon):
    # adapt from https://stackoverflow.com/a/66189882
    n = len(points)
    flag = np.empty(n, dtype=np.bool_)
    for i in nb.prange(n):
        flag[i] = is_inside(points[i], polygon)
    return flag


def saa_boundary(sat):
    sat = str(sat).upper()
    if sat == 'FERMI':
        return np.array([
            [ 33.9  , -30.   ],
            [ 12.398, -19.867],
            [ -9.103,  -9.733],
            [-30.605,   0.4  ],
            [-38.4  ,   2.   ],
            [-45.   ,   2.   ],
            [-65.   ,  -1.   ],
            [-84.   ,  -6.155],
            [-89.2  ,  -8.88 ],
            [-94.3  , -14.22 ],
            [-94.3  , -18.404],
            [-86.1  , -30.   ],
            [ 33.9  , -30.   ]
        ])

    elif sat == 'GECAM':
        return np.array([
            (-90, -20),
            (-79,  -5),
            (-60,  -1),
            (-30,   0),
            (  0, -10),
            ( 30, -23),
            ( 30, -30),
            (-90, -30),
            (-90, -20),
        ]).astype(float)

    elif sat == 'HXMT':
        return np.array([
            (-74.3, -45.0),
            (-88.2, -28.0),
            (-96.0, -13.0),
            (-92.0,  -9.0),
            (-70.0,  -2.5),
            (-45.0,   3.0),
            (-33.0,  -2.1),
            (-15.0, -15.0),
            (  0.8, -18.8),
            (-18.2, -23.0),
            ( 31.0, -31.0),
            ( 27.3, -39.0),
            ( 22.0, -45.0),
            (-74.3, -45.0)
        ])
    else:
        raise ValueError('only Fermi, GECAM, and HXMT supported')


def wgs84_to_lonlat(xyz):
    loc = EarthLocation(x=xyz[:, 0]*m, y=xyz[:, 1]*m, z=xyz[:, 2]*m)
    return np.column_stack([loc.geodetic.lon.value, loc.geodetic.lat.value])


def is_in_saa(loc_wgs84, sat):
    lonlat = wgs84_to_lonlat(loc_wgs84)
    saa = saa_boundary(sat)
    return is_inside_parallel(lonlat, saa)


def is_earth_occulted(ra, dec, loc_j2000):
    ra = np.radians(ra)
    dec = np.radians(dec)
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)
    sin_ra = np.sin(ra)
    cos_ra = np.cos(ra)
    src_j2000 = np.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec])

    r_earth = 6371393.0
    r = np.linalg.norm(loc_j2000, axis=1)
    earth_span = np.arcsin(r_earth / r)
    earth_direction = - loc_j2000 / r[:, None]

    angle = np.arccos(earth_direction @ src_j2000)

    return angle <= earth_span


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    points = np.random.uniform([-180, -90],[180, 90], (86400, 2))
    for i in ['gecam', 'fermi', 'hxmt']:
        polygon = saa_boundary(i)
        plt.figure()
        mask = is_inside_parallel(points, polygon)
        plt.scatter(*points[~mask].T, s=1)
        plt.scatter(*points[mask].T, s=1, c='r')
        plt.plot(*polygon.T, c='k')
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)

    from astropy.io import fits
    ra = 288.263
    dec = 19.803

    posatt_file = '/Users/xuewc/BurstData/GRB221009A/HEBS_Occultation/gc_posatt_221009_13_v06.fits'
    with fits.open(posatt_file) as hdul:
        posatt = hdul['Orbit_Attitude'].data

    loc_j2000 = np.column_stack([posatt['X_J2000'], posatt['Y_J2000'], posatt['Z_J2000']])
    loc_wgs84 = np.column_stack([posatt['X_WGS84'], posatt['Y_WGS84'], posatt['Z_WGS84']])
    mask1 = ~is_earth_occulted(ra, dec, loc_j2000)
    mask2 = ~is_in_saa(loc_wgs84, 'gecam')
    plt.figure()
    plt.scatter(*wgs84_to_lonlat(loc_wgs84)[mask2].T, c='k', s=2)
    plt.scatter(*wgs84_to_lonlat(loc_wgs84)[~mask2].T, c='r', s=2)
    plt.scatter(*wgs84_to_lonlat(loc_wgs84)[~mask1].T, c='b', s=2)
    plt.plot(*saa_boundary('gecam').T, 'r--')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
