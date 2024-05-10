# -*- coding: utf-8 -*-
"""
@author: xuewc<xuewc@ihep.ac.cn>
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from numpy.typing import ArrayLike, NDArray
from pymsis import msis00f, msis20f, msis21f  # type: ignore


def element_density(
    date: str,
    lons: ArrayLike,
    lats: ArrayLike,
    alts: ArrayLike,
    f107: float,
    f107a: float,
    ap: ArrayLike,
    summation: Optional[bool]=True,
    options: Optional[List[float]]=None,
    version: Optional[Union[float, str]]=2.1,
    **kwargs: dict,
) -> NDArray:
    """
    Get MSIS profile given the input information.

    Parameters
    ----------
    date : str
        Date and time of interest, in ISO-8601 format.
    lons : array_like
        Longitudes of interest (from 0 to 360, or from -180 to 180), in unit
        degree.
    lats : array_like
        Latitudes of interest (from -90 to 90), in unit degree.
    alts : array_like
        Altitudes of interest, in unit km.
    f107 : float
        Daily F10.7 of the previous day for the given date.
    f107a : float
        F10.7 running 81-day average centered on the given date.
    ap : array_like
        | Ap for the given date, 1-6 only used if ``geomagnetic_activity=-1``.
        | [0] Daily Ap
        | [1] 3 hr ap index for current time
        | [2] 3 hr ap index for 3 hrs before current time
        | [3] 3 hr ap index for 6 hrs before current time
        | [4] 3 hr ap index for 9 hrs before current time
        | [5] Average of eight 3 hr ap indices from 12 to 33 hrs
        |     prior to current time
        | [6] Average of eight 3 hr ap indices from 36 to 57 hrs
        |     prior to current time
        |
    summation : bool, optional
        Wether to return the sum of density for all given locations. This is
        for calculating column density. Remember to multiply the step length!
    options : array_like[25, float], optional
        A list of options (switches) to the model, if options is passed
        all keyword arguments specifying individual options will be ignored.
    version : float or str, default: 2.1
        MSIS version number, one of (0, 2.0, 2.1).
    **kwargs : dict
        Single options for the switches can be defined through keyword
        arguments.

    Returns
    -------
    d : ndarray (nlocs, 5)
        | The atmospheric density:
        | [0] H  # density (m^-3),
        | [1] He # density (m^-3),
        | [2] N  # density (m^-3),
        | [3] O  # density (m^-3),
        | [4] Ar # density (m^-3),

    Other Parameters
    ----------------
    f107 : float
        Account for F10.7 variations
    time_independent : float
        Account for time variations
    symmetrical_annual : float
        Account for symmetrical annual variations
    symmetrical_semiannual : float
        Account for symmetrical semiannual variations
    asymmetrical_annual : float
        Account for asymmetrical annual variations
    asymmetrical_semiannual : float
        Account for asymmetrical semiannual variations
    diurnal : float
        Account for diurnal variations
    semidiurnal : float
        Account for semidiurnal variations
    geomagnetic_activity : float
        Account for geomagnetic activity
        (1 = Daily Ap mode, -1 = Storm-time Ap mode)
    all_ut_effects : float
        Account for all UT/longitudinal effects
    longitudinal : float
        Account for longitudinal effects
    mixed_ut_long : float
        Account for UT and mixed UT/longitudinal effects
    mixed_ap_ut_long : float
        Account for mixed Ap, UT, and longitudinal effects
    terdiurnal : float
        Account for terdiurnal variations

    Notes
    -----
    1. The 10.7 cm radio flux is at the Sun-Earth distance,
       not the radio flux at 1 AU.
    2. aps[1:] are only used when ``geomagnetic_activity=-1``.

    """
    num_options = 25
    if options is None:
        options = create_options(**kwargs)  # type: ignore
    elif len(options) != num_options:
        raise ValueError(f"options needs to be a list of length {num_options}")

    inputs = create_input(date, lons, lats, alts, f107, f107a, ap)

    if np.any(~np.isfinite(inputs)):
        raise ValueError(
            "Input data has non-finite values, all input data must be valid."
        )

    # convert to string version
    version = str(version)
    if version in {"0", "00"}:
        msis00f.pytselec(options)
        output = msis00f.pygtd7d(
            inputs[:, 0],
            inputs[:, 1],
            inputs[:, 2],
            inputs[:, 3],
            inputs[:, 4],
            inputs[:, 5],
            inputs[:, 6],
            inputs[:, 7:],
        )

    elif version.startswith("2"):
        # We need to point to the MSIS parameter file that was installed with
        # the Python package
        msis_path = str(Path(msis21f.__file__).resolve().parent) + "/"

        # Select the proper library. Default to version 2.1, unless explicitly
        # requested "2.0" via string
        msis_lib = msis21f
        if version == "2.0":
            msis_lib = msis20f
        msis_lib.pyinitswitch(options, parmpath=msis_path)
        output = msis_lib.pymsiscalc(
            inputs[:, 0],
            inputs[:, 1],
            inputs[:, 2],
            inputs[:, 3],
            inputs[:, 4],
            inputs[:, 5],
            inputs[:, 6],
            inputs[:, 7:],
        )

    else:
        raise ValueError(
            f"The MSIS version selected: {version} is not "
            "one of the valid version numbers: (0, 2, 2.1)"
        )

    # The Fortran code puts 9.9e-38 in as NaN
    # Have to make sure this doesn't overlap 0 due to really small values
    # so atol should be less than the comparison value
    output[np.isclose(output, 9.9e-38, atol=1e-38)] = 0.0
    # Note: NaN usually occurs at low altitude for low density air components,
    # Note: it is reasonale to keep this small value in occultation analysis

    d = np.column_stack((
        # H
        output[:, 5],
        # He
        output[:, 4],
        # N2, N, and N in NO for MSIS 2.1
        output[:, 1]*2 + output[:, 7] + output[:, 9],
        # O2, O, Anomalous O, and O in NO for MSIS 2.1
        output[:, 2]*2 + output[:, 3] + output[:, 8] + output[:, 9],
        # Ar
        output[:, 6]
    ))

    if summation:
        d = d.sum(0)

    return d


def create_options(
    f107: float=1,
    time_independent: float=1,
    symmetrical_annual: float=1,
    symmetrical_semiannual: float=1,
    asymmetrical_annual: float=1,
    asymmetrical_semiannual: float=1,
    diurnal: float=1,
    semidiurnal: float=1,
    geomagnetic_activity: float=1,
    all_ut_effects: float=1,
    longitudinal: float=1,
    mixed_ut_long: float=1,
    mixed_ap_ut_long: float=1,
    terdiurnal: float=1,
) -> List[float]:
    """
    Create the options list based on keyword argument choices.
    Defaults to all 1's for the input options.

    Parameters
    ----------
    f107 : float
        Account for F10.7 variations
    time_independent : float
        Account for time variations
    symmetrical_annual : float
        Account for symmetrical annual variations
    symmetrical_semiannual : float
        Account for symmetrical semiannual variations
    asymmetrical_annual : float
        Account for asymmetrical annual variations
    asymmetrical_semiannual : float
        Account for asymmetrical semiannual variations
    diurnal : float
        Account for diurnal variations
    semidiurnal : float
        Account for semidiurnal variations
    geomagnetic_activity : float
        Account for geomagnetic activity
        (1 = Daily Ap mode, -1 = Storm-time Ap mode)
    all_ut_effects : float
        Account for all UT/longitudinal effects
    longitudinal : float
        Account for longitudinal effects
    mixed_ut_long : float
        Account for UT and mixed UT/longitudinal effects
    mixed_ap_ut_long : float
        Account for mixed Ap, UT, and longitudinal effects
    terdiurnal : float
        Account for terdiurnal variations

    Returns
    -------
    options : list
        25 options as a list ready for msis2 input

    """
    options = [
        f107,
        time_independent,
        symmetrical_annual,
        symmetrical_semiannual,
        asymmetrical_annual,
        asymmetrical_semiannual,
        diurnal,
        semidiurnal,
        geomagnetic_activity,
        all_ut_effects,
        longitudinal,
        mixed_ut_long,
        mixed_ap_ut_long,
        terdiurnal,
    ] + [1] * 11
    return options


def create_input(
    date: str,
    lons: ArrayLike,
    lats: ArrayLike,
    alts: ArrayLike,
    f107: float,
    f107a: float,
    ap: ArrayLike,
) -> NDArray:
    """
    Combine all input values into a single flattened array.
    Parameters

    ----------
    date : str
        Date and time of interest, in ISO-8601 format.
    lons : array_like
        Longitudes of interest (from 0 to 360, or from -180 to 180), in unit
        degree.
    lats : array_like
        Latitudes of interest (from -90 to 90), in unit degree.
    alts : array_like
        Altitudes of interest, in unit km.
    f107 : float
        Daily F10.7 of the previous day for the given date.
    f107a : float
        F10.7 running 81-day average centered on the given date.
    ap : array_like
        | Ap for the given date, 1-6 only used if ``geomagnetic_activity=-1``
        | [0] Daily Ap
        | [1] 3 hr ap index for current time
        | [2] 3 hr ap index for 3 hrs before current time
        | [3] 3 hr ap index for 6 hrs before current time
        | [4] 3 hr ap index for 9 hrs before current time
        | [5] Average of eight 3 hr ap indices from 12 to 33 hrs
        |     prior to current time
        | [6] Average of eight 3 hr ap indices from 36 to 57 hrs
        |     prior to current time
        |

    Returns
    -------
    inputs : ndarray
        The shape of the data is (nlocs, 14)

    """
    # Turn everything into arrays
    date = np.atleast_1d(np.array(date, dtype=np.datetime64))
    dyear = (
        date.astype("datetime64[D]") - date.astype("datetime64[Y]")
    ).astype(float) + 1  # DOY 1-366
    dseconds = (
        date.astype("datetime64[s]") - date.astype("datetime64[D]")
    ).astype(float)

    lons = np.atleast_1d(lons)
    # If any longitudes were input as negatives, try to change them to
    # the (0, 360) range
    lons[lons < 0] += 360
    lats = np.atleast_1d(lats)
    alts = np.atleast_1d(alts)

    f107 = np.atleast_1d(f107)
    f107a = np.atleast_1d(f107a)
    ap = np.atleast_2d(ap)

    nlons = len(lons)
    nlats = len(lats)
    nalts = len(alts)

    if ap.shape[1] != 7:
        raise ValueError(f'The length of ap ({ap.shape[1]}) must be 7')

    if not (nlons == nlats == nalts):
        raise ValueError(
            f'The length of lons ({nlons}), lats ({nlats}) and alts ({nalts}) '
            'must all be equal'
        )

    dyear = dyear.repeat(nlons)
    dseconds = dseconds.repeat(nlons)
    f107s = f107.repeat(nlons)
    f107as = f107a.repeat(nlons)
    aps = ap.repeat(nlons, 0)

    # This means the data came in preflattened, from a satellite
    # trajectory for example, where we don't want to make a grid
    # out of the input data, we just want to stack it together.
    arr = np.stack([dyear, dseconds, lons, lats, alts, f107s, f107as], -1)

    # ap has 7 components, so we need to concatenate it onto the
    # arrays rather than stack
    inputs = np.concatenate([arr, aps], axis=1, dtype=np.float32)

    return inputs
