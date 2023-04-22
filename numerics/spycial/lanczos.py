"""Original code taken from Boost, which is:

(C) Copyright John Maddock 2006.
Use, modification and distribution are subject to the
Boost Software License, Version 1.0. (See accompanying file
LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

Optimal values for G for each N are taken from
http://web.mala.bc.ca/pughg/phdThesis/phdThesis.pdf,
as are the theoretical error bounds.

Constants calculated using the method described by Godfrey
http://my.fit.edu/~gabdo/gamma.txt and elaborated by Toth at
http://www.rskey.org/gamma.htm using NTL::RR at 1000 bit precision.

"""
import numpy as np
from numba import njit

from .evalpoly import _devalrational

_lanczos_num = np.array([
    2.506628274631000270164908177133837338626,
    210.8242777515793458725097339207133627117,
    8071.672002365816210638002902272250613822,
    186056.2653952234950402949897160456992822,
    2876370.628935372441225409051620849613599,
    31426415.58540019438061423162831820536287,
    248874557.8620541565114603864132294232163,
    1439720407.311721673663223072794912393972,
    6039542586.35202800506429164430729792107,
    17921034426.03720969991975575445893111267,
    35711959237.35566804944018545154716670596,
    42919803642.64909876895789904700198885093,
    23531376880.41075968857200767445163675473
])

_lanczos_denom = np.array([
    1.0,
    66.0,
    1925.0,
    32670.0,
    357423.0,
    2637558.0,
    13339535.0,
    45995730.0,
    105258076.0,
    150917976.0,
    120543840.0,
    39916800.0,
    0.0
])

_lanczos_sum_expg_scaled_num = np.array([
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859
])

_lanczos_sum_expg_scaled_denom = np.array([
    1.0,
    66.0,
    1925.0,
    32670.0,
    357423.0,
    2637558.0,
    13339535.0,
    45995730.0,
    105258076.0,
    150917976.0,
    120543840.0,
    39916800.0,
    0.0
])

_lanczos_g = 6.024680040776729583740234375


@njit('float64(float64)')
def _lanczos_sum(x):
    return _devalrational(_lanczos_num, _lanczos_denom, x)


@njit('float64(float64)')
def _lanczos_sum_expg_scaled(x):
    return _devalrational(_lanczos_sum_expg_scaled_num,
                          _lanczos_sum_expg_scaled_denom, x)
