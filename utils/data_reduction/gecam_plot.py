# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:58:10 2023

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

from .gecam_evt import tehist_gecam, tehist_gecam, ehist_gecam
from ..time import met_to_utc, utc_to_met


DET = {
    'GECAM-A': [
        21, 12, 22, 14, 23,
        11,  4, 13,  5, 15,
        20,  3,  1,  6, 24,
        10,  2,  8,  7, 16,
        19,  9, 18, 17, 25
    ],
    'GECAM-C': list(range(1, 13))
}
DET['GECAM-B'] = DET['GECAM-A']

DAQ = {
    'GECAM-A': [
        5, 3, 4, 1, 3,
        1, 4, 2, 5, 4,
        2, 3, 1, 2, 5,
        5, 2, 5, 3, 1,
        4, 1, 3, 4, 2
    ],
    'GECAM-C': [
        1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2
    ]
}
DAQ['GECAM-B'] = DAQ['GECAM-A']


def _create_figure(sat):


def plot_ehist(evt_file, t0, tstart, tstop):
    with fits.open(evt_file) as hdul:
        telescope = hdul['PRIMARY'].header['TELESCOP']
        sat = 'GECAM-B' if telescope != 'HEBS' else 'GECAM-C'

    ehist_gecam





























