# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:18:52 2023

@author: xuewc
"""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
from pytensor.tensor.math import gammau, log, sqrt, switch
import pytensor.tensor as pt
from astropy.io import fits

from xspec import AllData

import os
os.chdir('/Users/xuewc/BurstData/FRB221014/HXMT/')



def wstat(obs_counts, mod_rates, obs_exp, back_counts, back_exp):
    n_chans = len(obs_counts)
    stat = np.empty(n_chans)
    mr = switch(mod_rates>1.0e-5/obs_exp, mod_rates, 1.0e-5/obs_exp)
    expo_sum = obs_exp + back_exp
    a = expo_sum
    b = expo_sum*mr - obs_counts - back_counts
    c = -back_counts*mr
    d = sqrt(b*b - 4.0*a*c)
    f = switch(b >= 0.0, -2*c / (b + d), -(b - d) / (2*a))
    stat1 = obs_exp*mr - back_counts*log(back_exp/expo_sum)
    stat2 = obs_exp*mr + expo_sum*f - obs_counts*log(obs_exp*(mr+f)) \
            - back_counts*log(back_exp*f) - obs_counts*(1-log(obs_counts)) \
            - back_counts*(1-log(back_counts))
    stat = switch(obs_counts==0, stat1, stat2)

    # for ch in range(n_chans):
    #     si = obs_counts[ch]
    #     bi = back_counts[ch]
    #     tsi = obs_exp
    #     tbi = back_exp
    #     yi = mr[ch]

    #     ti = tsi + tbi
    #     if si == 0.0:
    #         stat[ch] = tsi*yi - bi*log(tbi/ti)
    #     else:
    #         a = ti
    #         b = ti*yi - si - bi
    #         c = -bi*yi
    #         d = sqrt(b*b - 4.0*a*c)
    #         f = -2*c / (b + d) if b >= 0.0 else -(b - d) / (2*a)
    #         stat[ch] = tsi*yi + ti*f - si*log(tsi*yi+tsi*f) \
    #                 - bi*log(tbi*f) - si*(1-log(si)) \
    #                 - bi*(1-log(bi))
    return stat

AllData.clear()
AllData('1:1 LE_bmin5.grp 2:2 ME_bmin5.grp 3:3 HE_bmin5.grp')

ign_str = ['**-1.0 11.0-**', '**-8.0 35.0-**', '**-18.0 250.0-**']

spec = []
exposure = []
spec_bkg = []
exposure_bkg = []
rsp = []
Eph = []
for i in range(3):
    s = AllData(i+1)
    s.ignore(ign_str[i])

    spec.append(np.round(np.array(s.values) * s.exposure))
    exposure.append(s.exposure)
    spec_bkg.append(np.round(np.array(s.background.values)*s.background.exposure))
    exposure_bkg.append(s.background.exposure)
    Eph.append(np.array(s.response.energies))

    with fits.open(s.fileName) as hdul:
        data = hdul['SPECTRUM'].data
        indices = np.where(data['GROUPING'] == 1)[0]
        # spec.append(np.add.reduceat(data['COUNTS'], indices)[s.noticed])
    with fits.open(s.response.rmf) as hdul:
        rsp_matrix = hdul['MATRIX'].data['MATRIX']
        rsp.append(np.add.reduceat(rsp_matrix, indices, axis=1)[:, s.noticed])

with pm.Model() as cpl:
    PhoIndex = pm.Uniform('PhoIndex', lower=-5, upper=5)
    HighECut = pm.Uniform('HighECut', lower=-5, upper=5)
    log_norm = pm.Uniform('log_norm', lower=-20, upper=20)

    a = 1 - PhoIndex
    x = [i/HighECut for i in Eph]
    integral = [-np.exp(log_norm) * HighECut**a * gammau(a, x) for e in Eph]
    model_flux = [i[1:]-i[:-1] for i in integral]
    model_rate = [fi@rspi for fi, rspi in zip(model_flux, rsp)]

    le_like = pm.Potential('le_like',
                           wstat(spec[0], model_rate[0], exposure[0],
                                 spec_bkg[0], exposure_bkg[0]))
    trace = pm.sample()
