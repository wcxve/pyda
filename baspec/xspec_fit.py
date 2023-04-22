# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 19:45:39 2023

@author: xuewc
"""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from astropy.io import fits

from xspec import AllData, AllModels, Fit, AllChains, Model, Chain

import os
os.chdir('/Users/xuewc/BurstData/FRB221014/HXMT/')


AllChains.clear()
AllData.clear()
AllModels.clear()

AllData('1:1 LE_bmin5.grp 1:2 ME_bmin5.grp 1:3 HE_bmin5.grp')

AllData.ignore('1:**-1.0 11.0-** 2:**-8.0 35.0-** 3:**-18.0 250.0-**')

# m = Model('cutoffpl+bb')
# m.bbody.kT.prior          = 'cons'
# m.bbody.norm.prior        = 'jeffreys'
# m.cutoffpl.PhoIndex.prior = 'cons'
# m.cutoffpl.HighECut.prior = 'cons'
# m.cutoffpl.norm.prior     = 'jeffreys'
# m = Model('cutoffpl')
# m.cutoffpl.PhoIndex.prior = 'cons'
# m.cutoffpl.HighECut.prior = 'cons'
# m.cutoffpl.norm.prior     = 'jeffreys'
m = Model('wabs*powerlaw')
m.wabs.nH=2.79
m.wabs.nH.frozen = True
m.powerlaw.PhoIndex.prior = 'cons'
m.powerlaw.norm.prior     = 'jeffreys'
Fit.method = 'leven 1000000 1e-10 1e-10'
Fit.statMethod = 'cstat'
Fit.perform()
AllModels.show()
Fit.show()

mexpr = m.expression.replace(' ', '')
overwrite = False
if not os.path.exists(f'{mexpr}.fits') or overwrite:
    if os.path.exists(f'{mexpr}.fits'):
        os.remove(f'{mexpr}.fits')
    Chain(f'{mexpr}.fits',
          burn=20000, runLength=50000, rand=False, algorithm='gw', walkers=10)

with fits.open(f'{mexpr}.fits') as hdul:
    pars = hdul[1].data

import corner
fig = corner.corner(
    data=np.column_stack([pars[p] for p in pars.names[:-1]]),
    labels=[i[:-3] for i in pars.names[:-1]],
    label_kwargs={'fontsize': 14},
    quantiles=[0.15865, 0.5, 0.84135],
    levels=[[0.683, 0.954, 0.997],[0.683, 0.90]][1],
    show_titles=True,
    title_fmt='.2f',
    color='blue',
    smooth=0.5,
    # range=((0,130),(-1.6,-2.7),(0,3.1e-7))[1:],
    truths=[m(i).values[0] for i in range(1, m.nParameters+1)],
    truth_color='red',
    max_n_ticks=5,
    title_kwargs={'fontsize': 14},
    use_math_text=True,
    labelpad=-0.05
)
