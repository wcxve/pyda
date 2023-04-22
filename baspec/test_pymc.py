# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 00:49:23 2023

@author: Wang-Chen Xue < https://orcid.org/0000-0001-8664-5085 >
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc_spec import data_for_wstat, Wabs, Powerlaw, WStat
import arviz as az
path = '/Users/xuewc/BurstData/FRB221014/HXMT/'
LE = data_for_wstat(1,
                    11,
                    f'{path}/LE_bmin5.grp',
                    f'{path}/LE_phabkg20s_g0_0-94.pha',
                    f'{path}/LE_rsp.rsp')
ME = data_for_wstat(8,
                    35,
                    f'{path}/ME_bmin5.grp',
                    f'{path}/ME_phabkg20s_g0_0-53.pha',
                    f'{path}/ME_rsp.rsp')
HE = data_for_wstat(18,
                    250,
                    f'{path}/HE_bmin5.grp',
                    f'{path}/HE_phabkg20s_g0_0-12.pha',
                    f'{path}/HE_rsp.rsp')
if __name__ == '__main__':

    with pm.Model() as model:
        nH = pt.constant(2.79)
        alpha = pm.Uniform('alpha', lower=1, upper=3)
        # Ecut = pm.Uniform('Ecut', lower=0, upper=3000)
        norm = pm.Uniform('norm', lower=0, upper=5)

        # for inst in [LE, ME, HE][-1:]:
        inst = HE
        wabs = Wabs(inst['ebins_ph'])(nH)
        # cpl = norm*wabs*CutoffPowerlaw(inst['ebins_ph'])(alpha, Ecut)
        pl = norm*wabs*Powerlaw(inst['ebins_ph'])(alpha)
        src = pl
        # pl = pm.math.exp(norm)*Powerlaw(inst['ebins_ph'])(alpha)
        WStat(src,
              inst['response'],
              inst['n_on'],
              inst['n_off'],
              inst['t_on'],
              inst['t_off'],
              inst['name'],
              inst['channel'])
        # p_map = pm.find_MAP(return_raw=True)
        idata = pm.sample(20000,
                          # tune=5000,
                          # nuts={'target_accept':0.9},
                          # init='advi+adapt_diag',
                          # nuts_sampler='numpyro',
                          idata_kwargs={'log_likelihood': True},
                          cores=4,
                          chains=4,
                          mp_ctx='forkserver')
        # idata.to_netcdf('PL.nc')
        # idata = az.from_netcdf('CPL.nc')
        az.plot_trace(idata, var_names=['alpha', 'norm'])
        # pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        # input('pause...')
        # az.plot_ppc(idata)
        # az.waic(idata, var_name='N_on')
