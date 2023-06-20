# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:18:52 2023

@author: xuewc
"""

import arviz as az
import pymc as pm
import sys
sys.path.extend([
    '/Users/xuewc/Library/CloudStorage/OneDrive-个人/Documents/MyWork/DataAnalyzer/',
    '/Users/xuewc/heasoft-6.31.1/aarch64-apple-darwin22.3.0/lib/',
    '/Users/xuewc/heasoft-6.31.1/aarch64-apple-darwin22.3.0/lib/python/'
])
from baspec.scripts.pymc_spec import WStat, data_wstat
from baspec import Powerlaw, CutoffPowerlaw

if __name__ == '__main__':
    path = '/Users/xuewc/BurstData/GRB221009A/900-1100'
    N8 = data_wstat(erange=[(0, 20), (30, 40), (900, 1e8)],
                    spec_on=f'{path}/phase_n8_900-1100.pha',
                    spec_off=f'{path}/phase_n8_900-1100.bkg',
                    rspfile=f'{path}/n8_phase2.rsp',
                    name='N8',
                    is_ignore=True)
    B0 = data_wstat(erange=[(0,1000), (20000, 1e8)],
                    spec_on=f'{path}/phase_b0_900-1100.pha',
                    spec_off=f'{path}/phase_b0_900-1100.bkg',
                    rspfile=f'{path}/b0_phase2.rsp',
                    name='B0',
                    is_ignore=True)
    with pm.Model() as model:
        PhoIndex_CPL = pm.Uniform('PhoIndex_CPL', lower=0, upper=5)#pm.Flat('PhoIndex_CPL')
        HighECut = pm.Uniform('HighECut', lower=0, upper=1000)#pm.HalfFlat('HighECut')
        PhoIndex_PL = pm.Flat('PhoIndex_PL')
        norm_CPL = pm.HalfFlat('norm_CPL')
        norm_PL = pm.HalfFlat('norm_PL')
        factor_B0 = pm.HalfFlat('factor_B0')

        CPL_N8 = norm_CPL*CutoffPowerlaw(PhoIndex_CPL, HighECut)(N8['ebins_ph'])
        CPL_B0 = norm_CPL*CutoffPowerlaw(PhoIndex_CPL, HighECut)(B0['ebins_ph'])
        PL_N8 = norm_PL*Powerlaw(PhoIndex_PL)(N8['ebins_ph'])
        PL_B0 = norm_PL*Powerlaw(PhoIndex_PL)(B0['ebins_ph'])

        N8_wstat = WStat(PL_N8+CPL_N8, N8['response'], N8['Non'], N8['Noff'], N8['Ton'], N8['Toff'],
                           name='N8', channel=N8['channel'])
        B0_wstat = WStat(factor_B0*(PL_B0 + CPL_B0), B0['response'], B0['Non'], B0['Noff'], B0['Ton'],
              B0['Toff'],
              name='B0', channel=B0['channel'])
        # pi_J = pt.log(pm.math.det(model.d2logp())) / 2.0
        # pi_J = pm.Deterministic('prior', pi_J)
        # pm.Potential('pi_J', pi_J)
        idata = pm.sample(10000,
                          tune=2000,
                          random_seed=42,
                          # init='jitter+adapt_diag_grad',
                          # target_accept=0.95,
                          # idata_kwargs={'log_likelihood': True},
                          # chains=1
                          )
        pars_map = pm.find_MAP()
    #%%
    az.plot_trace(idata,
                  var_names=['PhoIndex_CPL', 'HighECut', 'PhoIndex_PL',
                             'norm_CPL', 'norm_PL', 'factor_B0'])
    #%%
    import corner
    corner.corner(
        data=idata,
        var_names=['PhoIndex_CPL', 'HighECut', 'PhoIndex_PL',
                   'norm_CPL', 'norm_PL', 'factor_B0'],
        # labels=['$\log A$', r'$\gamma$', '$\mathcal{F}$'],
        label_kwargs={'fontsize': 8},
        # quantiles=[0.15865, 0.5, 0.84135],
        levels=None,#[[0.683, 0.954, 0.997], [0.683, 0.90]][1],
        show_titles=True,
        title_fmt='.2f',
        color='#0C5DA5',
        #smooth=0.5,
        # range=((0,130),(-1.6,-2.7),(0,3.1e-7))[1:],
        truths=[pars_map,
                az.hdi(idata,0.02).mean(dim='hdi')][1],
        truth_color='red',
        max_n_ticks=5,
        hist_bin_factor=2
    )