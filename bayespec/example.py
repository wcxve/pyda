import pymc as pm
from pyda.bayespec import *


if __name__ == '__main__':
    import os
    path = '/Users/xuewc/BurstData/GRB221009A/900-1100'
    os.chdir(path)
    N8_data = Data(erange=[(20, 30), (40, 900)],
                   specfile=f'{path}/phase_n8_900-1100.pha',
                   backfile=f'{path}/phase_n8_900-1100.bkg',
                   respfile=f'{path}/n8_phase2.rsp')
    B0_data = Data(erange=[(1000, 20000),],
                   specfile=f'{path}/phase_b0_900-1100.pha',
                   backfile=f'{path}/phase_b0_900-1100.bkg',
                   respfile=f'{path}/b0_phase2.rsp')
    with pm.Model() as model:
        norm_CPL = pm.HalfFlat('norm_CPL')
        # PhoIndex_CPL = pm.Flat('PhoIndex_CPL')
        # HighECut = pm.HalfFlat('HighECut')
        PhoIndex_CPL = pm.Uniform('PhoIndex_CPL', lower=0, upper=5)
        HighECut = pm.Uniform('HighECut', lower=0, upper=1000)

        norm_PL = pm.HalfFlat('norm_PL')
        PhoIndex_PL = pm.Flat('PhoIndex_PL')

        factor_B0 = pm.HalfFlat('factor_B0')

        # CPL = norm_CPL*XspecModel('cutoffpl', PhoIndex_CPL, HighECut)
        CPL = norm_CPL*CutoffPowerlaw(PhoIndex_CPL, HighECut)
        PL = norm_PL*Powerlaw(PhoIndex_PL)
        src = CPL + PL
        Wstat(src, N8_data)
        Wstat(factor_B0*src, B0_data)
        # idata = pm.sample(20000,
        #                   tune=1000,
        #                   random_seed=42,
        #                   init='jitter+adapt_diag_grad',
        #                   target_accept=0.95,
        #                   idata_kwargs={'log_likelihood': True},
        #                   )
        pars_map = pm.find_MAP(include_transformed=False)
    print(pars_map)
    # #%%
    # pm.plot_trace(idata,
    #               var_names=['PhoIndex_CPL', 'HighECut', 'PhoIndex_PL',
    #                          'norm_CPL', 'norm_PL', 'factor_B0'])
    # #%%
    # import corner
    # corner.corner(
    #     data=idata,
    #     var_names=['PhoIndex_CPL', 'HighECut', 'PhoIndex_PL',
    #                'norm_CPL', 'norm_PL', 'factor_B0'],
    #     # labels=['$\log A$', r'$\gamma$', '$\mathcal{F}$'],
    #     label_kwargs={'fontsize': 8},
    #     # quantiles=[0.15865, 0.5, 0.84135],
    #     levels=None,#[[0.683, 0.954, 0.997], [0.683, 0.90]][1],
    #     show_titles=True,
    #     title_fmt='.2f',
    #     color='#0C5DA5',
    #     #smooth=0.5,
    #     # range=((0,130),(-1.6,-2.7),(0,3.1e-7))[1:],
    #     truths=[pars_map,
    #             az.hdi(idata,0.02).mean(dim='hdi')][0],
    #     truth_color='red',
    #     max_n_ticks=5,
    #     hist_bin_factor=2
    # )