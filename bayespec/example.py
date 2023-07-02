import pymc as pm
from pyda.bayespec import *


if __name__ == '__main__':
    import os
    path = '/Users/xuewc/BurstData/GRB221009A/900-1100'
    os.chdir(path)
    N8_data = Data(erange=[[20, 30], [40, 900]],
                   specfile=f'{path}/phase_n8_900-1100.pha',
                   backfile=f'{path}/phase_n8_900-1100.bkg',
                   respfile=f'{path}/n8_phase2.rsp',
                   name='N8')
    B0_data = Data(erange=[1000, 20000],
                   specfile=f'{path}/phase_b0_900-1100.pha',
                   backfile=f'{path}/phase_b0_900-1100.bkg',
                   respfile=f'{path}/b0_phase2.rsp',
                   name='B0')
    stat = Wstat
    with pm.Model() as model:
        norm_CPL = pm.Uniform('norm_CPL', lower=1e-5, upper=200)
        # PhoIndex_CPL = pm.Flat('PhoIndex_CPL')
        # HighECut = pm.HalfFlat('HighECut')
        PhoIndex_CPL = pm.Uniform('PhoIndex_CPL', lower=0.01, upper=5)
        HighECut = pm.Uniform('HighECut', lower=0.01, upper=1000)

        norm_PL = pm.Uniform('norm_PL', lower=1e-5, upper=100)
        PhoIndex_PL = pm.Uniform('PhoIndex_PL', lower=0.01, upper=5)
        factor_B0 = pm.Uniform('factor_B0', lower=1e-5, upper=10)
        CPL = norm_CPL*CutoffPowerlaw(PhoIndex_CPL, HighECut)
        PL = norm_PL*Powerlaw(PhoIndex_PL)
        src = CPL + PL
        stat(src, N8_data)
        stat(factor_B0 * src, B0_data)
        idata = mcmc_nuts()
        plot_trace(idata)
        plot_corner(idata)
        # print(find_MLE())