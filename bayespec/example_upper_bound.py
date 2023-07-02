import pymc as pm
from pyda.bayespec import *

if __name__ == '__main__':
    NaI_data = Data(
        [28, 250],
        '/Users/xuewc/BurstData/EXO2030+375/upper_limit/NaI_TOTAL.fits{1}',
        '/Users/xuewc/BurstData/EXO2030+375/upper_limit/NaI_BKG.fits{1}',
        '/Users/xuewc/BurstData/EXO2030+375/upper_limit/HE_rsp.rsp',
        name='NaI'
    )
    CsI_data = Data(
        [200, 600],
        '/Users/xuewc/BurstData/EXO2030+375/upper_limit/CsI_TOTAL.fits{1}',
        '/Users/xuewc/BurstData/EXO2030+375/upper_limit/CsI_BKG.fits{1}',
        '/Users/xuewc/BurstData/EXO2030+375/upper_limit/DA.rsp',
        name='CsI'
    )

    with pm.Model() as model:
        norm_posm = pm.Uniform('norm_posm', 1e-5, 1)
        # norm_cpl = 10**pm.Uniform('log_norm_cpl', -5, 2)
        # norm_cpl = pm.Uniform('norm_cpl', 1e-5, 100)
        # PhoIndex = pm.Uniform('PhoIndex', -3, 3)
        # HighECut = 10**pm.Uniform('log_HighECut', -5, 2)
        # HighECut = pm.Uniform('HighECut', 1e-5, 100)
        posm = norm_posm * xsmodel.posm()
        # cpl = norm_cpl * CutoffPowerlaw(PhoIndex, HighECut)
        cpl = 0.491 * CutoffPowerlaw(1.075, 20)
        src = cpl + posm
        # Wstat(src, NaI_data)
        Chi2(src, CsI_data)
        # idata = mcmc_nuts()
        # ci_boot = confint_boostrap(model)
        plot_spec(model_context=model)
        print(find_MLE(model))