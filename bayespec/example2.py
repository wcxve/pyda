import matplotlib.pyplot as plt
import pymc as pm
from pyda.bayespec import *

if __name__ == '__main__':
    path = '/Users/xuewc/BurstData/FRB221014/HXMT/'
    LE = Data([1, 11],
              f'{path}/LE_bmin5.grp',
              f'{path}/LE_phabkg20s_g0_0-94.pha',
              f'{path}/LE_rsp.rsp')
    ME = Data([8, 35],
              f'{path}/ME_bmin5.grp',
              f'{path}/ME_phabkg20s_g0_0-53.pha',
              f'{path}/ME_rsp.rsp')
    HE = Data([18, 250],
              f'{path}/HE_bmin5.grp',
              f'{path}/HE_phabkg20s_g0_0-12.pha',
              f'{path}/HE_rsp.rsp')

    stat = Wstat
    with pm.Model() as model:
        PhoIndex = pm.Flat('PhoIndex')
        # HighECut = pm.Uniform('HighECut', lower=0.01, upper=300)
        # kT = pm.Uniform('kT', lower=0.01, upper=100)
        norm = pm.HalfFlat('norm')

        # src = norm * OOTB(kT)
        src = norm * Powerlaw(PhoIndex)
        # src = norm * CutoffPowerlaw(PhoIndex, HighECut)
        wabs = xsmodel.wabs(2.79)
        stat(wabs*src, LE)
        stat(src, ME)
        stat(src, HE)
        idata = mcmc_nuts(model)
        plot_corner(idata)
        ppc(idata, model)

    # plot_spec(model_context=model)
    # assess_goodness(idata, model)
    # plot_corner(idata)
    #     pars_map = pm.find_MAP(include_transformed=False)
    #
    # channel_dims = [
    #     d for d in idata.log_likelihood._dims if d not in ['chain', 'draw']
    # ]
    # lnL = idata.log_likelihood.sum(channel_dims).to_array().sum('variable')
    # argmax = lnL.argmax(...)
    # pars_mle = idata.posterior.isel(argmax)
    #
    # pm.plot_trace(idata)
    # #%%
    # import corner
    # fig = corner.corner(
    #     data=idata,
    #     bins=40,
    #     # quantiles=[0.15865, 0.5, 0.84135],
    #     levels=[[0.683, 0.954, 0.997], [0.393, 0.865, 0.989], [0.393, 0.683, 0.9]][-1],
    #     show_titles=True,
    #     title_fmt='.2f',
    #     color='#0343DF',
    #     max_n_ticks=5,
    #     smooth1d=0.0,
    #     smooth=[0, 0.5][0],
    #     no_fill_contours=True,
    #     data_kwargs={'alpha': 1},
    #     pcolor_kwargs={'alpha': 0.95,
    #                    'edgecolors': 'face',
    #                    'linewidth': 0,
    #                    'rasterized': True},
    #     labelpad=-0.08,
    #     use_math_text=True,
    #     # label_kwargs={'fontsize': 14},
    #     # title_kwargs={'fontsize': 14}
    # )
    # # fig.axes[6].errorbar([1.77], [136.23], [[38.58], [59.96]],
    # #                      [[0.12], [0.08]], c='r', fmt='.', lw=1, ms=1)
    # # fig.axes[12].errorbar([1.77], [1.78], [[0.12], [0.11]],
    # #                      [[0.12], [0.08]], c='r', fmt='.', lw=1, ms=1)
    # # fig.savefig('test.pdf')
    #
    # import numpy as np
    # from scipy.stats import chi2, norm
    # from scipy.spatial import ConvexHull
    # cl = 1 - norm.sf(1)*2
    # dof = 2
    # delta = chi2.ppf(cl, dof)/2
    #
    # # mask = np.abs(lnL - (lnL.max() - delta)) < delta / 100
    # # samples = idata.posterior.to_array().to_numpy()[:, mask]
    # # idx = ConvexHull(samples.T).vertices
    # # idx = np.append(idx, idx[0])
    # # # fig.axes[2].scatter(*samples, c='tab:red', s=5)
    # # fig.axes[2].plot(*samples[:, idx], c='tab:red', ls=':')
    #
    # # delta = chi2.ppf(0.9, dof) / 2
    # # mask = np.abs(lnL - (lnL.max() - delta)) < delta / 100
    # # samples = idata.posterior.to_array().to_numpy()[:, mask]
    # # idx = ConvexHull(samples.T).vertices
    # # idx = np.append(idx, idx[0])
    # # # fig.axes[2].scatter(*samples, c='tab:red', s=5)
    # # fig.axes[2].plot(*samples[:, idx], c='tab:red', ls=':', label='68.3% & 90% CL')
    # # fig.axes[2].legend(frameon=False)
    # # fig.savefig('test.pdf')
    #
    # stat = lnL.values.reshape(-1)
    # argsort = stat.argsort()[::-1]
    # stat = stat[argsort]
    # pars = idata.posterior.to_array().to_numpy().reshape(2, -1)[:, argsort]
    # mask = stat >= stat[0] - delta
    # samples = pars[:, mask]
    # idx = ConvexHull(samples.T).vertices
    # idx = np.append(idx, idx[0])
    # fig.axes[2].plot(*samples[:, idx], c='tab:red', ls=':',
    #                  label='68.3% & 90% CL')
    # fig.axes[2].legend(frameon=False)
    # p1 = pars[0, mask]
    # # argsort = p1.argsort()
    # # fig.axes[0].twinx().scatter(p1[argsort], stat[mask][argsort], s=1, alpha=0.3)
    # idx = ConvexHull(np.column_stack((p1, stat[mask]))).vertices
    # fig.axes[0].twinx().plot(p1[idx], stat[mask][idx])
    # # pars_mask = np.abs((idata.posterior.PhoIndex-pars_mle[0])/pars_mle[0]) < 0.01