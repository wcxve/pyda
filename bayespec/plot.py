import matplotlib.pyplot as plt
import numpy as np

from corner import corner
from pymc import find_MAP, modelcontext
from scipy.interpolate import splev, splrep
from scipy.spatial import ConvexHull
from scipy.stats import chi2

__all__ = [
    'plot_corner', 'plot_spec', 'calc_vFv'
]

KEVTOERGS = 1.6021766339999e-9


def plot_corner(
    idata, delta1d=True, delta2d=False, level_idx=-1, smooth=0.0,
    fig_path=None, **kwargs
):
    CL = [
        [0.683, 0.954, 0.997],  # 1/2/3-sigma for 1d normal
        [0.393, 0.865, 0.989],  # 1/2/3-sigma for 2d normal
        [0.683, 0.9],  # 68.3% and 90%
        [0.393, 0.683, 0.9]  # 1-sigma for 2d, 68.3% and 90%
    ]

    fig = corner(
        data=idata,
        bins=40,
        quantiles=[0.15865, 0.5, 0.84135],
        levels=CL[level_idx],
        show_titles=True,
        title_fmt='.2f',
        color='#0343DF',
        # truths=[pars_mle, pars_map, pm.hdi(idata,0.02).mean(dim='hdi')][0],
        # truth_color='tab:red',
        max_n_ticks=5,
        smooth1d=0.0,
        smooth=smooth,
        no_fill_contours=True,
        data_kwargs={'alpha': 1},
        pcolor_kwargs={'alpha': 0.95,
                       'edgecolors': 'face',
                       'linewidth': 0,
                       'rasterized': True},
        labelpad=-0.08,
        use_math_text=True,
        # label_kwargs={'fontsize': 14},
        # title_kwargs={'fontsize': 14}
        **kwargs
    )

    if (delta1d or delta2d) and hasattr(idata, 'log_likelihood'):
        nvars = len(idata.posterior.data_vars)
        dims_to_sum = [
            d for d in idata.log_likelihood._dims if d not in ['chain', 'draw']
        ]
        lnL = idata.log_likelihood.sum(dims_to_sum).to_array().sum('variable')
        lnL = lnL.values.reshape(-1)
        lnL_max = lnL.max()
        pars = idata.posterior.to_array().to_numpy().reshape(nvars, -1)

        if delta1d:
            for i in range(nvars):
                x_best = pars[i, lnL.argmax()]
                y_best = lnL.max()
                pars_lnL = np.column_stack((pars[i], lnL))
                idx = ConvexHull(pars_lnL).vertices
                twinx = fig.axes[i * (nvars + 1)].twinx()
                x, y = get_profile(pars_lnL[idx])
                twinx.plot(x, -y, c='tab:red')
                twinx.scatter(x_best, -y_best, c='tab:red', marker='o', s=15)
                # mask1 = x < x_best
                # idx1 = np.abs(lnL_max - y[mask1] - 0.5).argmin()
                # twinx.scatter(x[mask1][idx1], -y[mask1][idx1], c='tab:red', marker='s', s=15)
                # mask2 = x > x_best
                # idx2 = np.abs(lnL_max - y[mask2] - 0.5).argmin()
                # twinx.scatter(x[mask2][idx2], -y[mask2][idx2], c='tab:red', marker='s', s=15)
                twinx.yaxis.set_visible(False)
                twinx.axhline(-lnL_max + 0.5, c='tab:red', ls=':')

        if delta2d:
            for cl in CL[level_idx]:
                dof = 2
                delta = chi2.ppf(cl, dof) / 2
                mask = lnL >= lnL_max - delta
                samples = pars[:, mask]
                for i in range(nvars):
                    for j in range(i):
                        pair = np.column_stack((samples[j], samples[i]))
                        idx = ConvexHull(pair).vertices
                        idx = np.append(idx, idx[0])
                        fig.axes[i * nvars + j].plot(*pair[idx].T, c='tab:red',
                                                     ls=':')

    if fig_path is not None:
        fig.savefig(fig_path)

    return fig


def plot_spec(plot_cmd='ldata delchi', model_context=None, idata=None, show_pars=[], **kwargs):
    # colors = ["#0d49fb", "#e6091c", "#26eb47", "#8936df", "#fec32d", "#25d7fd"]
    # colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377',
    #           '#BBBBBB']
    colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e']
    fig, axes = plt.subplots(
        3, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [1.6, 1, 1]},
        figsize=[4 * 1.5, 3 * 1.5],
        dpi=100
    )
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.align_ylabels(axes)

    MAP, opt_res = find_MAP(model=model_context,
                            include_transformed=False,
                            return_raw=True,
                            **kwargs)
    print(f'grad={np.linalg.norm(opt_res.jac):.3e}',
          {p: np.around(MAP[p], 3) for p in MAP})

    for i, data_name in enumerate(model_context.data_names):
        color = colors[i]
        data = getattr(model_context, f'{data_name}_data')
        if i == 1:
            tinfo = [int(data._spec_data['TIME'][0]),
                     int(data._spec_data['ENDTIME'][0])]
            axes[0].annotate(tinfo,
                             xy=(0.96, 0.95), xycoords='axes fraction',
                             ha='right', va='top')
        if i == 0 and len(show_pars):
            pars_info = []
            for pname in show_pars:
                pars_info.append(f'{pname}: {MAP[pname]: .2f}')
            axes[0].annotate('\n'.join(pars_info),
                             xy=(0.04, 0.05), xycoords='axes fraction',
                             ha='left', va='bottom')
        CE = getattr(model_context, f'{data_name}_CE')
        pars = {j: MAP[j] for j in MAP if j in CE.finder}
        CE = CE(**pars)

        emid = np.sqrt(data.ch_ebins[:, 0] * data.ch_ebins[:, 1])
        eerr = np.abs(data.ch_ebins.T - emid)
        ebin_width = data.ch_ebins[:, 1] - data.ch_ebins[:, 0]
        net = (
                          data.spec_counts / data.spec_exposure - data.back_counts / data.back_exposure) / ebin_width
        net_err = np.sqrt(
            np.square(data.spec_error / data.spec_exposure)
            + np.square(data.back_error / data.back_exposure)
        ) / ebin_width
        if f'{data_name}_f' in MAP:
            factor = MAP[f'{data_name}_f']
            label = f'{data_name} ({factor:.2f})'
        else:
            label = f'{data_name}'
        axes[0].errorbar(emid, net, net_err, eerr, fmt=' ', c=color, lw=0.7,
                         label=label)
        axes[0].loglog()

        mask = data.ch_ebins[1:, 0] != data.ch_ebins[:-1, 1]
        idx = [
            0,
            *(np.flatnonzero(mask) + 1),
            len(data.ch_ebins)
        ]
        net_cumsum = (
                    data.spec_counts - data.back_counts / data.back_exposure * data.spec_exposure).cumsum()
        CE_cumsum = (CE * ebin_width * data.spec_exposure).cumsum()
        cumsum_max = max(net_cumsum.max(), CE_cumsum.max())
        net_cumsum /= cumsum_max
        CE_cumsum /= cumsum_max
        for k in range(len(idx) - 1):
            slice_k = slice(idx[k], idx[k + 1])
            ch_ebins_k = data.ch_ebins[slice_k]
            ch_ebins_k = np.append(ch_ebins_k[:, 0], ch_ebins_k[-1, 1])
            CE_k = CE[slice_k]
            CE_k = np.append(CE_k, CE_k[-1])
            CE_cumsum_k = CE_cumsum[slice_k]
            CE_cumsum_k = np.append(CE_cumsum_k, CE_cumsum_k[-1])
            axes[0].step(ch_ebins_k, CE_k, where='post', c=color, lw=1.3)
            axes[1].step(ch_ebins_k, CE_cumsum_k, where='post', c=color, lw=1.3)
        axes[1].errorbar(emid, net_cumsum, xerr=eerr, fmt=' ', zorder=2,
                         c=color, lw=1)
        axes[1].set_ylabel('EDF')
        # axes[1].errorbar(emid, net/CE, net_err/CE, eerr, fmt=' ', zorder=2, c=color, lw=1)
        # axes[1].axhline(1, ls='-', c='#00FF00', zorder=0, lw=1)
        # axes[1].set_ylabel('data/model')
        axes[2].errorbar(emid, (net - CE) / net_err, 1, eerr, fmt=' ',
                         zorder=2, c=color, lw=1)
        axes[2].axhline(0, ls='-', c='#00FF00', zorder=0, lw=1)
        # axes[2].set_ylabel('(data$\,-\,$model)$\,$/$\,$error')
        axes[2].set_ylabel('$\chi$')
        axes[-1].set_xlabel('Energy [keV]')
        axes[0].set_ylabel('$C_E$ [s$^{-1}$ keV$^{-1}$]')

    # axes[1].set_yticks([-3,-2,-1,0,1,2,3])
    # axes[1].set_yticklabels([-3,-2,-1,0,1,2,3], ha='center')
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=10,
        fancybox=True, shadow=True, ncol=6, frameon=True,
        columnspacing=1, handletextpad=0.3
    )
    for ax in axes:
        ax.tick_params(axis='both', which='both', direction='in', top=True,
                       right=True)
    # axes[0].set_xlim(data.ch_ebins.min()*0.95, data.ch_ebins.max()*1.05)

    # axes[1].set_ylim(-3.9, 3.9)
    # ylim = np.abs(axes[1].get_ylim()).max()
    # axes[1].set_ylim(-ylim, ylim)


def calc_vFv(idata, model_context=None, data_names=None, q=0.683, ndraw=1000):
    rng = np.random.default_rng(seed=42)
    idx_c = rng.integers(0, len(idata.posterior.chain), ndraw)
    idx_d = rng.integers(0, len(idata.posterior.draw), ndraw)

    model_context = modelcontext(model_context)
    if data_names is None:
        data_names = model_context.data_names

    q = [0.5, 0.5 - q/2, 0.5 + q/2]

    vFv = {}
    for data_name in data_names:
        vFv_i = getattr(model_context, f'{data_name}_E2NE')
        pars_i = [p for p in vFv_i.finder if type(p) is str]
        pars_i = [idata.posterior[p].values[idx_c, idx_d] for p in pars_i]
        pars_i = np.column_stack(pars_i)
        vFv_i = [vFv_i(*p_i) for p_i in pars_i]
        ci = np.quantile(vFv_i, q, axis=0)
        vFv[data_name] = ci * KEVTOERGS

    return vFv

def calc_vFv_from_posterior(idata, model_context=None, ndraw=500):
    rng = np.random.default_rng(seed=42)
    idx_c = rng.integers(0, len(idata.posterior.chain), ndraw)
    idx_d = rng.integers(0, len(idata.posterior.draw), ndraw)

    model_context = modelcontext(model_context)
    vFv = {}
    for i, data_name in enumerate(model_context.data_names):
        vFv_i = getattr(model_context, f'{data_name}_E2NE')
        pars_i = [p for p in vFv_i.finder if type(p) is str]
        pars_i = [idata.posterior[p].values[idx_c, idx_d] for p in pars_i]
        pars_i = np.column_stack(pars_i)
        vFv_i = np.array([vFv_i(*p_i) for p_i in pars_i])
        vFv[data_name] = vFv_i * KEVTOERGS

    return vFv


def calc_vFv_from_MLE(model_context=None, **kwargs):
    model_context = modelcontext(model_context)
    MAP = find_MAP(model=model_context, include_transformed=False, **kwargs)
    vFv = {}
    for i, data_name in enumerate(model_context.data_names):
        vFv_i = getattr(model_context, f'{data_name}_E2NE')
        pars_i = [p for p in vFv_i.finder if type(p) is str]
        pars_i = [MAP[p] for p in pars_i]
        vFv_i = vFv_i(*pars_i)
        vFv[data_name] = vFv_i * KEVTOERGS

    return vFv


def credible_interval(sample, line=0.5, quantile=0.6826894921370859):
    q = [line, 0.5-quantile/2, 0.5+quantile/2]
    return np.quantile(sample, q, axis=0)


def confidence_interval():
    pass


def get_profile(pars_lnL):
    pars, lnL = pars_lnL.T
    argsort = pars.argsort()
    pars = pars[argsort]
    lnL = lnL[argsort]
    p_best = pars[lnL.argmax()]
    lnL_max = lnL.max()
    lnL_pmin = lnL[pars.argmin()]
    lnL_pmax = lnL[pars.argmax()]
    x = []
    y = []
    for i, j in zip(pars, lnL):
        if i <= p_best:
            if lnL_pmin <= j <= lnL_max:
                x.append(i)
                y.append(j)
        else:
            if lnL_pmax <= j <= lnL_max:
                x.append(i)
                y.append(j)
    spl = splrep(x[1:-1], y[1:-1], k=2)
    newx = np.linspace(pars.min(), pars.max(), 100)
    newy = splev(newx, spl)
    return newx, newy