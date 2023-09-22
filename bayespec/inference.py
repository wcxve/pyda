import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scienceplots
import scipy.stats as stats
import tqdm
import xarray as xr

from scipy.optimize import minimize, root_scalar
from iminuit import Minuit
import iminuit
from pymc.initial_point import make_initial_point_fn
from pymc.model import modelcontext
from pymc.sampling_jax import sample_numpyro_nuts, sample_blackjax_nuts
from pymc.util import get_default_varnames
from pytensor.compile.function import function
from pytensor.gradient import grad


__all__ = [
    'confint_boostrap', 'confint_profile', 'find_MLE', 'mcmc_nuts', 'ppc',
    'waic'
]


class ModelContext:
    pass


def find_MLE(
    model=None, start=None, method='L-BFGS-B', random_seed=42,
    return_raw=False, **kwargs
):
    model = modelcontext(model)
    vars_to_fit = model.continuous_value_vars
    fit_names = [var.name for var in vars_to_fit]
    fun = function(vars_to_fit, -model.datalogp)
    # jac = function(vars_to_fit, grad(-model.datalogp, vars_to_fit))
    vars_of_interest = [
        v
        for v in get_default_varnames(
            model.unobserved_value_vars,
            include_transformed=False
        )
        if '_BKG' not in v.name
    ]
    get_voi = function(vars_to_fit, vars_of_interest)

    ipfn = make_initial_point_fn(
        model=model,
        jitter_rvs=set(),
        return_transformed=True,
        overrides=start,
    )
    start = ipfn(random_seed)
    model.check_start_vals(start)
    x0 = [start[name] for name in fit_names]

    opt_res = minimize(fun=lambda x: fun(*x),
                       x0=x0,
                       method=method,
                       # jac=lambda x: jac(*x),
                       jac='3-point',
                       **kwargs)

    dof = -len(vars_to_fit)
    for i in model.data_names:
        dof += getattr(model, i+'_data').channel.size

    if opt_res.fun / dof > 1.0: # reduced chi2 > 2.0
        print(
            f'"{method}" seems not converged, use "powell" method to find MLE'
        )
        opt_res = minimize(fun=lambda x: fun(*x),
                           x0=opt_res.x,
                           method='powell')

        opt_res = minimize(fun=lambda x: fun(*x),
                           x0=opt_res.x,
                           method=method,
                           # jac=lambda x: jac(*x),
                           jac='3-point',
                           **kwargs)

    voi_name = [p.name for p in vars_of_interest]
    voi_value = get_voi(*opt_res.x)
    res = {name: value for name, value in zip(voi_name, voi_value)}
    if hasattr(opt_res, 'jac'):
        res['grad'] = np.linalg.norm(opt_res.jac)
        res['grad_vec'] = opt_res.jac
    res['stat'] = 2.0 * opt_res.fun
    res['dof'] = dof
    res['gof'] = stats.chi2.sf(res['stat'], res['dof'])
    k = len(vars_of_interest)
    n = res['dof'] + k
    res['AIC'] = res['stat'] + 2*k + 2*k*(k+1)/(n-k-1)
    res['BIC'] = res['stat'] + k*np.log(n)

    if return_raw:
        res['opt_res'] = opt_res

    return res


def mle(model=None, start=None, method='L-BFGS-B', random_seed=42):
    model = modelcontext(model)
    pars = model.free_RVs
    fit_pars = [model.rvs_to_values[p] for p in pars]
    fit_names = [p.name for p in fit_pars]

    deviance = function(fit_pars, -2 * model.datalogp)
    gradient = function(fit_pars, grad(-2 * model.datalogp, fit_pars))

    ipfn = make_initial_point_fn(
        model=model,
        jitter_rvs=set(),
        return_transformed=True,
        overrides=start,
    )
    start = ipfn(random_seed)
    model.check_start_vals(start)
    x0 = np.array([start[name] for name in fit_names])

    opt_res = minimize(fun=lambda x: deviance(*x),
                       x0=x0,
                       method='L-BFGS-B',
                       jac=lambda x: gradient(*x))
    m = Minuit(deviance, *opt_res.x)
    # print(iminuit.minimize(fun=lambda x: deviance(*x),
    #                    x0=x0,
    #                    method='migrad',
    #                    jac=lambda x: gradient(*x)))
    return m


def draw_posterior_samples(idata, nsample, random_seed=42):
    posterior = idata.posterior
    rng = np.random.default_rng(random_seed)
    i, j = rng.integers(low=[[0], [0]],
                        high=[[posterior.chain.size], [posterior.draw.size]],
                        size=(2, nsample))

    coords = {
        'chain': ('chain', [0]),
        'draw': ('draw', np.arange(nsample))
    }
    coords.update({
        k: v.values
        for k, v in posterior.coords.items()
        if k not in ['chain', 'draw']
    })
    posterior_dataset = xr.Dataset(
        data_vars={
            k: (v.coords.dims, np.expand_dims(v.values[i, j], axis=0))
            for k, v in posterior.data_vars.items()
        },
        coords={
            'chain': ('chain', [0]),
            'draw': ('draw', np.arange(nsample))
        }
    )
    idata2 = az.InferenceData(posterior=posterior_dataset)

    if 'log_likelihood' in idata.groups():
        log_likelihood = idata.log_likelihood
        coords = {
            'chain': ('chain', [0]),
            'draw': ('draw', np.arange(nsample))
        }
        coords.update({
            k: v.values
            for k, v in log_likelihood.coords.items()
            if k not in ['chain', 'draw']
        })
        log_likelihood_dataset = xr.Dataset(
            data_vars={
                k: (v.coords.dims, np.expand_dims(v.values[i, j], axis=0))
                for k, v in log_likelihood.data_vars.items()
            },
            coords=coords
        )
        idata2.add_groups({'log_likelihood': log_likelihood_dataset})

    return idata2


def ppc(idata, model=None, nsim=500, q=0.68269, random_seed=42):
    idata2 = draw_posterior_samples(idata, nsim, random_seed)

    if 'log_likelihood' not in idata2.groups():
        pm.compute_log_likelihood(idata2, model=model)

    log_likelihood = idata2.log_likelihood
    dims_to_sum = [
        d for d in log_likelihood.dims.keys()
        if d != 'chain' and d != 'draw'
    ]
    lnL = log_likelihood.sum(dims_to_sum).to_array().sum('variable').values

    posterior = idata2.posterior

    pars_name = [var.name for var in model.free_RVs]
    vars_to_fit = [model.rvs_to_values[rv] for rv in model.free_RVs]
    trans_name = [v.name for v in vars_to_fit]

    transforms = [model.rvs_to_transforms[rv] for rv in model.free_RVs]

    vec = pt.dvector()
    transform_forward = {
        v.name: function(
            [vec],
            trans.forward(vec, *v.owner.inputs) if trans is not None else vec
        )
        for trans, v in zip(transforms, model.free_RVs)
    }
    # transform_backward = {
    #     v.name: function(
    #         [vec],
    #         trans.backward(vec, *v.owner.inputs) if trans is not None else vec
    #     )
    #     for trans, v in zip(transforms, model.free_RVs)
    # }

    post_pars = {
        p: posterior[p].values[0]
        for p in pars_name
    }

    transformed_pars = {
        t: transform_forward[p](post_pars[p])
        for p, t in zip(pars_name, trans_name)
    }

    pm.sample_posterior_predictive(
        idata2, model, extend_inferencedata=True, random_seed=random_seed,
        progressbar=False
    )
    pdata = {
        k.replace('_Non', '_spec_counts').replace('_Noff', '_back_counts'):
            v.values[0]  # [0] is because there is only one chain in idata2
        for k, v in idata2.posterior_predictive.items()
    }

    deviance = function(vars_to_fit, -2 * model.datalogp)

    # # this is not right for profiled background
    # # calc D^rep
    # Drep = np.zeros(nsim)
    #
    # for i in model.data_names:
    #     data = getattr(model, f'{i}_data')
    #
    #     CE = getattr(getattr(model, f'{i}_model'), 'CE')
    #     CE_pars = [p for p in pars_name if p in CE.finder]
    #     CE_kwargs = {
    #         i: getattr(data, i) for i in CE.finder
    #         if type(i) == str and i not in pars_name
    #     }
    #
    #     nchan = data.channel.size
    #
    #     model_tot = np.zeros((nsim, nchan))
    #     factor_src = (data.ch_emax - data.ch_emin) * data.spec_exposure
    #
    #     for n in range(nsim):
    #         pars = {p: posterior_pars[p][n] for p in CE_pars}
    #         model_tot[n] = CE(**pars, **CE_kwargs) * factor_src
    #
    #     if getattr(model, f'_{i}_include_back'):
    #         back_poisson = getattr(model, f'_{i}_back_poisson')
    #         if back_poisson:
    #             bkg_type = 'BKGW'
    #         else:
    #             bkg_type = 'BKGPG'
    #         bkg_rate = posterior[f'{i}_{bkg_type}'].values[0]
    #         model_bkg = bkg_rate*data.back_exposure
    #         model_tot += bkg_rate*data.spec_exposure
    #         if back_poisson:
    #             Drep += -2 * poisson_lnL(model_bkg,
    #                                      pdata[f'{i}_back_counts'])
    #             print(poisson_lnL(model_bkg,
    #                                      pdata[f'{i}_back_counts']).mean())
    #         else:
    #             Drep += -2 * normal_lnL(model_bkg,
    #                                     pdata[f'{i}_back_counts'],
    #                                     data.back_error)
    #
    #     if getattr(model, f'_{i}_spec_poisson'):
    #         Drep += -2 * poisson_lnL(model_tot,
    #                                  pdata[f'{i}_spec_counts'])
    #         print(poisson_lnL(model_tot,
    #                           pdata[f'{i}_spec_counts']).mean())
    #     else:
    #         Drep += -2 * normal_lnL(model_tot,
    #                                 pdata[f'{i}_spec_counts'],
    #                                 data.spec_error)

    mle = find_MLE(model, return_raw=True)
    Dmin_obs = mle['stat']
    # pars_mle = {p: mle[p] for p in pars_name}

    Drep = np.zeros(nsim)
    Dmin = np.empty(nsim)
    # pars_fit = np.zeros((nsim, len(trans_names)))
    for i in tqdm.tqdm(range(nsim), desc='compute Dmin', file=sys.stdout):
        pm.set_data({d: pdata[d][i] for d in pdata}, model)
        pars_dict = {t: transformed_pars[t][i] for t in transformed_pars}
        Drep[i] = deviance(**pars_dict)
        opt_res = minimize(fun=lambda x: deviance(*x),
                           x0=[transformed_pars[t][i] for t in trans_name],
                           jac='2-point',
                           method='L-BFGS-B')
        Dmin[i] = opt_res.fun
        # pars_fit[i] = opt_res.x

    # pars_fit = {
    #     p: transform_backward[p](pars_fit[:, i])
    #     for i, p in enumerate(pars_names)
    # }

    observed = {
        k.replace('_Non', '_spec_counts').replace('_Noff', '_back_counts'):
            v.values
        for k, v in idata.observed_data.items()
    }
    pm.set_data(observed, model)

    plt.style.use(['nature', 'science', 'no-latex'])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(1, 2, figsize=(4, 2.2), dpi=150)
    fig.subplots_adjust(wspace=0.05)
    axes[0].set_box_aspect(1)
    axes[1].set_box_aspect(1)

    D = -2 * lnL
    _min = min(D.min(), Drep.min())*0.9
    _max = max(D.max(), Drep.max())*1.1
    axes[0].plot([_min, _max], [_min, _max], ls=':', color='gray')
    axes[0].set_xlim(_min, _max)
    axes[0].set_ylim(_min, _max)
    ppp1 = (Drep > D).sum()/D.size
    axes[0].set_title(f'$p$-value$=${ppp1:.3f}')
    axes[0].scatter(D, Drep, s=1)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('$D$')
    axes[0].set_ylabel(r'$D^{\rm rep}$')

    ppp2 = (Dmin > Dmin_obs).sum()/Dmin.size
    axes[1].set_title(f'$p$-value$=${ppp2:.3f}')
    axes[1].hist(Dmin, bins='auto')
    axes[1].axvline(Dmin_obs, c='r', ls='--')
    axes[1].set_xlabel(r'$D_{\rm min}$')
    axes[1].set_ylabel('$N$ simulation')
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_ticks_position('both')

    fig, axes = plt.subplots(2, 1, sharex=True, dpi=150)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.align_ylabels(axes)

    for i, color in zip(model.data_names, colors):
        data = getattr(model, f'{i}_data')
        # src_factor = (data.ch_emax - data.ch_emin) * data.spec_exposure
        bkg_factor = data.spec_exposure / data.back_exposure

        d = data.spec_counts
        if data.has_back:
            d = d - data.back_counts * bkg_factor
        Qd = d.cumsum()
        Qd /= Qd[-1]

        # CE = getattr(getattr(model, f'{i}_model'), 'CE')
        # CE_pars = [p for p in pars_name if p in CE.finder]
        # CE_kwargs = {
        #     j: getattr(data, j) for j in CE.finder
        #     if type(j) == str and j not in pars_name
        # }

        # m = np.empty((nsim, data.channel.size))
        # for n in range(nsim):
        #     m[n] = CE(**{p: post_pars[p][n] for p in CE_pars}, **CE_kwargs)
        # m *= src_factor
        # Qm = m.cumsum(-1)
        # Qm /= Qm[:, -1:]

        dsim = pdata[f'{i}_spec_counts']
        if f'{i}_back_counts' in pdata:
            dsim = dsim - pdata[f'{i}_back_counts'] * bkg_factor
        Qdsim = dsim.cumsum(-1)
        Qdsim /= Qdsim[:, -1:]

        # Qm_hdi = az.hdi(np.expand_dims(Qm, axis=0), hdi_prob=q).T
        Qdsim_hdi = az.hdi(np.expand_dims(Qdsim, axis=0), hdi_prob=q).T
        # Qdm_hdi = az.hdi(np.expand_dims(Qd - Qm, axis=0), hdi_prob=q).T
        # Qdsimm = Qdsim - Qm
        # Qdsimm_hdi = az.hdi(np.expand_dims(Qdsimm, axis=0), hdi_prob=q).T
        Qddsim = az.hdi(np.expand_dims(Qd - Qdsim, axis=0), hdi_prob=q).T

        d_err = Qdsim.std(axis=0, ddof=1)
        d_err[d_err==0.0] = 1
        Qddsim /= d_err

        # dm_err = Qdsimm.std(axis=0, ddof=1)
        # dm_err[dm_err==0.0] = 1.0
        # Qdm_hdi /= dm_err
        # Qdsimm_hdi /= dm_err

        mask = data.ch_emin[1:] != data.ch_emax[:-1]
        idx = [0, *(np.flatnonzero(mask) + 1), len(data.channel)]
        for j in range(len(idx) - 1):
            slice_j = slice(idx[j], idx[j + 1])
            ebins = np.append(data.ch_emin[slice_j], data.ch_emax[slice_j][-1])

            axes[0].step(
                ebins, np.append(Qd[slice_j], Qd[slice_j][-1]),
                lw=0.6, where='post', color=color
            )
            # axes[0].fill_between(
            #     ebins,
            #     np.append(Qm_hdi[0][slice_j], Qm_hdi[0][slice_j][-1]),
            #     np.append(Qm_hdi[1][slice_j], Qm_hdi[1][slice_j][-1]),
            #     lw=0.2, step='post', alpha=0.6, color=color
            # )
            axes[0].fill_between(
                ebins,
                np.append(Qdsim_hdi[0][slice_j], Qdsim_hdi[0][slice_j][-1]),
                np.append(Qdsim_hdi[1][slice_j], Qdsim_hdi[1][slice_j][-1]),
                lw=0, step='post', alpha=0.4, color='gray'
            )

            # axes[1].fill_between(
            #     ebins,
            #     np.append(Qdm_hdi[0][slice_j], Qdm_hdi[0][slice_j][-1]),
            #     np.append(Qdm_hdi[1][slice_j], Qdm_hdi[1][slice_j][-1]),
            #     lw=0.2, step='post', alpha=0.6, color=color
            # )
            axes[1].fill_between(
                ebins,
                np.append(Qddsim[0][slice_j], Qddsim[0][slice_j][-1]),
                np.append(Qddsim[1][slice_j], Qddsim[1][slice_j][-1]),
                lw=0, step='post', alpha=0.4, color='gray'
            )

    axes[1].axhline(0, ls=':', c='gray')
    axes[0].set_xscale('log')
    axes[0].set_ylabel('EDF')
    axes[1].set_ylabel('EDF residual')
    axes[1].set_xlabel('Energy [keV]')

    plt.style.use('default')

    return ppp1, ppp2


def waic(idata):
    if not hasattr(idata, 'log_likelihood'):
        raise ValueError('InferenceData has no log_likelihood')
    idata = idata.copy()
    log_likelihood = idata.log_likelihood
    data_name = [i.replace('_Non', '')
                 for i in log_likelihood.data_vars if '_Non' in i]
    lnL = []
    for name in data_name:
        lnL_i = log_likelihood[name+'_Non'].values
        if name+'_Noff' in log_likelihood.data_vars:
            lnL_i = lnL_i + log_likelihood[name + '_Noff'].values
        lnL.append(lnL_i)

    log_likelihood['all_channel'] = (
        ('chain', 'draw', 'channel'),
        np.concatenate(lnL, axis=-1)
    )
    elpd, se, p, *_ = pm.waic(idata, var_name='all_channel', scale='deviance')
    return elpd, se, p


def mcmc_nuts(
    model=None, draws=20000, tune=2000, target_accept=0.8, backend='numpyro',
    random_seed=42, **kwargs
):
    if backend not in ['numpyro', 'blackjax']:
        raise ValueError('sampling backend should be "numpyro" or "blackjax"')
    model = modelcontext(model)
    # pars_mle = find_MLE(model)
    # pars_dict = {v.name: pars_mle[v.name] for v in model.free_RVs}
    if backend == 'numpyro':
        idata = sample_numpyro_nuts(draws=draws,
                                    tune=tune,
                                    random_seed=random_seed,
                                    target_accept=target_accept,
                                    idata_kwargs={'log_likelihood': True},
                                    model=model,
                                    # initvals=pars_dict,
                                    **kwargs)
    else:
        idata = sample_blackjax_nuts(draws=draws,
                                     tune=tune,
                                     random_seed=random_seed,
                                     target_accept=target_accept,
                                     idata_kwargs={'log_likelihood': True},
                                     model=model,
                                     # initvals=pars_dict,
                                     **kwargs)
    return idata


def wbic(model):
    raise NotImplementedError

def _minimize_pll(model, par_names, start):
    ...


def confint_profile(
    model=None, cl=1.0, par_names=None, start=None,
    mini_meth='L-BFGS-B', root_meth='', random_seed=42,
    mini_kwargs={}, root_kwargs={}
):
    model = modelcontext(model)
    pars = model.free_RVs
    fit_pars = [model.rvs_to_values[p] for p in pars]
    fit_names = [p.name for p in fit_pars]

    nll = function(fit_pars, -model.datalogp)
    # nll_grad = function(fit_pars, grad(-model.datalogp, fit_pars))
    # jac = lambda x: nll_grad(*x)
    jac = '2-point'

    ipfn = make_initial_point_fn(
        model=model,
        jitter_rvs=set(),
        return_transformed=True,
        overrides=start,
    )
    start = ipfn(random_seed)
    model.check_start_vals(start)
    x0 = np.array([start[name] for name in fit_names])

    # opt_res = minimize(fun=lambda x: nll(*x),
    #                    x0=x0,
    #                    method='powell')
    opt_res = minimize(fun=lambda x: nll(*x),
                       x0=x0,
                       method=mini_meth,
                       jac=jac,
                       **mini_kwargs)

    f_opt = opt_res.fun
    x_opt = {fit_names[i]: v for i, v in enumerate(opt_res.x)}
    if type(opt_res.hess_inv) != np.ndarray:
        cov = opt_res.hess_inv.todense()
    else:
        cov = opt_res.hess_inv
    x_err = np.sqrt(cov.diagonal())
    x_err = {fit_names[i]: v for i, v in enumerate(x_err)}

    _par_names = [p.name for p in pars]
    if par_names is None:
        par_names = _par_names
        CI_names = fit_names
    else:
        CI_names = []
        for name in par_names:
            if name not in _par_names:
                raise ValueError(f'model has no parameter "{name}"')

        for i in range(len(fit_names)):
            if _par_names[i] in par_names:
                CI_names.append(fit_names[i])
            else:
                CI_names.append(None)

    if cl >= 1.0:
        delta = stats.chi2.ppf(1 - stats.norm.sf(cl) * 2, 1) / 2
        nsigma = cl
        target_stat = f_opt + delta
    else:
        delta = stats.chi2.ppf(cl, 1) / 2
        nsigma = stats.norm.isf((1 - cl) / 2)
        target_stat = f_opt + delta

    transforms = [model.rvs_to_transforms[rv] for rv in model.free_RVs]
    vec = pt.dvector()
    transform_backward = {
        v.name: function(
            [vec],
            trans.backward(vec, *v.owner.inputs) if trans is not None else vec
        )
        for trans, v in zip(transforms, model.free_RVs)
    }

    CI = {}
    for i, name in enumerate(CI_names):
        if name is not None:
            def nll_fixed_p(free, fixed):
                free_pars.update({p:v for p, v in zip(free_names, free)})
                return nll(**free_pars, **{name: fixed})
                # return nll(**free_pars, **{name: fixed[0]})

            def profile_nll(fixed_par):
                if len(free_names):
                    r = minimize(fun=nll_fixed_p,
                                 x0=[free_pars[p] for p in free_names],
                                 args=(fixed_par,),
                                 method=mini_meth,
                                 jac=jac,
                                 **mini_kwargs)
                    # if not r.success and np.linalg.norm(r.jac) > 1e-2:
                    #     print(
                    #         f'WARING: failed to minimize {par_names[i]} profile '
                    #         'likelihood'
                    #     )
                    free_pars.update({k: v for k, v in zip(free_names, r.x)})
                    return r.fun - target_stat
                else:
                    return nll(**{name: fixed_par}) - target_stat


            free_names = [p for p in fit_names if p != name]
            free_pars = {p: x_opt[p] for p in free_names}
            # prange = np.linspace(x_opt[name] - nsigma * x_err[name],
            #                      x_opt[name] + nsigma * x_err[name],
            #                      1000)
            # plt.figure()
            # plt.scatter(transform_backward[_par_names[i]](np.r_[x_opt[name]]),
            #             profile_nll(x_opt[name]))
            # plt.plot(transform_backward[_par_names[i]](prange),
            #          [profile_nll(i) for i in prange])
            try:
                l = root_scalar(profile_nll,
                                x0=x_opt[name] - nsigma * x_err[name],
                                bracket=(x_opt[name] - 3 * nsigma * x_err[name],
                                         # -np.inf,
                                         x_opt[name]))
                if not l.converged:
                    print(f'WARING: failed to estimate {par_names[i]} lower error ({l})')
                    l = np.nan
                else:
                    l = l.root
            except Exception as e:
                print(f'WARING: failed to estimate {par_names[i]} lower error ({e})')
                l = np.nan
            #
            # if x_err != 0.0:
            #     pdelta = -nsigma * x_err[name]
            # else:
            #     pdelta = -0.01*x_opt[name]
            # l = root(profile_nll, x_opt[name] + pdelta)
            #
            # if not l.success:
            #     print(f'WARING: failed to estimate {par_names[i]} lower error')

            free_pars = {k: v for k, v in x_opt.items() if k != name}
            try:
                u = root_scalar(profile_nll,
                                x0=x_opt[name] + nsigma * x_err[name],
                                bracket=(x_opt[name],
                                          x_opt[name] + 3 * nsigma * x_err[name]
                                         #np.inf
                                         ))
                if not u.converged:
                    print(f'WARING: failed to estimate {par_names[i]} upper error ({u})')
                    u = np.nan
                else:
                    u = u.root
            except Exception as e:
                print(f'WARING: failed to estimate {par_names[i]} upper error ({e})')
                u = np.nan

            # if x_err != 0.0:
            #     pdelta = nsigma * x_err[name]
            # else:
            #     pdelta = 0.01*x_opt[name]
            # u = root(profile_nll, x_opt[name] + pdelta)

            # if not u.success:
            #     print(f'WARING: failed to estimate {par_names[i]} upper error')

            pname = _par_names[i]
            transformed_pval = np.array([x_opt[name], l, u])
            # transformed_pval = np.array([x_opt[name], l.root, u.root])
            pval = transform_backward[pname](transformed_pval)
            # CI[pname] = (pval[0], pval[1] - pval[0], pval[2] - pval[0])
            CI[pname] = (pval[0], pval[1], pval[2])

    return CI


def confint_boostrap(
    model=None, cl=1.0, nboot=1000, start=None,  random_seed=42,
    opt_method='L-BFGS-B', **opt_kwargs
):
    model = modelcontext(model)
    pars_name = [v.name for v in model.free_RVs]
    transforms = [model.rvs_to_transforms[rv] for rv in model.free_RVs]
    vec = pt.dvector()
    back_transform = [
        function(
            [vec],
            trans.backward(vec, *v.owner.inputs) if trans is not None else vec
        )
        for trans, v in zip(transforms, model.free_RVs)
    ]
    trans_pars = [model.rvs_to_values[v] for v in model.free_RVs]
    trans_name = [v.name for v in trans_pars]

    loss = function(trans_pars, -model.datalogp)

    ipfn = make_initial_point_fn(
        model=model,
        jitter_rvs=set(),
        return_transformed=True,
        overrides=start,
    )
    start = ipfn(random_seed)
    model.check_start_vals(start)
    trans_init = np.array([start[name] for name in trans_name])

    mle_res = minimize(fun=lambda x: loss(*x),
                       x0=trans_init,
                       jac='3-point',
                       method=opt_method,
                       **opt_kwargs)

    trans_mle = np.array(mle_res.x)
    pars_mle = [
        f([p])[0] for f, p in zip(back_transform, trans_mle)
    ]
    pars_mle = np.array(pars_mle)

    mle_dataset = xr.Dataset(
        data_vars={
            name: (('chain', 'draw'), np.full((1, nboot), pars_mle[i]))
            for i, name in enumerate(pars_name)
        },
        coords={
            'chain': ('chain', [0]),
            'draw': ('draw', np.arange(nboot))
        }
    )
    idata = az.InferenceData(posterior=mle_dataset)
    pm.sample_posterior_predictive(
        idata, model,
        progressbar=False, extend_inferencedata=True, random_seed=42
    )
    pdata = {
        k.replace('_Non', '_spec_counts').replace('_Noff', '_back_counts'):
            v.values[0]  # [0] is because there is only one chain in idata
        for k, v in idata.posterior_predictive.items()
    }

    trans_boot = np.empty((nboot, len(trans_name)))
    for i in tqdm.tqdm(range(nboot), desc='Bootstrap', file=sys.stdout):
        pm.set_data({d: pdata[d][i] for d in pdata}, model)
        opt_res = minimize(fun=lambda x: loss(*x),
                           x0=trans_mle,
                           jac='2-point',
                           method='L-BFGS-B')
        trans_boot[i] = opt_res.x
    observed = {
        k.replace('_Non', '_spec_counts').replace('_Noff', '_back_counts'):
            v.values
        for k, v in idata.observed_data.items()
    }
    pm.set_data(observed, model)

    pars_boot = np.empty((nboot, len(trans_name)))
    for i in range(len(trans_name)):
        pars_boot[:, i] = back_transform[i](trans_boot[:, i])

    boot_dataset = xr.Dataset(
        data_vars={
            name: (('chain', 'draw'), np.expand_dims(pars_boot[:, i], axis=0))
            for i, name in enumerate(pars_name)
        },
        coords={
            'chain': ('chain', [0]),
            'draw': ('draw', np.arange(nboot))
        }
    )
    if cl >= 1.0:
        q = 1 - stats.norm.sf(cl) * 2
    else:
        q = cl

    idata_boot = az.InferenceData(posterior=boot_dataset)
    CI = {
        k: np.array([pars_mle[i], v[0], v[1]])
        for i, (k, v) in enumerate(az.hdi(idata_boot, q).items())
    }
    return CI
