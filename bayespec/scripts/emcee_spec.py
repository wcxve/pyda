# -*- coding: utf-8 -*-
"""
@author: xuewc
"""

import numpy as np

from astropy.io import fits
from numba import njit
from numpy import log, sqrt

from mxspec._pymXspec import callModFunc


@njit(
    'Tuple((float64[:], float64[:]))'
    '(float64[:], float64[:], float64[:], float64, float64)'
)
def wstat(rate_src, n_on, n_off, t_on, t_off):
    n = len(n_on)
    stat = np.empty(n)
    mu_bkg = np.empty(n)

    mu_src = rate_src * t_on
    a = t_on / t_off
    v1 = a + 1.0      # a + 1
    v2 = 1.0 + 1.0/a  # 1 + 1/a
    v3 = 2.0 * v1     # 2*(a+1)
    v4 = 4 * a * v1   # 4*a*(a+1)

    for i in range(n):
        on = n_on[i]
        off = n_off[i]
        s = mu_src[i]

        if on == 0.0:
            stat[i] = s + off*log(v1)
            mu_bkg[i] = off / v2 if off > 0.0 else 0.0
        else:
            if off == 0.0:
                if s <= on / v2:
                    stat[i] = -s/a + on*log(v2)
                    mu_bkg[i] = on / v2 - s
                else:
                    stat[i] = s + on*(log(on/s) - 1.0)
                    mu_bkg[i] = 0.0
            else:
                c = a * (on + off) - v1 * s
                d = sqrt(c*c + v4 * off * s)
                b = (c + d) / v3
                stat[i] = s + v2 * b \
                        - on * (log((s + b)/on) + 1) \
                        - off * (log(b/a/off) + 1)
                mu_bkg[i] = b

    return stat, mu_bkg/t_on


def data_for_wstat(emin, emax, spec_on, spec_off, respfile):
    emin = np.atleast_1d(emin)[:, np.newaxis]
    emax = np.atleast_1d(emax)[:, np.newaxis]
    with fits.open(spec_on) as hdul:
        data = hdul['SPECTRUM'].data
        indices = np.where(data['GROUPING'] == 1)[0]
        n_on = np.add.reduceat(data['COUNTS'], indices)
        t_on = hdul['SPECTRUM'].header['EXPOSURE']

    with fits.open(spec_off) as hdul:
        data = hdul['SPECTRUM'].data
        n_off = np.add.reduceat(data['COUNTS'], indices)
        t_off = hdul['SPECTRUM'].header['EXPOSURE']

    with fits.open(respfile) as hdul:
        matrix = hdul['MATRIX'].data
        Eph_bins = np.append(matrix['ENERG_LO'], matrix['ENERG_HI'][-1])
        drm = np.add.reduceat(matrix['MATRIX'], indices, axis=1)

        ebounds = hdul['EBOUNDS'].data[indices]
        Ech_bins = np.append(ebounds['E_MIN'], ebounds['E_MAX'][-1])
        Ech_bins = np.column_stack((Ech_bins[:-1], Ech_bins[1:]))
        chmask = (emin <= Ech_bins[:, 0]) & (Ech_bins[:, 1] <= emax)
        chmask = np.any(chmask, axis=0)

    data = {
        'N_on': np.asarray(n_on[chmask], dtype=np.float64),
        'N_off': np.asarray(n_off[chmask], dtype=np.float64),
        'T_on': t_on,
        'T_off': t_off,
        'Eph_bins': np.asarray(Eph_bins, dtype=np.float64),
        'Ech_bins': np.asarray(Ech_bins[chmask], dtype=np.float64),
        'DRM': np.asarray(drm[:, chmask], dtype=np.float64)
    }

    return data


class PowerLaw:
    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64)

    def __call__(self, PhoIndex):
        return self._eval(PhoIndex, self.ebins)

    @staticmethod
    @njit('float64[::1](float64, float64[::1])')
    def _eval(PhoIndex, ebins):
        if PhoIndex != 1.0:
            NE = ebins**(1.0 - PhoIndex) / (1.0 - PhoIndex)
        else:
            NE = np.log(ebins)

        return (NE[1:] - NE[:-1])


class CutoffPowerLaw:
    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64)

    def __call__(self, alpha, Ecut):
        return self._eval(alpha, Ecut, self.ebins)

    @staticmethod
    def _eval(alpha, Ecut, ebins):
        flux = []
        callModFunc('cutoffpl', ebins, (alpha, Ecut), flux, [], 1, '')

        return np.asarray(flux)


class BBodyRad:
    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64)

    def __call__(self, kT):
        return self._eval(kT, self.ebins)

    @staticmethod
    @njit('float64[::1](float64, float64[::1])')
    def _eval(kT, ebins):
        # this is from xspec
        N = len(ebins)
        flux = np.empty(N-1)

        el = ebins[0]
        x = el/kT
        if x <= 1.0e-4:
            nl = el*kT # limit_{el/kT->1} el*el/(exp(el/kT)-1) = el*kT
        elif x > 60.0:
            flux[:] = 0.0
            return flux
        else:
            nl = el*el/(np.exp(x) - 1)

        # norm of 2-point approximation to integral
        norm = 1.0344e-3 / 2.0 # BBodyRad
        # kT2 = kT*kT
        # norm = 8.0525 / (kT2*kT2) / 2.0 # BBody

        for i in range(N-1):
            eh = ebins[i+1]
            x = eh/kT
            if x <= 1.0e-4:
                nh = eh*kT
            elif x > 60.0:
                flux[i:] = 0.0
                break
            else:
                nh = eh*eh/(np.exp(x)-1.0)
            flux[i] = norm * (nl + nh) * (eh - el)
            el = eh
            nl = nh

        return flux


class WAbs:
    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64)

    def __call__(self, nH):
        return self._eval(nH, self.ebins)

    @staticmethod
    def _eval(nH, ebins):
        abs_coef = []
        callModFunc('wabs', ebins, (nH,), abs_coef, [], 1, '')
        return np.asarray(abs_coef)


if __name__ == '__main__':
    import arviz as az
    import corner
    import emcee

    path = '/Users/xuewc/BurstData/FRB221014/HXMT/'

    LE = data_for_wstat(emin=1.0,
                        emax=11.0,
                        spec_on=f'{path}/LE_bmin5.grp',
                        spec_off=f'{path}/LE_phabkg20s_g0_0-94.pha',
                        respfile=f'{path}/LE_rsp.rsp')

    ME = data_for_wstat(emin=8,
                        emax=35,
                        spec_on=f'{path}/ME_bmin5.grp',
                        spec_off=f'{path}/ME_phabkg20s_g0_0-53.pha',
                        respfile=f'{path}/ME_rsp.rsp')

    HE = data_for_wstat(emin=18,
                        emax=250,
                        spec_on=f'{path}/HE_bmin5.grp',
                        spec_off=f'{path}/HE_phabkg20s_g0_0-12.pha',
                        respfile=f'{path}/HE_rsp.rsp')

    WABS_LE = WAbs(LE['Eph_bins'])(2.79)
    WABS_ME = WAbs(ME['Eph_bins'])(2.79)
    WABS_HE = WAbs(HE['Eph_bins'])(2.79)

    #%% Models
    PL_LE = PowerLaw(LE['Eph_bins'])
    PL_ME = PowerLaw(ME['Eph_bins'])
    PL_HE = PowerLaw(HE['Eph_bins'])

    CPL_LE = CutoffPowerLaw(LE['Eph_bins'])
    CPL_ME = CutoffPowerLaw(ME['Eph_bins'])
    CPL_HE = CutoffPowerLaw(HE['Eph_bins'])

    BB_LE = BBodyRad(LE['Eph_bins'])
    BB_ME = BBodyRad(ME['Eph_bins'])
    BB_HE = BBodyRad(HE['Eph_bins'])

    # PL
    bounds = [
        (0, 5),
        (0, 3),
    ]
    src_LE = lambda p: (p[1] * WABS_LE * PL_LE(p[0])) @ LE['DRM']
    src_ME = lambda p: (p[1] * WABS_ME * PL_ME(p[0])) @ ME['DRM']
    src_HE = lambda p: (p[1] * WABS_HE * PL_HE(p[0])) @ HE['DRM']

    # PL+BB
    # bounds = [
    #     (0, 4),
    #     (0, 10),
    #     (0, 2),
    #     (0, 2000),
    # ]
    # src_LE = lambda p: (WABS_LE * (p[2]*PL_LE(p[0]) + p[3]*BB_LE(p[1]))) @ LE['DRM']
    # src_ME = lambda p: (WABS_ME * (p[2]*PL_ME(p[0]) + p[3]*BB_ME(p[1]))) @ ME['DRM']
    # src_HE = lambda p: (WABS_HE * (p[2]*PL_HE(p[0]) + p[3]*BB_HE(p[1]))) @ HE['DRM']

    # CPL
    # bounds = [
    #     (0, 10),
    #     (0, 1000),
    #     (0, 2),
    # ]
    # src_LE = lambda p: (p[2] * WABS_LE * CPL_LE(p[0], p[1])) @ LE['DRM']
    # src_ME = lambda p: (p[2] * WABS_ME * CPL_ME(p[0], p[1])) @ ME['DRM']
    # src_HE = lambda p: (p[2] * WABS_HE * CPL_HE(p[0], p[1])) @ HE['DRM']

    # BB+BB
    # bounds = [
    #     (0, 3.5),
    #     (3, 30),
    #     (1, 90),
    #     (0, 0.5),
    # ]
    # src_LE = lambda p: (WABS_LE * (p[2]*BB_LE(p[0]) + p[3]*BB_LE(p[0]+p[1]))) @ LE['DRM']
    # src_ME = lambda p: (WABS_ME * (p[2]*BB_ME(p[0]) + p[3]*BB_ME(p[0]+p[1]))) @ ME['DRM']
    # src_HE = lambda p: (WABS_HE * (p[2]*BB_HE(p[0]) + p[3]*BB_HE(p[0]+p[1]))) @ HE['DRM']

    def lnprob(p, n_on1, n_on2, n_on3, n_off1, n_off2, n_off3, beta):
        for pi, bi in zip(p, bounds):
            if pi < bi[0] or pi > bi[1]:
                logprob = -np.inf
                pwll1 = np.full(len(n_on1), np.nan)
                pwll2 = np.full(len(n_on2), np.nan)
                pwll3 = np.full(len(n_on3), np.nan)
                pp1 = np.full(len(n_on1), np.nan)
                pp2 = np.full(len(n_on2), np.nan)
                pp3 = np.full(len(n_on3), np.nan)
                return logprob, pwll1, pwll2, pwll3, pp1, pp1, pp2, pp2, pp3, pp3

        # lnprob = 0.0 #- np.log(p[0])
        src1 = src_LE(p)
        src2 = src_ME(p)
        src3 = src_HE(p)
        loglike1, bkg1 = wstat(src1, LE['N_on'], LE['N_off'], LE['T_on'], LE['T_off'])
        loglike2, bkg2 = wstat(src2, ME['N_on'], ME['N_off'], ME['T_on'], ME['T_off'])
        loglike3, bkg3 = wstat(src3, HE['N_on'], HE['N_off'], HE['T_on'], HE['T_off'])

        logprob = -beta*(loglike1.sum()+loglike2.sum()+loglike3.sum())# - np.log(p[1])

        # pointwise_loglike = np.hstack((loglike1, loglike2, loglike3))

        pp1_on = np.random.poisson((src1+bkg1)*LE['T_on'])
        pp2_on = np.random.poisson((src2+bkg2)*ME['T_on'])
        pp3_on = np.random.poisson((src3+bkg3)*HE['T_on'])
        # pp_on = np.hstack((pp1_on, pp2_on, pp3_on))

        pp1_off = np.random.poisson(bkg1*LE['T_off'])
        pp2_off = np.random.poisson(bkg2*ME['T_off'])
        pp3_off = np.random.poisson(bkg3*HE['T_off'])
        # pp_off = np.hstack((pp1_off, pp2_off, pp3_off))

        return logprob, -loglike1, -loglike2, -loglike3, pp1_on, pp2_on, pp3_on, pp1_off, pp2_off, pp3_off


    sample_size = 20000*30
    ndim = len(bounds)
    nwalker = ndim * 10
    init = np.random.uniform(*np.array(bounds).T, (nwalker, ndim))
    # init[:,-1] = 1.14e-2 * np.random.uniform(0, 2, nwalker)

    burn = 0.6
    thin = 2
    steps = round(sample_size * thin / (1 - burn) / nwalker)
    burn = round(burn * steps)
    sampler = emcee.EnsembleSampler(
        nwalker, ndim, lnprob,
        # moves=[
        #     (emcee.moves.StretchMove(), 0.45),
        #     (emcee.moves.DEMove(), 0.45),
        #     (emcee.moves.DESnookerMove(), 0.1),
        # ],
        args=(LE['N_on'], ME['N_on'], HE['N_on'],
              LE['N_off'], ME['N_off'], HE['N_off'],
              [1.0, 1.0 / np.log(len(LE['N_on'])+len(ME['N_on'])+len(HE['N_on']))][0]
              ),
        blobs_dtype=[('LE:loglike', np.ndarray),
                     ('ME:loglike', np.ndarray),
                     ('HE:loglike', np.ndarray),
                     ('LE:N_on_ppc', np.ndarray),
                     ('ME:N_on_ppc', np.ndarray),
                     ('HE:N_on_ppc', np.ndarray),
                     ('LE:N_off_ppc', np.ndarray),
                     ('ME:N_off_ppc', np.ndarray),
                     ('HE:N_off_ppc', np.ndarray)],

    )
    pos, _blob, prob, state = sampler.run_mcmc(init, steps, progress=True)

    dims = {'PhoIndex': ['chain', 'draw'],
            'norm': ['chain', 'draw'],
            'alpha': ['chain', 'draw'],
            'kT': ['chain', 'draw'],
            'detla_kT': ['chain', 'draw'],
            'LE:N_on': ['le_channel'],
            'ME:N_on': ['me_channel'],
            'HE:N_on': ['he_channel'],
            'LE:N_off': ['le_channel'],
            'ME:N_off': ['me_channel'],
            'HE:N_off': ['he_channel'],
            'LE:loglike': ['chain', 'draw', 'le_channel'],
            'ME:loglike': ['chain', 'draw', 'me_channel'],
            'HE:loglike': ['chain', 'draw', 'he_channel'],
            'LE:N_on_ppc': ['chain', 'draw', 'le_channel'],
            'ME:N_on_ppc': ['chain', 'draw', 'me_channel'],
            'HE:N_on_ppc': ['chain', 'draw', 'he_channel'],
            'LE:N_off_ppc': ['chain', 'draw', 'le_channel'],
            'ME:N_off_ppc': ['chain', 'draw', 'me_channel'],
            'HE:N_off_ppc': ['chain', 'draw', 'he_channel'],
            }
    coords = {'chain': range(nwalker),
              'draw': range(steps),
              'le_channel': range(len(LE['N_on'])),
              'me_channel': range(len(ME['N_on'])),
              'he_channel': range(len(HE['N_on'])),
              'all_channel': range(len(LE['N_on'])+len(ME['N_on'])+len(HE['N_on']))
              }
    idata_raw = az.from_emcee(
        sampler,
        var_names=[['PhoIndex', 'norm'],['kT', 'Delta kT', 'norm1', 'norm2'],['alpha', 'Ecut', 'norm'],["PhoIndex", "kT", "norm_PL", "norm_BB"]][0],
        arg_names=["LE:N_on", "ME:N_on", "HE:N_on", "LE:N_off", "ME:N_off", "HE:N_off", "beta"],
        arg_groups=["observed_data"] * 6 + ['constant_data'],
        # blob_names=["LE:loglike", "ME:loglike", "HE:loglike",
        #             "LE:N_on_ppc", "LE:N_off_ppc", "ME:N_on_ppc",
        #             "ME:N_off_ppc", "HE:N_on_ppc", "HE:N_off_ppc"],
        # blob_groups=["log_likelihood"]*3 + ["posterior_predictive"]*6,
        dims=dims,
        coords=coords,
    )
    az.plot_trace(idata_raw)
    #%%
    import xarray as xr
    blobs = sampler.get_blobs()
    format_data = lambda v: np.ascontiguousarray(np.transpose(np.array(v.tolist(), float), (1,0,2)))
    leloglike=format_data(blobs['LE:loglike'])
    meloglike=format_data(blobs['ME:loglike'])
    heloglike=format_data(blobs['HE:loglike'])
    jointloglike=np.concatenate((leloglike,meloglike,heloglike),axis=-1)
    pwll = xr.Dataset(
        data_vars={
            'LE:loglike': (['chain', 'draw', 'le_channel'], leloglike),
            'ME:loglike': (['chain', 'draw', 'me_channel'], meloglike),
            'HE:loglike': (['chain', 'draw', 'he_channel'], heloglike),
            'loglike': (['chain', 'draw', 'all_channel'], jointloglike),
        },
        coords=coords
    )
    ppc = xr.Dataset(
        data_vars={
            'LE:N_on': (['chain', 'draw', 'le_channel'], format_data(blobs['LE:N_on_ppc'])),
            'ME:N_on': (['chain', 'draw', 'me_channel'], format_data(blobs['ME:N_on_ppc'])),
            'HE:N_on': (['chain', 'draw', 'he_channel'], format_data(blobs['HE:N_on_ppc'])),
            'LE:N_off': (['chain', 'draw', 'le_channel'], format_data(blobs['LE:N_off_ppc'])),
            'ME:N_off': (['chain', 'draw', 'me_channel'], format_data(blobs['ME:N_off_ppc'])),
            'HE:N_off': (['chain', 'draw', 'he_channel'], format_data(blobs['HE:N_off_ppc'])),
        },
        coords=coords
    )
    idata_raw = az.from_emcee(
        sampler,
        var_names=["PhoIndex", "norm"],
        arg_names=["LE:N_on", "ME:N_on", "HE:N_on", "LE:N_off", "ME:N_off", "HE:N_off", "beta"],
        arg_groups=["observed_data"] * 6 + ['constant_data'],
        # blob_names=["LE:loglike", "ME:loglike", "HE:loglike",
        #             "LE:N_on_ppc", "LE:N_off_ppc", "ME:N_on_ppc",
        #             "ME:N_off_ppc", "HE:N_on_ppc", "HE:N_off_ppc"],
        # blob_groups=["log_likelihood"]*3 + ["posterior_predictive"]*6,
        dims=dims,
        coords=coords,
    )
    idata_raw.add_groups({"log_likelihood": pwll})
    idata_raw.add_groups({"posterior_predictive": ppc})

    idata = idata_raw.sel(draw=slice(burn,None,thin))
    az.plot_trace(idata)

    fig = corner.corner(
        data=idata,
        # labels=['$\log A$', r'$\gamma$', '$\mathcal{F}$'],
        label_kwargs={'fontsize': 8},
        quantiles=[0.15865, 0.5, 0.84135],
        levels=[[0.683, 0.954, 0.997],[0.683, 0.95]][1],
        show_titles=True,
        title_fmt='.2f',
        color='#0C5DA5',
        smooth=0.5,
        # range=((0,130),(-1.6,-2.7),(0,3.1e-7))[1:],
        # truths=(*res[0], flux_map),
        # truth_color='red',
        max_n_ticks=5,
        hist_bin_factor=2
    )

    argmax = idata_raw.log_likelihood['loglike'].sum('all_channel').argmax(...)
    p_MAP = idata_raw.posterior.isel(argmax)
    p_MAP = np.array([p_MAP[i].to_numpy() for i in p_MAP])

    # az.waic(idata, var_name='loglike', scale='deviance')
    # -2 * idata.log_likelihood.loglike.sum(dim='all_channel').mean()



    #%% PPC
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1, sharex=True, height_ratios=[1,0.618])
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.align_ylabels(axes)
    le_Ech = LE['Ech_bins'].mean(-1)
    le_Ech_bins = np.append(LE['Ech_bins'][:,0], LE['Ech_bins'][-1,-1])
    le_Ech_width = np.squeeze(np.diff(LE['Ech_bins'], axis=-1))
    le_net = (LE['N_on']/LE['T_on'] - LE['N_off']/LE['T_off']) / le_Ech_width
    # plt.errorbar(LE['Ech_bins'].mean(-1), le_net, xerr=le_Ech_width/2, fmt=' ')
    # plt.step(le_Ech_bins, np.append(le_net, le_net[-1]), where='post', alpha=0.5)
    axes[0].scatter(LE['Ech_bins'].mean(-1), le_net, s=2, c='k')

    me_Ech = ME['Ech_bins'].mean(-1)
    me_Ech_bins = np.append(ME['Ech_bins'][:,0], ME['Ech_bins'][-1,-1])
    me_Ech_width = np.squeeze(np.diff(ME['Ech_bins'], axis=-1))
    me_net = (ME['N_on']/ME['T_on'] - ME['N_off']/ME['T_off']) / me_Ech_width
    # plt.errorbar(ME['Ech_bins'].mean(-1), me_net, xerr=me_Ech_width/2, fmt=' ')
    # plt.step(me_Ech_bins, np.append(me_net, me_net[-1]), where='post', alpha=0.5)
    axes[0].scatter(ME['Ech_bins'].mean(-1), me_net, s=2, c='k')

    he_Ech = HE['Ech_bins'].mean(-1)
    he_Ech_bins = np.append(HE['Ech_bins'][:,0], HE['Ech_bins'][-1,-1])
    he_Ech_width = np.squeeze(np.diff(HE['Ech_bins'], axis=-1))
    he_net = (HE['N_on']/HE['T_on'] - HE['N_off']/HE['T_off']) / he_Ech_width
    # plt.errorbar(HE['Ech_bins'].mean(-1), he_net, xerr=he_Ech_width/2, fmt=' ')
    # plt.step(he_Ech_bins, np.append(he_net, he_net[-1]), where='post', alpha=0.5)
    axes[0].scatter(HE['Ech_bins'].mean(-1), he_net, s=2, c='k')

    le_ppc=(
        (idata.posterior_predictive['LE:N_on']/LE['T_on']
        - idata.posterior_predictive['LE:N_off']/LE['T_off']) / le_Ech_width
    ).to_numpy().reshape(-1, LE['N_on'].size)
    me_ppc=(
        (idata.posterior_predictive['ME:N_on']/ME['T_on']
        - idata.posterior_predictive['ME:N_off']/ME['T_off']) / me_Ech_width
    ).to_numpy().reshape(-1, ME['N_on'].size)
    he_ppc=(
        (idata.posterior_predictive['HE:N_on']/HE['T_on']
        - idata.posterior_predictive['HE:N_off']/HE['T_off']) / he_Ech_width
    ).to_numpy().reshape(-1, HE['N_on'].size)

    nppc=1000
    idx = np.random.randint(0,sample_size,nppc)
    for i in range(nppc):
        axes[0].errorbar(le_Ech, le_ppc[idx[i]], c='tab:blue', xerr=le_Ech_width/2, alpha=0.01, fmt=' ')
        axes[0].errorbar(me_Ech, me_ppc[idx[i]], c='tab:orange', xerr=me_Ech_width/2, alpha=0.01, fmt=' ')
        axes[0].errorbar(he_Ech, he_ppc[idx[i]], c='tab:green', xerr=he_Ech_width/2, alpha=0.01, fmt=' ')

    ymin = np.min((min(le_net), min(me_net), min(he_net)))
    ymax = np.max((max(le_net), max(me_net), max(he_net)))
    [axes[0].semilogx(), axes[0].set_ylim(ymin*1.1, ymax*1.1)]
    # axes[0].loglog()

    # le_resd = (le_ppc.mean() - le_net) / np.std(le_ppc, axis=0, ddof=1)
    # me_resd = (me_ppc.mean() - me_net) / np.std(me_ppc, axis=0, ddof=1)
    # he_resd = (he_ppc.mean() - he_net) / np.std(he_ppc, axis=0, ddof=1)
    # plt.errorbar(le_Ech, le_resd, 1.0, le_Ech_width/2, fmt=' ')
    # plt.errorbar(me_Ech, me_resd, 1.0, me_Ech_width/2, fmt=' ')
    # plt.errorbar(he_Ech, he_resd, 1.0, he_Ech_width/2, fmt=' ')
    # plt.axhline(0, ls=':', zorder=0)
    # plt.semilogx()
    LE_best = src_LE(p_MAP)/le_Ech_width
    ME_best = src_ME(p_MAP)/me_Ech_width
    HE_best = src_HE(p_MAP)/he_Ech_width
    axes[0].step(le_Ech_bins, np.append(LE_best, LE_best[-1]), where='post')
    axes[0].step(me_Ech_bins, np.append(ME_best, ME_best[-1]), where='post')
    axes[0].step(he_Ech_bins, np.append(HE_best, HE_best[-1]), where='post')
    le_resd = (le_ppc - le_net) / np.std(le_ppc, axis=0, ddof=1)
    me_resd = (me_ppc - me_net) / np.std(me_ppc, axis=0, ddof=1)
    he_resd = (he_ppc - he_net) / np.std(he_ppc, axis=0, ddof=1)
    for i in range(nppc):
        axes[1].errorbar(le_Ech, le_resd[i], c='tab:blue', xerr=le_Ech_width/2, alpha=0.01, fmt=' ')
        axes[1].errorbar(me_Ech, me_resd[i], c='tab:orange', xerr=me_Ech_width/2, alpha=0.01, fmt=' ')
        axes[1].errorbar(he_Ech, he_resd[i], c='tab:green', xerr=he_Ech_width/2, alpha=0.01, fmt=' ')
    # plt.semilogx()
    plt.axhline(0, c='k', ls=':', zorder=0)
    axes[1].set_xlim(1.6, 250)
    axes[1].set_ylim(-5,5)
    axes[0].set_title('WABS*PL')
    axes[0].set_ylabel(r'$D_{\rm obs}$ [s$^{-1}$ keV$^{-1}$]')
    axes[1].set_ylabel(r'$D_{\rm rep}$ - $D_{\rm obs}$ / Var$^{1/2}$($D_{\rm rep}$)')
    axes[1].set_xlabel('Energy [keV]')
    #%%
    from scipy.stats import gaussian_kde
    le_resd = (le_ppc - le_net) / np.std(le_ppc, axis=0, ddof=1)
    me_resd = (me_ppc - me_net) / np.std(me_ppc, axis=0, ddof=1)
    he_resd = (he_ppc - he_net) / np.std(he_ppc, axis=0, ddof=1)
    resd = np.concatenate((le_resd, me_resd, he_resd),axis=-1)
    y = np.linspace(-5, 5, 101)
    dy = 1#np.diff(y).mean()
    ymid = (y[:-1]+y[1:])/2
    le_resd_kde = np.array([gaussian_kde(le_resd[:, i])(ymid)*dy for i in range(le_resd.shape[1])])
    me_resd_kde = np.array([gaussian_kde(me_resd[:, i])(ymid)*dy for i in range(me_resd.shape[1])])
    he_resd_kde = np.array([gaussian_kde(he_resd[:, i])(ymid)*dy for i in range(he_resd.shape[1])])
    le_resd_kde[le_resd_kde<0.001]=np.nan
    me_resd_kde[me_resd_kde<0.001]=np.nan
    he_resd_kde[he_resd_kde<0.001]=np.nan
    plt.figure()
    X, Y = np.meshgrid(le_Ech_bins, y)
    plt.pcolormesh(X, Y, le_resd_kde.T, cmap='Blues', alpha=0.5, zorder=10)
    X, Y = np.meshgrid(me_Ech_bins, y)
    plt.pcolormesh(X, Y, me_resd_kde.T, cmap='Oranges', alpha=0.5, zorder=10)
    X, Y = np.meshgrid(he_Ech_bins, y)
    plt.pcolormesh(X, Y, he_resd_kde.T, cmap='Greens', alpha=0.5, zorder=10)
    plt.semilogx()
    plt.axhline(0, c='k', ls=':', zorder=0)






