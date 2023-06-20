# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:18:20 2023

@author: Wang-Chen Xue < https://orcid.org/0000-0001-8664-5085 >
"""

import matplotlib.pyplot as plt
import numpy as np
import pycwt

from stingray import Lightcurve, Powerspectrum


def fft_power(t, counts):
    len_t = len(t)
    len_s = len(counts)
    if len_t != len_s:
        raise ValueError(
            f'length of `t` ({len_t}) and `counts` ({len_s}) should be matched'
        )

    t = np.ascontiguousarray(t)
    counts = np.ascontiguousarray(counts)

    dt = np.diff(t)
    if np.any(np.abs(np.diff(dt)) > 1e-8):
        raise ValueError('`t` should be evenly spaced')
    else:
        dt = dt.mean()

    freqs = np.fft.rfftfreq(t.size, dt)
    mod = np.abs(np.fft.rfft(counts))
    power = 2.0*mod*mod/counts.sum()

    # exclude DC component and Nyquist frequency
    if freqs[-1] >= 0.5/dt:
        freqs = freqs[1:-1]
        power = power[1:-1]
    else:
        freqs = freqs[1:]
        power = power[1:]

    return freqs, power


def time_freq_power(t, counts, exposure=None, dj=1/4, norm=None, wavelet=None):
    # TODO: exposure correction
    t = np.asarray(t)
    counts = np.asarray(counts)

    N = t.size
    N_ph = np.sum(counts)

    if wavelet is None or wavelet.lower().startswith('morl'):
        mother = pycwt.Morlet()
    elif wavelet.lower().startswith('mexi'):
        mother = pycwt.MexicanHat()
    elif wavelet.lower().startswith('paul'):
        mother = pycwt.Paul()
    elif wavelet.lower().startswith('dog'):
        if wavelet.lower() == 'dog' or wavelet[3] == '2':
            mother = pycwt.DOG(2)
        elif wavelet[3] == '6':
            mother = pycwt.DOG(6)
        else:
            raise ValueError('DOG2 and DOG6 supported only')
    else:
        raise ValueError(f'{wavelet} not supported')

    dt = t[1] - t[0]
    s0 = 2*dt / mother.flambda()
    J = round((np.log2(N)-1) / dj)
    cwt_result = pycwt.cwt(counts, dt, dj, s0, J, mother)
    wave, scales, freqs, coi, fftcoefs, fftfreqs = cwt_result

    # exclude Nyquist frequency, which is at index=0
    wave = wave[1:]
    scales = scales[1:]
    freqs = freqs[1:]
    return wave, freqs, coi

    period = 1 / freqs

    cwt_power = np.abs(wave) ** 2
    # use Leahy normalization for global power
    cwt_power_global = np.sum(cwt_power, axis=1) * (2 / N_ph)

    # plot power
    ps = Powerspectrum(Lightcurve(t, counts), norm="leahy").rebin_log(0.5)
    plt.figure()
    plt.step(1/ps.freq, ps.power, c='g', where='mid', zorder=9, label='FFT')
    plt.plot(period, cwt_power_global, 'o-', ms=3, zorder=10, label='CWT')
    # plt.axhline(2, c='#00FF00', ls='--', zorder=0)
    plt.legend(framealpha=0)
    plt.xlabel('$\delta t$ [s]')
    plt.ylabel('Leahy Power')
    plt.loglog()
    plt.xlim(period.min(), period.max())
    plt.ylim(cwt_power_global.min()/2, cwt_power_global.max()*2)
    # plt.gca().set_aspect('equal')
    plt.show()

    # plot power with scale correction
    # plt.figure()
    # power_scale = cwt_power_global / scales
    # plt.plot(period, power_scale, 'o-', ms=3, zorder=10, label='CWT')
    # plt.legend(framealpha=0)
    # plt.xlabel('$\delta t$ [s]')
    # plt.ylabel('Power')
    # plt.loglog()
    # plt.xlim(period.min(), period.max())
    # plt.ylim(power_scale.min()/2, power_scale.max()*2)
    # # plt.gca().set_aspect('equal')
    # plt.show()


if __name__ == '__main__':
    # 截取数据
    from astropy.io import fits
    evt_file = '/Users/xuewc/BurstData/GRB230307A/gbg_evt_230307_15_v01.fits'
    t0 = 131903046.67
    tstart = -1
    tstop = 40
    dets=[1,3,4,5]
    with fits.open(evt_file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        data = []
        data = hdul[f'EVENTS{str(dets[0]).zfill(2)}'].data
        for i in dets[1:]:
            data = np.append(data, hdul[f'EVENTS{str(i).zfill(2)}'].data)
        data = fits.FITS_rec(data)

    data = data[data['GAIN_TYPE']==0]
    tmask = ((t0 + tstart) <= data['TIME']) \
                    & (data['TIME'] <= t0 + tstop)
    data = data[tmask]
    emin = 8
    emax = 400
    emask = (emin <= ebounds['E_MIN']) & (ebounds['E_MAX'] <= emax)
    chmin = ebounds['CHANNEL'][emask].min()
    chmax = ebounds['CHANNEL'][emask].max()
    chmask = (chmin <= data['PI']) & (data['PI'] <= chmax)
    data_HG = data[chmask]


    with fits.open(evt_file) as hdul:
        ebounds = hdul['EBOUNDS'].data
        data = []
        data = hdul[f'EVENTS{str(dets[0]).zfill(2)}'].data
        for i in dets[1:]:
            data = np.append(data, hdul[f'EVENTS{str(i).zfill(2)}'].data)
        data = fits.FITS_rec(data)

    data = data[data['GAIN_TYPE']==1]
    tmask = ((t0 + tstart) <= data['TIME']) \
                    & (data['TIME'] <= t0 + tstop)
    data = data[tmask]
    emin = 400
    emax = 5000
    emask = (emin <= ebounds['E_MIN']) & (ebounds['E_MAX'] <= emax)
    chmin = ebounds['CHANNEL'][emask].min()
    chmax = ebounds['CHANNEL'][emask].max()
    chmask = (chmin <= data['PI']) & (data['PI'] <= chmax)
    data_LG = data[chmask]


    data = np.append(data_HG, data_LG)
    data = fits.FITS_rec(data)
    #%% PSD画图
    plt.figure()
    tbins = np.linspace(tstart, tstop,
                       2**int(np.ceil(np.log2((tstop-tstart)/1e-4)))+1)
    t = (tbins[:-1] + tbins[1:])/2.0
    lc = np.histogram(data['TIME'], bins=tbins+t0)[0]
    wave, freqs, coi = time_freq_power(t, lc)
    power = np.abs(wave) ** 2 * (2/lc.sum())
    ps = Powerspectrum(Lightcurve(t, lc), norm="leahy").rebin_log(0.05)
    plt.plot(ps.freq, ps.power)
    plt.plot(freqs, power.sum(-1))
    plt.loglog()
    #%%
    # fit ps
    from numba import njit, guvectorize, prange
    @guvectorize(
        ['void(float64,float64,float64,float64[:],float64[:])'],
        '(),(),(),(n)->(n)',
        nopython=True,
        target='parallel'
    )
    def pl(gamma, norm, const, freqs, powers):
        index = -gamma
        for i in prange(len(freqs)):
            powers[i] = norm * freqs[i]**index + const
    @guvectorize(
        ['void(float64,float64,float64,float64,float64,float64[:],float64[:])'],
        '(),(),(),(),(),(n)->(n)',
        nopython=True,
        target='parallel'
    )
    def bpl(gamma1, v_break, gamma2, norm, const, freqs, powers):
        index1 = -gamma1
        index2 = -gamma2
        tmp = v_break ** (gamma2 - gamma1)
        for i in prange(len(freqs)):
            freq = freqs[i]
            if freq <= v_break:
                powers[i] = norm * freq**index1 + const
            else:
                powers[i] = norm * tmp * freq**index2 + const
    @guvectorize(
        ['void(float64[:], float64[:], float64[:], float64[:])'],
        '(n),(n),(n)->()',
        nopython=True,
        target='parallel'
    )
    def ln_likelihood(model, data, nbin, res):
        r = 0.0
        for i in prange(len(model)):
            mi = model[i]
            di = data[i]
            ni = nbin[i]
            r -= ni*(np.log(mi) + di/mi)
        res[0] = r

    @njit
    def get_mask(pars, bounds):
        mask = np.full(len(pars), True, dtype=np.bool_)
        for i in range(len(bounds)):
            mask &= (bounds[i, 0] <= pars[:, i]) & (pars[:, i] <= bounds[i, 1])
        return mask

    def lnprob(pars, bounds, model_func, freqs, data, nbin):
        res = np.empty(len(pars))
        mask = get_mask(pars, bounds)
        model = model_func(*pars[mask].T, freqs)
        res[~mask] = -np.inf
        res[mask] = ln_likelihood(model, data, nbin)
        return res

    gamma1_lim = (-3, 10)
    v_break_lim = (0.03, 14)
    gamma2_lim = (0, 20)
    norm_lim = (0, 50000)
    const_lim = (0, 5)
    e_lim = (0, 80)
    sigma_lim = (0,20)
    norm2_lim = (0, 200)
    bounds = (
        gamma1_lim,
        v_break_lim,
        gamma2_lim,
        norm_lim,
        const_lim,
    )
    bounds = np.array(bounds)
    model_func = bpl
    sample_size = 50000
    ndim = len(bounds)
    nwalker = ndim * 4
    init = np.random.uniform(*bounds.T, (nwalker, ndim))
    # init[:,-1] = 1.14e-2 * np.random.uniform(0, 2, nwalker)

    burn = 0.4
    thin = 2
    steps = round(sample_size * thin / (1 - burn) / nwalker)
    burn = round(burn * steps)
    import emcee
    freq = ps.freq
    power = ps.power
    nbin = np.full_like(ps.m, freq) if type(ps.m) is int else ps.m
    sampler = emcee.EnsembleSampler(
        nwalker, ndim, lnprob,
        args=(bounds, model_func, freq, power, nbin),
        vectorize=True
    )
    pos, prob, state = sampler.run_mcmc(init, steps, progress=True)
    chain = sampler.get_chain(thin=thin, discard=burn)
    #%%
    plt.figure()
    plt.plot(ps.freq, ps.power, c='k', zorder=0)
    flatchain = chain.reshape(-1,len(bounds))
    idx = np.random.randint(0, len(flatchain), 500)
    for i in idx:
        plt.plot(ps.freq, model_func(*flatchain[i], ps.freq), c='tab:orange', alpha=0.02)
    # plt.plot(ps.freq, pl(-5/3, chain[:,:,1].mean(), chain[:,:,2].mean(), ps.freq), ls='--', zorder=10,
    #          c='tab:red',label='Kolmogorov spectral shape (-5/3) for isotropic turbulence')
    plt.title(f'PSD, from {tbins[0]:.1f} to {tbins[-1]:.1f} s')
    # plt.legend()
    plt.loglog()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Leahy Power')
    #%%
    import corner
    corner.corner(
        data=chain,
        labels=['gamma1', 'vbreak', 'gamma2', 'norm', 'wn'],
        label_kwargs={'fontsize': 8},
        quantiles=[0.15865, 0.5, 0.84135],
        levels=[[0.683, 0.954, 0.997],[0.683, 0.95]][0],
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
    #%% RMS画图
    plt.figure()
    tbins = np.linspace(tstart, tstop,
                       2**int(np.ceil(np.log2((tstop-tstart)/1e-4)))+1)
    t = (tbins[:-1] + tbins[1:])/2.0
    lc = np.histogram(data['TIME'], bins=tbins+t0)[0]
    wave, freqs, coi = time_freq_power(t, lc)
    power = np.abs(wave) ** 2 * (2*(tstop-tstart)/lc.sum()**2)
    ps = Powerspectrum(Lightcurve(t, lc), norm="frac").rebin_log(0.5)
    plt.plot(ps.freq, ps.power)
    plt.plot(freqs, power.sum(-1))
    plt.loglog()
    plt.figure()
    dt = tbins[1]-tbins[0]
    Dt = 0.1
    rebin = round(Dt/dt)
    fbins = np.append(1/dt/2, freqs)
    tbins_rebin = tbins[::rebin]
    power_rebin = np.add.reduceat(power, range(0, t.size, rebin), axis=1)
    if len(tbins_rebin) == power_rebin.shape[1]:
        tbins_rebin = np.append(tbins_rebin, tstop)

    T, F = np.meshgrid(tbins_rebin, fbins)
    plt.pcolormesh(T, F, np.log10(power_rebin), cmap='afmhot')
    plt.plot(t, 1/coi)
    plt.fill_between(t, 1/coi, facecolor='gray', edgecolor='k', hatch='x' ,alpha=0.5)
    plt.ylim(fbins.min(), fbins.max())
    plt.xlabel('$t-T_0$ [s]')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label=r'$\log_{10}$ frac RMS')
    plt.semilogy()

    rms = (-power*np.diff(fbins)[:,None]).sum(0)
    rms = np.add.reduceat(rms, range(0, t.size, rebin))
    plt.figure()
    t2 = (tbins_rebin[:-1] + tbins_rebin[1:])/2.0
    lc2 = np.histogram(data['TIME'], bins=tbins_rebin+t0)[0]
    plt.plot(t2,lc2/np.diff(tbins_rebin), alpha=0.8)
    ax=plt.gca()
    ax.set_xlabel('$t-T_0$ [s]')
    ax.set_ylabel('Rate [s$^{-1}$]')
    ax.spines['left'].set_color('tab:blue')
    ax.yaxis.label.set_color('tab:blue')
    ax.tick_params(axis='y', colors='tab:blue', which='both')
    twinx=plt.twinx(ax)
    twinx.plot(tbins_rebin[:-1], rms*100, c='tab:orange', alpha=0.8)
    twinx.set_ylabel('frac rms [%]')
    twinx.spines['right'].set_color('tab:orange')
    twinx.spines['left'].set_color('tab:blue')
    twinx.yaxis.label.set_color('tab:orange')
    twinx.tick_params(axis='y', colors='tab:orange', which='both')

    plt.figure()
    plt.scatter(lc2,rms, s=5)
    mask = (17.5<=t2) & (t2<=19)
    plt.scatter(lc2[mask],rms[mask], s=5)
    plt.xlabel('Rate [s$^{-1}$]')
    plt.ylabel('frac rms [%]')
    #%% 频率-能量的rms直方图
    import tqdm
    pi = np.unique(data['PI'])
    pi_bins = np.append(pi, pi[-1]+0.5)
    tbins = np.linspace(tstart, tstop,
                       2**int(np.ceil(np.log2((tstop-tstart)/1e-4)))+1)
    # tbins = np.linspace(17, 20,
    #                    2**int(np.ceil(np.log2((tstop-tstart)/1e-4)))+1)
    lc_pi = np.histogram2d(data['PI'], data['TIME'], bins=(pi_bins, tbins+t0))[0]
    powers = []
    fracs = lc_pi.sum(1)/lc.sum()
    i = 0
    ebins = [ebounds['E_MIN'][pi[i]]]
    for j in tqdm.tqdm(range(3, len(pi), 5)):
        ps_i = Powerspectrum(Lightcurve(t, lc_pi[i:j].sum(0)), norm="frac").rebin_log(0.5)
        powers.append(ps_i.power*fracs[i:j].sum())
        ebins.append(ebounds['E_MIN'][pi[j]])
        i=j
    powers = np.asarray(powers)
    ebins = np.asarray(ebins)
    E, F = np.meshgrid(ebins, np.append(ps_i.freq, 1/dt))

    plt.pcolormesh(E, F, np.log(powers.T))
    plt.colorbar(label='$\log_{10}$ frac rms')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Frequency [Hz]')
    plt.title(f'$T_0+${tbins[0]:.1f} to +{tbins[-1]:.1f}s')
    plt.loglog()
    #%% 光变分解
    from pyda.timing.wmtsa.MODWT import get_DS, pyramid, inv_pyramid
    X = lc
    L = 12
    J0 = round(np.log2(X.size / (L - 1) - 1)) - 1
    J = np.arange(1, J0+1)
    N = X.size
    (W, V) = pyramid(X, 'Haar', J0)
    X_rep = inv_pyramid(W, V, 'Haar', L)
    (D, S) = get_DS(X, W, 'Haar', L)
    MRA = np.row_stack((D, S[-1]))
    fig, axes = plt.subplots(len(MRA)+1, 1, sharex=True)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.align_ylabels(axes)
    axes[0].plot(t, X, 'k', label='X')
    axes[0].set_xlim(np.min(t), np.max(t))
    axes[0].set_ylabel('counts', rotation=0)
    labels = [f'D{i}' for i in range(1, L+1)] + [f'S{L}']
    Lj = [(2 ** (j + 1) - 1) * (L - 1) + 1 for j in range(L)]
    Lj = Lj + Lj[-1:]
    Lj = np.array(Lj)
    coi = np.column_stack((Lj - 2, N - Lj + 1))
    for j in range(L+1):
        axes[j+1].plot(t, MRA[j], 'k')
        axes[j+1].axvspan(t.min(), t[coi[j,0]], color='red')
        axes[j+1].axvspan(t[coi[j,1]], t.max(), color='red')
        axes[j+1].set_ylabel(labels[j], rotation=0)
    axes[0].set_title('MODWT Decomposition')
    axes[-1].set_xlabel('$t-T_0$ [s]')
