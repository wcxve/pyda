# -*- coding: utf-8 -*-
"""
@author: xuewc
"""

import matplotlib.pyplot as plt
import numpy as np
import pycwt

from scipy.stats import chi2
from stingray import Lightcurve, Powerspectrum
from tqdm import tqdm

def calc_mvts(
    t, counts, back_rate=0.0, nsim=1000, exposure=None, confidence=0.99, dj=1/4,
    plot_fig=(0, 1), mult_hypo_corr='n', wavelet='morlet', sci_nota=0,
    title=None
):
    # TODO: exposure correction
    t = np.asarray(t)
    counts = np.asarray(counts)

    N = t.size
    N_ph = np.sum(counts)

    if wavelet.lower().startswith('morl'):
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

    period = 1 / freqs

    cwt_power = np.abs(wave) ** 2
    # use Leahy normalization for global power
    cwt_power_global = np.sum(cwt_power, axis=1) * (2 / N_ph)

    # alpha from confidence
    if mult_hypo_corr.lower().startswith('n'): # no correction
        alpha = 1 - confidence
    elif mult_hypo_corr.lower().startswith('s'): # Sidak correction
        alpha = 1 - confidence ** (1 / scales.size)
    elif mult_hypo_corr.lower().startswith('b'): # Bonferroni correction
        alpha = (1 - confidence) / scales.size
    else:
        raise ValueError(f'wrong input for `mult_hypo_corr` ({mult_hypo_corr})')
    p = np.array([alpha/2, 0.5, 1 - alpha/2])


    # >>> mvt by using chi2 as reference distribution >>>
    # dof is reduced by approximately one-half of the number within the COI
    na = N - np.sum(period[:,None] >= coi, axis=1) / 2

    # calculate edof
    dofmin = mother.dofmin
    gamma = mother.gamma
    edof = dofmin * np.sqrt(1 + (na * dt / (gamma * scales)) ** 2)
    edof[edof < dofmin] = dofmin

    # factor 2 is for Leahy normalization
    upper1, median1, lower1 = edof / chi2.ppf(p[:, None], edof) * 2

    mvt1 = np.min(period[cwt_power_global > upper1])
    if sci_nota:
        mvt_s1, mvt_s2 = f'{mvt1:.2e}'.split('e')
        mvt_s2 = int(mvt_s2)
        mvt1_str = r'MVT$=%s\times10^{%d}\,$s' % (mvt_s1, mvt_s2)
    else:
        mvt1_str = 'MVT$=%.3f\,$s' % mvt1
    # <<< mvt using chi2 as reference distribution <<<

    # >>> mvt by simulation
    sim_flag = back_rate > 0.0 and nsim > 0
    if sim_flag:
        sj = s0 * 2 ** (np.arange(0, J + 1) * dj)
        sj_col = sj[:, None]
        ftfreqs = 2 * np.pi * np.fft.fftfreq(N, dt)
        psi_ft_bar = ((sj_col * ftfreqs[1] * N) ** 0.5 *
                      np.conjugate(mother.psi_ft(sj_col * ftfreqs)))

        print('Generating background signals...')
        back_sim = np.random.poisson(back_rate*dt, size=(nsim, N))
        zero_mask = back_sim.sum(1) == 0.0
        nleft = zero_mask.sum()
        while nleft:
            print(f'{nsim - nleft}/{nsim}')
            sim = np.random.poisson(back_rate*dt, size=(nleft, N))
            valid_mask = sim.sum(1) != 0.0
            valid_n = valid_mask.sum()
            if valid_n == 0:
                continue
            fill_mask = np.full_like(zero_mask, False)
            fill_mask[np.where(zero_mask)[0][:valid_n]] = True
            back_sim[fill_mask] = sim[valid_mask]
            zero_mask[fill_mask] = False
            nleft -= valid_n
        print(f'{nsim - nleft}/{nsim}')

        back_power = np.empty((nsim, J))
        for i in tqdm(range(nsim), desc='CWT: '):
            # exclude Nyquist frequency
            back_power[i] = _cwt(back_sim[i], psi_ft_bar)[1:]

        lower2, median2, upper2 = np.quantile(back_power, p, axis=0)

        mvt2 = np.min(period[cwt_power_global > upper2])
        if sci_nota:
            mvt_s1, mvt_s2 = f'{mvt2:.2e}'.split('e')
            mvt_s2 = int(mvt_s2)
            mvt2_str = r'MVT$=%s\times10^{%d}\,$s' % (mvt_s1, mvt_s2)
        else:
            mvt2_str = 'MVT$=%.3f\,$s' % mvt2

    # plot power
    if plot_fig[0]:
        ps = Powerspectrum(Lightcurve(t, counts), norm="leahy").rebin_log(0.5)
        plt.figure()
        plt.step(1/ps.freq, ps.power, c='g', where='mid', zorder=9, label='FFT')
        plt.plot(period, cwt_power_global, 'o-', ms=3, zorder=10, label='CWT')
        plt.plot(period, median1, '--', c='tab:blue')#, label=r'Median ($\chi^2$)')
        plt.fill_between(
            period, lower1, upper1, alpha=0.5,
            label=r'$\chi^2$'
        )
        plt.axvline(mvt1, ymax=0.5, c='tab:blue', ls=':', label=mvt1_str, zorder=9.9)
        if sim_flag:
            plt.plot(period, median2, '--', c='tab:orange')#, label='Median (sim)')
            plt.fill_between(
                period, lower2, upper2, alpha=0.5,
                label=r'Sim'
                )
            plt.axvline(mvt2, ymax=0.5, c='tab:orange', ls=':', label=mvt2_str, zorder=9.9)
        # plt.axhline(2, c='#00FF00', ls='--', zorder=0)
        plt.legend(framealpha=0)
        plt.title(title)
        plt.xlabel('$\delta t$ [s]')
        plt.ylabel('Leahy Power')
        plt.loglog()
        plt.xlim(period.min(), period.max())
        plt.ylim(cwt_power_global.min()/2, cwt_power_global.max()*2)
        # plt.gca().set_aspect('equal')
        plt.show()

    # plot power with scale correction
    if plot_fig[1]:
        plt.figure()
        power_scale = cwt_power_global / scales
        plt.plot(period, power_scale, 'o-', ms=3, zorder=10, label='CWT')
        plt.plot(period, median1/scales, '--', c='tab:blue')#, label=r'Median ($\chi^2$)')
        plt.fill_between(
            period, lower1/scales, upper1/scales, alpha=0.5,
            label=r'$\chi^2$'
        )
        plt.axvline(mvt1, ymax=0.5, ls=':', c='tab:blue', label=mvt1_str)
        if sim_flag:
            plt.plot(period, median2/scales, '--', c='tab:orange')#, label=r'Median (Sim)')
            plt.fill_between(
                period, lower2/scales, upper2/scales, alpha=0.5,
                label='Sim'
            )
            plt.axvline(mvt2, ymax=0.5, ls=':', c='tab:orange', label=mvt2_str)
        plt.legend(framealpha=0)
        plt.title(title)
        plt.xlabel('$\delta t$ [s]')
        plt.ylabel('Power')
        plt.loglog()
        plt.xlim(period.min(), period.max())
        plt.ylim(power_scale.min()/2, power_scale.max()*2)
        # plt.gca().set_aspect('equal')
        plt.show()

def _cwt(signal, psi_ft_bar):
    if signal.sum() == 0:
        raise ValueError('zero input!')

    N_ph = signal.sum()

    signal_ft = np.fft.fft(signal, n=signal.size)

    W = np.fft.ifft(signal_ft * psi_ft_bar, axis=1, n=signal_ft.size)

    # Checks for NaN in transform results
    if np.any(np.isnan(W)):
        return np.nan

    W = np.abs(W)

    return (W*W).sum(1) * (2 / N_ph)

if __name__ == '__main__':
    Fs = 10**-5
    T = 1
    dt = T*Fs
    t = np.arange(0, T, dt)
    comp1 = np.cos(2*np.pi*200*t)*(t>0.6)
    comp2 = np.cos(2*np.pi*10*t)*(t<0.4)
    trend = np.sin(2*np.pi*1/2*t)
    X = comp1 + comp2 + trend
    lc = np.random.poisson(X - X.min())
    # calc_mvts(t, lc, title='DOG(2)', wavelet='mexicanhat', confidence=0.99, back_rate=1)
    cwt_power_global, back_power = calc_mvts(
        t, lc, title='morlet', wavelet='morlet', confidence=0.99, back_rate=1000,
        nsim=500, plot_fig=(1,1)
    )

