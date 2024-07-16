"""
Created at 20:32:38 on 2023-04-26

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import norm
from astropy.cosmology import Planck18, z_at_value


def significance_lima(Non, Noff, alpha):
    term1 = Non * np.log((1 + alpha)/alpha * Non/(Non + Noff))
    term2 = Noff * np.log((1 + alpha) * Noff/(Non + Noff))
    return np.where(Non >= alpha * Noff, 1, -1) * np.sqrt(2*(term1 + term2))


def significance_reduce_signal_by(factors, Non, Noff, alpha):
    factors = np.asarray(factors)
    if np.any(factors > 1.0):
        raise ValueError('`siganl_factors` must less than 1')

    alpha_ = alpha/(factors + (1 - factors)*alpha*Noff/Non)
    return significance_lima(Non, Noff, alpha_)


def find_factor_given_significance(S, Non, Noff, alpha, cl=0.0):
    S0 = significance_lima(Non, Noff, alpha)
    if np.any(S > S0):
        raise ValueError(
            f'`S` ({S:.2f}) must less than original significane ({S0:.2f})'
        )

    def obj(factor, Non, Noff, alpha, S):
        obj = significance_reduce_signal_by(factor, Non, Noff, alpha) - S
        return obj

    factor = root_scalar(obj,
                         args=(Non, Noff, alpha, S),
                         method='bisect',
                         bracket=(1e-5, 1.0)).root
    if cl > 0.0:
        delta = norm.isf((1 - cl) / 2)
        factor_lower = root_scalar(obj,
                                   args=(Non, Noff, alpha, S - delta),
                                   method='bisect',
                                   bracket=(1e-5, 1.0)).root
        factor_upper = root_scalar(obj,
                                   args=(Non, Noff, alpha, S + delta),
                                   method='bisect',
                                   bracket=(1e-5, 1.0)).root
        return np.array([factor, factor_lower, factor_upper])
    else:
        return factor


def factor_to_Ton(factor, orig_Ton, Non, Noff, alpha):
    return orig_Ton / (factor + (1 - factor)*Noff/Non*alpha)


def factor_to_distance(factor, orig_distance):
    return orig_distance / np.sqrt(factor)


def factor_to_redshift(factor, orig_redshift):
    d = Planck18.luminosity_distance(orig_redshift) / np.sqrt(factor)
    return z_at_value(Planck18.luminosity_distance, d).value


if __name__ == '__main__':
    # mu_src = 30
    # mu_bkg = 50
    # Ton = 10
    # Toff = 20
    # Non = np.random.poisson((mu_src + mu_bkg)*Ton)
    # Noff = np.random.poisson(mu_bkg*Toff)
    # alpha = Ton/Toff
    # print(significance_lima(Non, Noff, alpha))
    # factor = find_factor_given_significance(3.0, Non, Noff, alpha)
    # print(factor)
    # print(significance_reduce_signal_by(factor, Non, Noff, alpha))
    # print(factor_to_distance(factor, 1500))
    # print(factor_to_redshift(factor, 0.1))

    import matplotlib.pyplot as plt
    from pyda.utils import thist_gecam, events_gecam
    from pyda.stats.bayesian_blocks_v2 import blocks_tte
    evt_file = '/Users/xuewc/BurstData/GRB230307A/gbg_evt_230307_15_v01.fits'
    t0 = 131903046.67
    lc = thist_gecam(evt_file, 1, 0, (-1, 1), (20, 300), 0.01, t0)
    evts = events_gecam(evt_file, 1, 0, (-1, 1), (20, 300), t0)
    lc['rate'] = lc['counts'] / lc['exposure']
    lc['t'] = lc['tbins'].mean('edge')
    blocks = blocks_tte(evts, niter=10)[1]
    lc_blocks = np.histogram(evts, blocks)[0]
    Non = lc_blocks[9:12].sum()
    Noff = lc_blocks[0]
    Ton = blocks[12] - blocks[9]
    Toff = blocks[1] - blocks[0]
    alpha = Ton/Toff
    #%%
    plt.figure()
    sig_lm = significance_lima(Non, Noff, alpha)
    plt.step(np.append(lc['tbins'][:, 0], lc['tbins'][-1, 1]),
             np.append(lc['rate'], lc['rate'][-1]),
             where='post', label='GRB 230307A @ $z$=0.065')
    rate = lc_blocks / np.diff(blocks)
    plt.step(blocks, np.append(rate, rate[-1]),
             where='post', label='Bayesian Blocks', alpha=0.6)
    plt.axvspan(blocks[0], blocks[1],
                color='tab:blue', alpha=0.1, label='Background')
    plt.axvspan(blocks[9], blocks[12],
                color='tab:red', alpha=0.1,
                label=f'Li-Ma Significance={sig_lm:.2f}')
    plt.xlabel('$t-T_0$ [s]')
    plt.ylabel('Rate [s$^{-1}$]')
    plt.legend()

    #%%
    plt.figure()
    plt.axvspan(blocks[0], blocks[1],
                color='tab:blue', alpha=0.1)
    plt.axvspan(blocks[9], blocks[12],
                color='tab:red', alpha=0.1)
    mask2 = lc['t'] < blocks[12]
    plt.step(np.append(lc['tbins'][mask2, 0], lc['tbins'][mask2, 1][-1]),
             np.append(lc['rate'][mask2], lc['rate'][mask2][-1]),
             where='post', label='GRB 230307A @ $z$=0.065')

    mask = (blocks[9] <= lc['t']) & (lc['t'] <= blocks[12])

    S = 5
    factor = find_factor_given_significance(S, Non, Noff, alpha)
    z = factor_to_redshift(factor, 0.065)
    expo = factor_to_Ton(factor,
                         lc['exposure'][mask],
                         lc['counts'][mask],
                         Noff,
                         lc['exposure'][mask]/Toff)

    rate = lc['counts'][mask] / expo
    l=plt.step(lc['t'][mask], rate,
               where='mid',
               alpha=0.8,
               label=rf'$\eta$={factor:.2f}, SNR={S:.2f}'
                     f'\ncorresponding to $z$={z:.2f}')
    plt.step(lc['t'][mask]-lc['t'][mask][0]-0.75, rate,
             alpha=0.8, where='mid', color=l[0].get_c())

    S = 3
    factor = find_factor_given_significance(S, Non, Noff, alpha)
    z = factor_to_redshift(factor, 0.065)
    expo = factor_to_Ton(factor,
                         lc['exposure'][mask],
                         lc['counts'][mask],
                         Noff,
                         lc['exposure'][mask] / Toff)

    rate = lc['counts'][mask] / expo
    l = plt.step(lc['t'][mask], rate,
                 where='mid',
                 alpha=0.8,
                 label=rf'$\eta$={factor:.2f}, SNR={S:.2f}'
                       f'\ncorresponding to $z$={z:.2f}')
    plt.step(lc['t'][mask] - lc['t'][mask][0] - 0.4, rate,
             alpha=0.8, where='mid', color=l[0].get_c())
    plt.xlabel('$t-T_0$ [s]')
    plt.ylabel('Rate [s$^{-1}$]')
    plt.legend()