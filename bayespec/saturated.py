import numpy as np
import pymc as pm
import pytensor.tensor as pt

def poisson_logp(value, mu):
    return value*pt.log(mu) - mu

def poisson_random(mu, rng=None, size=None):
    return rng.poisson(lam=mu, size=size)

class Wsaturated(pm.Model):
    def __init__(
        self, n_on, n_off, t_on, t_off, name='', channel=None, store_bkg=False,
        return_log=False
    ):
        name_ = f'{name}_' if name else ''
        _name = f'_{name}' if name else ''
        chdim = f'channel{_name}'
        nchan = len(n_on)
        coords = {chdim: np.arange(nchan) if channel is None else channel}

        n_on = np.asarray(n_on, dtype=np.float64, order='C')
        n_off = np.asarray(n_off, dtype=np.float64, order='C')

        super().__init__(coords=coords)

        if return_log:
            log_s = pm.Flat(f'{name_}log_src', shape=nchan, dims=chdim)
            s = pt.exp(log_s) * t_on
        else:
            s = pm.HalfFlat(f'{name_}src', shape=nchan, dims=chdim) * t_on
        a = t_on / t_off
        c = a * (n_on + n_off) - (a + 1) * s
        d = pt.sqrt(c*c + 4*a*(a+1) * n_off * s)
        b = pt.switch(
            pt.eq(n_on, 0),
            n_off / (1 + 1/a),
            pt.switch(
                pt.eq(n_off, 0),
                pt.switch(
                    pt.le(s, n_on / (1 + 1/a)),
                    n_on / (1 + 1/a) - s,
                    0.0
                ),
                (c + d) / (2*(a+1))
            )
        )

        db = pt.switch(
            pt.eq(n_on, 0),
            0,
            pt.switch(
                pt.eq(n_off, 0),
                pt.switch(
                    pt.le(s, n_on / (1 + 1/a)),
                    -1,
                    0.0
                ),
                ((1 + a)*s + a*(n_off - n_on) - d) / (2*d)
            )
        )
        db2 = db*db
        dbp1 = db + 1
        if return_log:
            pm.Potential('prior', pt.log((s*db2 + b*(db2 + a*dbp1*dbp1))/(a*b*(s + b)))/2+log_s)
        else:
            pm.Potential('prior', pt.log((s*db2 + b*(db2 + a*dbp1*dbp1))/(a*b*(s + b)))/2)

        pm.CustomDist(
            f'{name_}Non', s+b,
            logp=poisson_logp, random=poisson_random,
            observed=n_on, dims=chdim
        )

        pm.CustomDist(
            f'{name_}Noff', b/a,
            logp=poisson_logp, random=poisson_random,
            observed=n_off, dims=chdim
        )

        if store_bkg:
            pm.Deterministic(f'{name_}BKG', b/t_off, dims=chdim)


from numba import njit
from numpy import log, sqrt
from scipy.optimize import root_scalar
from scipy.stats import chi2, norm

@njit('float64(float64, float64, float64, float64, float64)')
def wstat(rate_src, n_on, n_off, t_on, t_off):
    mu_src = rate_src * t_on
    a = t_on / t_off
    v1 = a + 1.0      # a + 1
    v2 = 1.0 + 1.0/a  # 1 + 1/a
    v3 = 2.0 * v1     # 2*(a+1)
    v4 = 4 * a * v1   # 4*a*(a+1)

    on = n_on
    off = n_off
    s = mu_src

    if on == 0.0:
        stat = s + off*log(v1)
    else:
        if off == 0.0:
            if s <= on / v2:
                stat = -s/a + on*log(v2)
            else:
                stat = s + on*(log(on/s) - 1.0)
        else:
            c = a * (on + off) - v1 * s
            d = sqrt(c*c + v4 * off * s)
            b = (c + d) / v3
            stat = s + v2 * b \
                    - on * (log((s + b)/on) + 1) \
                    - off * (log(b/a/off) + 1)
    return stat


def wstat_ci(n_on, n_off, t_on, t_off, cl=0.68269):
    delta = chi2.ppf(cl, 1.0)/2.0
    nsigma = norm.isf(0.5 - cl/2.0)
    rate_on = n_on/t_on
    rate_off = n_off/t_off
    mle = rate_on - rate_off
    err = np.sqrt(rate_on/t_on + rate_off/t_off)
    if mle >= 0.0:
        s = mle
    else:
        s = 0.0
    # print(mle, err, s, nsigma, (mle-nsigma*err, mle), (mle, mle+nsigma*err))
    stat_min = wstat(s, n_on, n_off, t_on, t_off)
    f = lambda x: wstat(x, n_on, n_off, t_on, t_off) - (stat_min + delta)
    if (lower := root_scalar(f, bracket=(mle-2*nsigma*err, s)).root) < 0.0:
        lower = 0.0
    upper = root_scalar(f, bracket=(s, mle+2*nsigma*err)).root
    return s, lower, upper


if __name__ == '__main__':
    t_on = 1
    t_off = 1
    n = 50
    src = np.full(n, 3)
    bkg = np.full_like(src, 5)
    np.random.seed(42)
    Non = np.random.poisson((src+bkg)*t_on)
    Noff = np.random.poisson(bkg*t_off)

    with pm.Model() as model:
        Wsaturated(Non, Noff, t_on, t_off)
        idata = pm.sample(20000, random_seed=42, target_accept=0.95)
        MAP = pm.find_MAP()['src']
        hdi = pm.hdi(idata, 0.683).to_array().values[0]

    # with pm.Model() as model:
    #     Wsaturated(Non, Noff, t_on, t_off, return_log=True)
    #     idata2 = pm.sample(10000, random_seed=42, target_accept=0.95)
    #     MAP2 = np.exp(pm.find_MAP()['log_src'])
    #     hdi2 = np.exp(pm.hdi(idata2, 0.683).to_array().values[0])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.errorbar(np.arange(n), MAP, np.abs(hdi-MAP[:, None]).T, fmt='. ')
    # plt.errorbar(np.arange(n) + 0.15, MAP2, np.abs(hdi2 - MAP2[:, None]).T, fmt='. ')
    MLE, lower, upper = np.transpose([wstat_ci(Non[i], Noff[i], t_on, t_off) for i in range(n)])
    plt.errorbar(np.arange(n) + 0.15, MLE, [MLE-lower, upper-MLE],
                 fmt='. ')
    MLE = Non/t_on-Noff/t_off
    err = np.sqrt(Non/t_on/t_on + Noff/t_off/t_off)
    plt.errorbar(np.arange(n)+0.3, MLE, err, fmt='. ')
    plt.semilogy()
    # pm.plot_trace(idata)

    j = 0
    for i, s, b in zip(hdi, src, bkg):
        if i[0] <= s <= i[1]:
            j += 1
    p = j/n
