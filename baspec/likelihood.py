"""
Created at 02:23:34 on 2023-05-10

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import pymc as pm
import pytensor as pt


def poisson_logp(value, mu, beta, gof):
    return beta * (pm.Poisson.logp(value, mu) - gof)


def poisson_random(*pars, rng=None, size=None):
    return rng.poisson(lam=pars[0], size=size)


def wstat_background(src_rate, n_on, n_off, t_on, t_off):
    s = src_rate * t_on
    a = t_on / t_off
    c = a * (n_on + n_off) - (a + 1) * s
    d = pm.math.sqrt(c * c + 4 * a * (a + 1) * n_off * s)
    b = pm.math.switch(
        pm.math.eq(n_on, 0),
        n_off / (1 + 1 / a),
        pm.math.switch(
            pm.math.eq(n_off, 0),
            pm.math.switch(
                pm.math.le(s, n_on / (1 + 1 / a)),
                n_on / (1 + 1 / a) - s,
                0.0
            ),
            (c + d) / (2 * (a + 1))
        )
    )
    return b

def pgstat_background():
    ...

class Nomal(pm.Model):
    ...

class PGstat(pm.Model):
    ...

class Wstat(pm.Model):
    # TODO: initialize src model inside Wstat, given Eph_bins, model class,
    # TODO: and corresponding parameters as args.
    # TODO: this requires parsing operators like + or * etc between models
    def __init__(
        self, rate_src, ebins, response, n_on, n_off, t_on, t_off,
        name='', channel=None, beta=1.0, store_bkg=False
    ):
        name_ = f'{name}_' if name else ''
        _name = f'_{name}' if name else ''
        chdim = f'channel{_name}'
        coords = {} if channel is None else {chdim: channel}

        super().__init__(coords=coords)

        s = (pm.math.dot(rate_src, response)) * t_on
        a = t_on / t_off
        c = a * (n_on + n_off) - (a + 1) * s
        d = pm.math.sqrt(c*c + 4*a*(a+1) * n_off * s)
        b = pm.math.switch(
            pm.math.eq(n_on, 0),
            n_off / (1 + 1/a),
            pm.math.switch(
                pm.math.eq(n_off, 0),
                pm.math.switch(
                    pm.math.le(s, n_on / (1 + 1/a)),
                    n_on / (1 + 1/a) - s,
                    0.0
                ),
                (c + d) / (2*(a+1))
            )
        )

        gof_on = pm.Poisson.logp(n_on, n_on).eval()
        pm.CustomDist(
            f'{name_}Non', s+b, beta, gof_on,
            logp=poisson_logp, random=poisson_random,
            observed=n_on, dims=chdim
        )

        gof_off = pm.Poisson.logp(n_off, n_off).eval()
        pm.CustomDist(
            f'{name_}Noff', b/a, beta, gof_off,
            logp=poisson_logp, random=poisson_random,
            observed=n_off, dims=chdim
        )
        # loglike_on = pm.Poisson(
        #     name=f'{name_}Non', mu=s+b, observed=n_on, dims=chdim
        # )
        #
        # loglike_off = pm.Poisson(
        #     name=f'{name_}Noff', mu=b/a, observed=n_off, dims=chdim
        # )

        if store_bkg:
            pm.Deterministic(f'{name_}BKG', b/t_off, dims=chdim)