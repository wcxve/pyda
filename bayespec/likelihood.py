import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.function import function

__all__ = ['Chi', 'Cstat', 'Pstat', 'PGstat', 'Wstat']


def normal_logp(value, mu, sigma, beta):
    resd = (value - mu) / sigma

    return -beta/2.0 * resd*resd


def nomal_random(*pars, rng=None, size=None):
    return rng.normal(loc=pars[0], scale=pars[1], size=size)


def poisson_logp(value, mu, beta):
    gof = beta*(pt.xlogx.xlogx(value) - value)
    logp = beta*(pt.xlogx.xlogy0(value, mu) - mu)
    return logp - gof.eval()


def poisson_random(*pars, rng=None, size=None):
    return rng.poisson(lam=pars[0], size=size)


def pgstat_background(s, n, b_est, sigma, a):
    sigma2 = sigma*sigma
    e = b_est - a*sigma2
    f = n*sigma2 + e*s
    c = a*e - s
    d = pt.sqrt(c*c + 4.0*a*f)
    b = pt.switch(
        pt.or_(pt.ge(e, 0.0), pt.ge(f, 0.0)),
        pt.switch(
            pt.gt(n, 0.0),
            (c + d)/(2*a),
            e
        ),
        0.0
    )
    return b


def wstat_background(s, n_on, n_off, a):
    c = a*(n_on + n_off) - (a + 1)*s
    d = pt.sqrt(c*c + 4*a*(a + 1)*n_off*s)
    b = pt.switch(
        pt.eq(n_on, 0),
        n_off/(1 + a),
        pt.switch(
            pt.eq(n_off, 0),
            pt.switch(
                pt.le(s, a/(a + 1)*n_on),
                n_on/(1 + a) - s/a,
                0.0
            ),
            (c + d) / (2*a*(a + 1))
        )
    )
    return b
    # lambda_on = s + b, lambda_off = b/a
    # c = a * (n_on + n_off) - (a + 1) * s
    # d = pm.math.sqrt(c * c + 4 * a * (a + 1) * n_off * s)
    # b = pm.math.switch(
    #     pm.math.eq(n_on, 0),
    #     n_off / (1 + 1 / a),
    #     pm.math.switch(
    #         pm.math.eq(n_off, 0),
    #         pm.math.switch(
    #             pm.math.le(s, n_on / (1 + 1 / a)),
    #             n_on / (1 + 1 / a) - s,
    #             0.0
    #         ),
    #         (c + d) / (2 * (a + 1))
    #     )
    # )
    # return b


class Likelihood:
    pass

def _check_model(model):
    if model.mtype != 'add':
        raise TypeError(
            f'photon flux is undefined for "{model.mtype}" type model'
        )

class Chi(Likelihood):
    def __init__(self, model, data, beta=1.0, context=None):
        _check_model(model)

        name = data.name
        ph_ebins = data.ph_ebins
        resp_matrix = data.resp_matrix
        rate = data.spec_counts/data.spec_exposure
        error = data.spec_error/data.spec_exposure
        if data.has_back:
            rate -= data.back_counts/data.back_exposure
            back_error = data.back_error/data.back_exposure
            error = pt.sqrt(error*error + back_error*back_error)
        channel = data.channel
        chdim = f'channel_{name}'

        NE_dEph = model(ph_ebins)
        CE_dEch = pm.math.dot(NE_dEph, resp_matrix)

        context = pm.modelcontext(context)
        context.add_coord(chdim, channel)

        setattr(context, f'{name}_data', data)
        setattr(context, f'{name}_model', model)
        if hasattr(context, 'data_names'):
            context.data_names.append(name)
        else:
            context.data_names = [name]
        # super().__init__(coords={chdim: channel})

        with context:
            pm.CustomDist(
                f'{name}_N', CE_dEch, error, beta,
                logp=normal_logp, random=nomal_random,
                observed=rate, dims=chdim
            )


class Cstat(Likelihood):
    def __init__(self, model, data, beta=1.0, context=None):
        _check_model(model)

        if not data.spec_poisson:
            raise ValueError(
                'Poisson data is required for using C-statistics'
            )
        if data.has_back:
            back_type = 'Poisson' if data.back_poisson else 'Gaussian'
            stat_type = 'W' if data.back_poisson else 'PG'
            raise ValueError(
                f'C-statistics is not valid for Poisson data with {back_type} '
                f'background, use {stat_type}-statistics ({stat_type}stat) '
                f'instead'
            )

        name = data.name
        ph_ebins = data.ph_ebins
        resp_matrix = data.resp_matrix
        spec_counts = data.spec_counts
        spec_exposure = data.spec_exposure
        channel = data.channel
        chdim = f'channel_{name}'

        NE_dEph = model(ph_ebins)
        CE_dEch = pm.math.dot(NE_dEph, resp_matrix)

        context = pm.modelcontext(context)
        context.add_coord(chdim, channel)

        setattr(context, f'{name}_data', data)
        setattr(context, f'{name}_model', model)
        if hasattr(context, 'data_names'):
            context.data_names.append(name)
        else:
            context.data_names = [name]
        # super().__init__(coords={chdim: channel})

        with context:
            pm.CustomDist(
                f'{name}_N', CE_dEch*spec_exposure, beta,
                logp=poisson_logp, random=poisson_random,
                observed=spec_counts, dims=chdim
            )


class Pstat(Likelihood):
    def __init__(self, model, data, beta=1.0, context=None):
        _check_model(model)

        if not data.spec_poisson:
            raise ValueError(
                'Poisson data is required for using P-statistics'
            )
        if not data.has_back:
            raise ValueError(
                'Background is required for using P-statistics'
            )

        name = data.name
        ph_ebins = data.ph_ebins
        resp_matrix = data.resp_matrix
        spec_counts = data.spec_counts
        back_counts = data.back_counts
        spec_exposure = data.spec_exposure
        back_exposure = data.back_exposure
        channel = data.channel
        chdim = f'channel_{name}'

        NE_dEph = model(ph_ebins)
        CE_dEch = pm.math.dot(NE_dEph, resp_matrix)

        context = pm.modelcontext(context)
        context.add_coord(chdim, channel)

        setattr(context, f'{name}_data', data)
        setattr(context, f'{name}_model', model)
        if hasattr(context, 'data_names'):
            context.data_names.append(name)
        else:
            context.data_names = [name]
        # super().__init__(coords={chdim: channel})

        with context:
            back_rate = back_counts/back_exposure
            pm.CustomDist(
                f'{name}_Non', (CE_dEch + back_rate)*spec_exposure, beta,
                logp=poisson_logp, random=poisson_random,
                observed=spec_counts, dims=chdim
            )


class PGstat(Likelihood):
    def __init__(self, model, data, beta=1.0, store_bkg=False, context=None):
        _check_model(model)

        if not data.spec_poisson:
            raise ValueError(
                'Poisson data is required for using PG-statistics'
            )
        if not data.has_back:
            raise ValueError(
                'Background is required for using PG-statistics'
            )

        name = data.name
        ph_ebins = data.ph_ebins
        resp_matrix = data.resp_matrix
        spec_counts = data.spec_counts
        back_counts = data.back_counts
        back_error = data.back_error
        spec_exposure = data.spec_exposure
        back_exposure = data.back_exposure
        channel = data.channel
        chdim = f'channel_{name}'

        NE_dEph = model(ph_ebins)
        CE_dEch = pm.math.dot(NE_dEph, resp_matrix)

        context = pm.modelcontext(context)
        context.add_coord(chdim, channel)

        setattr(context, f'{name}_data', data)
        setattr(context, f'{name}_model', model)
        if hasattr(context, 'data_names'):
            context.data_names.append(name)
        else:
            context.data_names = [name]
        # super().__init__(coords={chdim: channel})

        with context:
            s = CE_dEch * spec_exposure
            a = spec_exposure/back_exposure
            b = pgstat_background(s, spec_counts, back_counts, back_error, a)

            pm.CustomDist(
                f'{name}_Non', s + a*b, beta,
                logp=poisson_logp, random=poisson_random,
                observed=spec_counts, dims=chdim
            )

            pm.CustomDist(
                f'{name}_Noff', b, back_error, beta,
                logp=normal_logp, random=nomal_random,
                observed=back_counts, dims=chdim
            )

            if store_bkg:
                pm.Deterministic(f'{name}_BKG', b/back_exposure, dims=chdim)


class Wstat(Likelihood):
    def __init__(self, model, data, beta=1.0, store_bkg=False, context=None):
        _check_model(model)

        if not data.spec_poisson:
            raise ValueError(
                'Poisson data is required for using W-statistics'
            )
        if not (data.has_back and data.back_poisson):
            raise ValueError(
                'Poisson background is required for using W-statistics'
            )

        name = data.name
        ph_ebins = data.ph_ebins
        resp_matrix = data.resp_matrix
        spec_counts = data.spec_counts
        back_counts = data.back_counts
        spec_exposure = data.spec_exposure
        back_exposure = data.back_exposure
        channel = data.channel
        chdim = f'channel_{name}'

        NE_dEph = model(ph_ebins)
        CE_dEch = pm.math.dot(NE_dEph, resp_matrix)

        context = pm.modelcontext(context)
        context.add_coord(chdim, channel)

        setattr(context, f'{name}_data', data)
        setattr(context, f'{name}_model', model)
        if hasattr(context, 'data_names'):
            context.data_names.append(name)
        else:
            context.data_names = [name]
        # super().__init__(coords={chdim: channel})

        with context:
            s = CE_dEch * spec_exposure
            a = spec_exposure/back_exposure
            b = wstat_background(s, spec_counts, back_counts, a)

            pm.CustomDist(
                f'{name}_Non', s + a*b, beta,
                logp=poisson_logp, random=poisson_random,
                observed=spec_counts, dims=chdim
            )

            pm.CustomDist(
                f'{name}_Noff', b, beta,
                logp=poisson_logp, random=poisson_random,
                observed=back_counts, dims=chdim
            )

            if store_bkg:
                pm.Deterministic(f'{name}_BKG', b/back_exposure, dims=chdim)