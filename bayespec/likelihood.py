import numpy as np
import pymc as pm
import pytensor.tensor as pt

__all__ = ['Chi2', 'Cstat', 'Pstat', 'PGstat', 'Wstat']


def normal_logp(value, mu, sigma, beta):
    resd = (value - mu) / sigma

    return -beta/2.0 * resd*resd


def nomal_random(*pars, rng=None, size=None):
    return rng.normal(loc=pars[0], scale=pars[1], size=size)


def poisson_logp(value, mu, beta):
    gof = beta*(pt.xlogx.xlogx(value) - value)
    # logp = beta*(pt.xlogx.xlogy0(value, mu) - mu)
    # logp = beta * (value*pt.log(mu) - mu)
    logp = beta * pt.switch(
        pt.eq(value, 0.0),
        -mu,
        value * pt.log(mu) - mu
    )
    return logp - gof


def poisson_random(*pars, rng=None, size=None):
    return rng.poisson(lam=pars[0], size=size)


def pgstat_background(s, n, b_est, sigma, a):
    sigma2 = sigma*sigma
    e = b_est - a*sigma2
    f = a*sigma2*n + e*s
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
    # b = pt.switch(
    #     pt.gt(n, 0.0),
    #     (c + d) / (2 * a),
    #     e
    # )
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

# TODO: warn if likelihood is not correctly specified for data

class Chi2(Likelihood):
    def __init__(self, model, data, beta=1.0, context=None):
        _check_model(model)

        context = pm.modelcontext(context)

        if not hasattr(context, 'data_names'):
            context.data_names = []

        name = data.name
        channel = data.channel
        chdim = f'{name}_channel'

        counts = data.spec_counts
        error = data.spec_error
        if data.has_back:
            counts = counts - data.back_counts/data.back_exposure*data.spec_exposure
            back_error = data.back_error/data.back_exposure*data.spec_exposure
            error = np.sqrt(error*error + back_error*back_error)

        if name not in context.data_names:
            context.data_names.append(name)
            context.add_coord(chdim, channel, mutable=True)
            setattr(context, f'{name}_data', data)
            setattr(context, f'{name}_model', model)
            setattr(context, f'_{name}_spec_poisson', False)
            setattr(context, f'_{name}_include_back', False)

            # if data.spec_poisson:
            #     spec_random = poisson_random
            # else:
            #     spec_random = nomal_random

            # the sampling distribution should be the same (?)
            spec_random = nomal_random

            with context:
                ph_ebins = pm.MutableData(f'{name}_ph_ebins', data.ph_ebins)
                resp_matrix = pm.MutableData(f'{name}_resp_matrix', data.resp_matrix)
                counts = pm.MutableData(f'{name}_spec_counts', counts)
                error = pm.MutableData(f'{name}_spec_error', error)
                exposure = pm.MutableData(f'{name}_exposure', data.spec_exposure)

                NE_dEph = model(ph_ebins)
                CE_dEch = pm.math.dot(NE_dEph, resp_matrix)

                pm.CustomDist(
                    f'{name}_Non', CE_dEch*exposure, error, beta,
                    logp=normal_logp, random=spec_random,
                    observed=counts, dims=chdim
                )
        else:
            context.set_dim(chdim, len(channel), channel)
            context.set_data(f'{name}_ph_ebins', data.ph_ebins)
            context.set_data(f'{name}_resp_matrix', data.resp_matrix)
            context.set_data(f'{name}_spec_counts', counts)
            context.set_data(f'{name}_spec_error', error)
            setattr(context, f'{name}_data', data)


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

        context = pm.modelcontext(context)

        if not hasattr(context, 'data_names'):
            context.data_names = []

        name = data.name
        channel = data.channel
        chdim = f'{name}_channel'

        if name not in context.data_names:
            context.data_names.append(name)
            context.add_coord(chdim, channel, mutable=True)
            setattr(context, f'{name}_data', data)
            setattr(context, f'{name}_model', model)
            setattr(context, f'_{name}_spec_poisson', True)
            setattr(context, f'_{name}_include_back', False)

            with context:
                ph_ebins = pm.MutableData(f'{name}_ph_ebins',data.ph_ebins)
                resp_matrix = pm.MutableData(f'{name}_resp_matrix', data.resp_matrix)
                spec_counts = pm.MutableData(f'{name}_spec_counts', data.spec_counts)
                spec_exposure = pm.MutableData(f'{name}_spec_exposure', data.spec_exposure)

                NE_dEph = model(ph_ebins)
                CE_dEch = pm.math.dot(NE_dEph, resp_matrix)

                pm.CustomDist(
                    f'{name}_Non', CE_dEch*spec_exposure, beta,
                    logp=poisson_logp, random=poisson_random,
                    observed=spec_counts, dims=chdim
                )
        else:
            context.set_dim(chdim, len(channel), channel)
            context.set_data(f'{name}_ph_ebins', data.ph_ebins)
            context.set_data(f'{name}_resp_matrix', data.resp_matrix)
            context.set_data(f'{name}_spec_counts', data.spec_counts)
            context.set_data(f'{name}_spec_exposure', data.spec_exposure)
            setattr(context, f'{name}_data', data)


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

        context = pm.modelcontext(context)

        if not hasattr(context, 'data_names'):
            context.data_names = []

        name = data.name
        channel = data.channel
        chdim = f'{name}_channel'

        if name not in context.data_names:
            context.data_names.append(name)
            context.add_coord(chdim, channel, mutable=True)
            setattr(context, f'{name}_data', data)
            setattr(context, f'{name}_model', model)
            setattr(context, f'_{name}_spec_poisson', True)
            setattr(context, f'_{name}_include_back', False)

            with context:
                ph_ebins = pm.MutableData(f'{name}_ph_ebins',data.ph_ebins)
                resp_matrix = pm.MutableData(f'{name}_resp_matrix', data.resp_matrix)
                spec_counts = pm.MutableData(f'{name}_spec_counts', data.spec_counts)
                back_counts = pm.MutableData(f'{name}_back_counts', data.back_counts)
                spec_exposure = pm.MutableData(f'{name}_spec_exposure', data.spec_exposure)
                back_exposure = pm.MutableData(f'{name}_back_exposure', data.back_exposure)

                NE_dEph = model(ph_ebins)
                CE_dEch = pm.math.dot(NE_dEph, resp_matrix)

                back_rate = back_counts/back_exposure
                pm.CustomDist(
                    f'{name}_Non', (CE_dEch + back_rate)*spec_exposure, beta,
                    logp=poisson_logp, random=poisson_random,
                    observed=spec_counts, dims=chdim
                )
        else:
            context.set_dim(chdim, len(channel), channel)
            context.set_data(f'{name}_ph_ebins', data.ph_ebins)
            context.set_data(f'{name}_resp_matrix', data.resp_matrix)
            context.set_data(f'{name}_spec_counts', data.spec_counts)
            context.set_data(f'{name}_back_counts', data.back_counts)
            context.set_data(f'{name}_spec_exposure', data.spec_exposure)
            context.set_data(f'{name}_back_exposure', data.back_exposure)
            setattr(context, f'{name}_data', data)


class PGstat(Likelihood):
    def __init__(self, model, data, beta=1.0, context=None):
        _check_model(model)

        if not data.spec_poisson:
            raise ValueError(
                'Poisson data is required for using PG-statistics'
            )
        if not data.has_back:
            raise ValueError(
                'Background is required for using PG-statistics'
            )

        context = pm.modelcontext(context)

        if not hasattr(context, 'data_names'):
            context.data_names = []

        name = data.name
        channel = data.channel
        chdim = f'{name}_channel'

        if name not in context.data_names:
            context.data_names.append(name)
            context.add_coord(chdim, channel, mutable=True)
            setattr(context, f'{name}_data', data)
            setattr(context, f'{name}_model', model)
            setattr(context, f'_{name}_spec_poisson', True)
            setattr(context, f'_{name}_include_back', True)
            setattr(context, f'_{name}_back_poisson', False)

            # if data.back_poisson:
            #     back_random = poisson_random
            # else:
            #     back_random = nomal_random

            # the sampling distribution should be the same (?)
            back_random = nomal_random

            with context:
                ph_ebins = pm.MutableData(f'{name}_ph_ebins',data.ph_ebins)
                resp_matrix = pm.MutableData(f'{name}_resp_matrix', data.resp_matrix)
                spec_counts = pm.MutableData(f'{name}_spec_counts', data.spec_counts)
                back_counts = pm.MutableData(f'{name}_back_counts', data.back_counts)
                back_error = pm.MutableData(f'{name}_back_error', data.back_error)
                spec_exposure = pm.MutableData(f'{name}_spec_exposure', data.spec_exposure)
                back_exposure = pm.MutableData(f'{name}_back_exposure', data.back_exposure)

                NE_dEph = model(ph_ebins)
                CE_dEch = pm.math.dot(NE_dEph, resp_matrix)

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
                    logp=normal_logp, random=back_random,
                    observed=back_counts, dims=chdim
                )

                pm.Deterministic(f'{name}_BKGPG', b/back_exposure, dims=chdim)
        else:
            context.set_dim(chdim, len(channel), channel)
            context.set_data(f'{name}_ph_ebins', data.ph_ebins)
            context.set_data(f'{name}_resp_matrix', data.resp_matrix)
            context.set_data(f'{name}_spec_counts', data.spec_counts)
            context.set_data(f'{name}_back_counts', data.back_counts)
            context.set_data(f'{name}_back_error', data.back_error)
            context.set_data(f'{name}_spec_exposure', data.spec_exposure)
            context.set_data(f'{name}_back_exposure', data.back_exposure)
            setattr(context, f'{name}_data', data)


class Wstat(Likelihood):
    def __init__(self, model, data, beta=1.0, store_bkg=True, context=None):
        _check_model(model)

        if not data.spec_poisson:
            raise ValueError(
                'Poisson data is required for using W-statistics'
            )
        if not (data.has_back and data.back_poisson):
            raise ValueError(
                'Poisson background is required for using W-statistics'
            )

        context = pm.modelcontext(context)

        if not hasattr(context, 'data_names'):
            context.data_names = []

        name = data.name
        channel = data.channel
        chdim = f'{name}_channel'

        if name not in context.data_names:
            context.data_names.append(name)
            context.add_coord(chdim, channel, mutable=True)
            setattr(context, f'{name}_data', data)
            setattr(context, f'{name}_model', model)
            setattr(context, f'_{name}_spec_poisson', True)
            setattr(context, f'_{name}_include_back', True)
            setattr(context, f'_{name}_back_poisson', True)

            with context:
                ph_ebins = pm.MutableData(f'{name}_ph_ebins',data.ph_ebins)
                resp_matrix = pm.MutableData(f'{name}_resp_matrix', data.resp_matrix)
                spec_counts = pm.MutableData(f'{name}_spec_counts', data.spec_counts)
                back_counts = pm.MutableData(f'{name}_back_counts', data.back_counts)
                spec_exposure = pm.MutableData(f'{name}_spec_exposure', data.spec_exposure)
                back_exposure = pm.MutableData(f'{name}_back_exposure', data.back_exposure)

                NE_dEph = model(ph_ebins)
                CE_dEch = pm.math.dot(NE_dEph, resp_matrix)
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

                pm.Deterministic(f'{name}_BKGW', b/back_exposure, dims=chdim)
        else:
            context.set_dim(chdim, len(channel), channel)
            context.set_data(f'{name}_ph_ebins', data.ph_ebins)
            context.set_data(f'{name}_resp_matrix', data.resp_matrix)
            context.set_data(f'{name}_spec_counts', data.spec_counts)
            context.set_data(f'{name}_back_counts', data.back_counts)
            context.set_data(f'{name}_spec_exposure', data.spec_exposure)
            context.set_data(f'{name}_back_exposure', data.back_exposure)
            setattr(context, f'{name}_data', data)
