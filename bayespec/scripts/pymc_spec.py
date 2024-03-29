# -*- coding: utf-8 -*-
"""
@author: xuewc
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from pytensor.ifelse import ifelse
from numba import njit, vectorize


def poisson_logp(value, mu, beta, gof):
    return beta * (pm.Poisson.logp(value, mu) - gof)


def poisson_random(*pars, rng=None, size=None):
    return rng.poisson(lam=pars[0], size=size)


class WStat(pm.Model):
    # TODO: initialize src model inside Wstat, given Eph_bins, model class,
    # TODO: and corresponding parameters as args.
    # TODO: this requires parsing operators like + or * etc between models
    def __init__(
        self, rate_src, response, n_on, n_off, t_on, t_off,
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

        # pm.Deterministic(f'{name_}loglike', loglike_on.sum()+loglike_off.sum())

class Powerlaw(pt.Op):
    itypes = [pt.dscalar]
    otypes = [pt.dvector]

    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64, order='C')
        self._grad = PowerlawGrad(self.ebins)

    def perform(self, node, inputs, outputs):
        # return value
        outputs[0][0] = self._perform(*inputs, self.ebins)

    def grad(self, inputs, outputs):
        # return grad Op, in backward mode
        g = self._grad(*inputs)
        return [pt.dot(outputs[0], g)]

    @staticmethod
    @njit
    def _perform(alpha, ebins):
        if alpha != 1.0:
            NE = ebins**(1.0 - alpha) / (1.0 - alpha)
        else:
            NE = np.log(ebins)

        return (NE[1:] - NE[:-1])


class PowerlawGrad(pt.Op):
    itypes=[pt.dscalar]
    otypes=[pt.dvector]

    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64, order='C')
        self.log_ebins = np.log(self.ebins)

    def perform(self, node, inputs, outputs):
        # return value
        outputs[0][0] = self._perform(*inputs, self.ebins, self.log_ebins)

    @staticmethod
    @njit
    def _perform(alpha, ebins, log_ebins):
        if alpha != 1.0:
            v1 = 1.0 - alpha
            v2 = ebins ** v1
            dalpha = v2 * (1 - v1*log_ebins) / (v1*v1)
        else:
            dalpha = - log_ebins*log_ebins / 2.0

        return dalpha[1:] - dalpha[:-1]



from mxspec._pymXspec import callModFunc
# from pyda.numerics.specfun import cutoffpl, cutoffpl_dalpha, cutoffpl_dbeta
from pytensor.gradient import grad_not_implemented
class CutoffPowerlaw(pt.Op):
    itypes = [pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64, order='C')
        self._grad = CutoffPowerlawGrad(self.ebins)

    def perform(self, node, inputs, outputs):
        # return value
        outputs[0][0] = cutoffpl(*inputs, self.ebins)

    def grad(self, inputs, outputs):
        # return grad Op, in backward mode
        g = self._grad(*inputs)
        return [
            pt.dot(outputs[0], g[0]),
            # grad_not_implemented(self, 0, inputs[0]),
            pt.dot(outputs[0], g[1]),
            # grad_not_implemented(self, 1, inputs[1])
        ]
    # def grad(self, inputs, outputs):
    #     # return grad Op, in backward mode
    #     g = self._grad(*inputs)
    #
    #     return [pt.dot(outputs[0], g[0]), pt.dot(outputs[0], g[1])]


class CutoffPowerlawGrad(pt.Op):
    itypes=[pt.dscalar, pt.dscalar]
    otypes=[pt.dvector, pt.dvector]

    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64, order='C')

    def perform(self, node, inputs, outputs):
        # return value
        outputs[0][0] = cutoffpl_dalpha(*inputs, self.ebins)
        outputs[1][0] = cutoffpl_dbeta(*inputs, self.ebins)

class BBodyRad(pt.Op):
    itypes = [pt.dscalar]
    otypes = [pt.dvector]

    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64, order='C')
        # self._grad = BBodyRadGrad(self.ebins)

    def perform(self, node, inputs, outputs):
        # return value
        outputs[0][0] = self._perform(*inputs, self.ebins)

    # def grad(self, inputs, outputs):
    #     # return grad Op, in backward mode
    #     grad_op = self._grad(*inputs)
    #     return [pt.dot(outputs[0], grad_op)]

    @staticmethod
    @njit
    def _perform(kT, ebins):
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

        norm = 1.0344e-3 / 2.0 # norm of 2-point approximation to integral

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

    # @staticmethod
    # @vectorize(nopython=True)
    # @njit('float64(float64)')
    # def polylog2(z):
    #     tol = 1.0e-8
    #     l = 0.0
    #     k = 1.0
    #     zk = z
    #     while True:
    #         term = zk / (k*k)
    #         l += term
    #         if term < tol:
    #             return l
    #         zk *= z
    #         k += 1.0

    # @staticmethod
    # @vectorize(nopython=True)
    # @njit('float64(float64)')
    # def polylog3(z):
    #     tol = 1.0e-8
    #     l = 0.0
    #     k = 1.0
    #     zk = z
    #     while True:
    #         term = zk / (k*k*k)
    #         l += term
    #         if term < tol:
    #             return l
    #         zk *= z
    #         k += 1.0

# from mxspec._pymXspec import callModFunc
# class XspecModel(pt.Op):
#     itypes = [pt.dscalar, pt.dscalar]
#     otypes = [pt.dvector]

#     def __init__(self, ebins):
#         ...

#     def make_node(self, inputs):
#         ...

#     def perform(self, node, inputs, output_storage):
#         # return value
#         output_storage[0][0] = self._eval(*inputs, self.ebins)

#     @staticmethod
#     def _eval(*inputs):
#         ...


class Wabs(pt.Op):
    itypes=[pt.dscalar]
    otypes=[pt.dvector]

    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64, order='C').tolist()
        self.n = 0

    def perform(self, node, inputs, outputs):
        # return value
        outputs[0][0] = self._perform(*inputs, self.ebins)

    # @staticmethod
    def _perform(self, nH, ebins):
        coeff = []
        callModFunc('wabs', ebins, (nH,), coeff, [], 1, '')
        return np.asarray(coeff, dtype=np.float64, order='C')

from astropy.io import fits

def data_wstat(erange, spec_on, spec_off, rspfile, name=None, is_ignore=False):
    with fits.open(spec_on) as hdul:
        name = name or hdul['SPECTRUM'].header['INSTRUME']
        data = hdul['SPECTRUM'].data
        if 'GROUPING' in data.names:
            indices = np.flatnonzero(data['GROUPING'] == 1)
        else:
            indices = np.arange(len(data))
        n_on = np.add.reduceat(data['COUNTS'], indices)
        t_on = hdul['SPECTRUM'].header['EXPOSURE']

    with fits.open(spec_off) as hdul:
        data = hdul['SPECTRUM'].data
        n_off = np.add.reduceat(data['COUNTS'], indices)
        t_off = hdul['SPECTRUM'].header['EXPOSURE']

    with fits.open(rspfile) as hdul:
        if 'MATRIX' in [hdu.name for hdu in hdul]:
            matrix = hdul['MATRIX'].data
        else:
            matrix = hdul['SPECRESP MATRIX'].data

        mask = [np.any(i['MATRIX'] > 0.0) for i in matrix]
        matrix = matrix[mask]

        ebins_ph = np.append(matrix['ENERG_LO'], matrix['ENERG_HI'][-1])

        rsp = matrix['MATRIX']
        if rsp.dtype is np.dtype('O'):
            rsp = np.asarray(rsp.tolist())
        rsp = np.add.reduceat(rsp, indices, axis=1)

        ebounds = hdul['EBOUNDS'].data[indices]
        ebins_ch = np.append(ebounds['E_MIN'], ebounds['E_MAX'][-1])
        ebins_ch = np.column_stack((ebins_ch[:-1], ebins_ch[1:]))

        erange = np.atleast_2d(erange)
        emin = np.expand_dims(erange[:, 0], axis=1)
        emax = np.expand_dims(erange[:, 1], axis=1)

        if is_ignore:
            chmask = (emax <= ebins_ch[:, 0]) | (ebins_ch[:, 1] <= emin)
            chmask = np.all(chmask, axis=0)
        else:
            chmask = (emin <= ebins_ch[:, 0]) & (ebins_ch[:, 1] <= emax)
            chmask = np.any(chmask, axis=0)

    data = {
        'name': name,
        'Non': np.asarray(n_on[chmask], dtype=np.int64),
        'Noff': np.asarray(n_off[chmask], dtype=np.int64),
        'Ton': t_on,
        'Toff': t_off,
        'ebins_ph': np.asarray(ebins_ph, dtype=np.float64),
        'ebins_ch': np.asarray(ebins_ch[chmask], dtype=np.float64),
        'response': np.asarray(rsp[:, chmask], dtype=np.float64),
        'channel': [f'{name}_{c}' for c in np.squeeze(np.argwhere(chmask))]
    }

    return data



# class PL:
#     def __init__(self, PhoIndex, model=None):
#         pm.modelcontext(model)
#         def powerlaw(ebins):

# init PL -> creat op
# call pl -> call over ebins


if __name__ == '__main__':
    import arviz as az
    path = '/Users/xuewc/BurstData/FRB221014/HXMT/'
    LE = data_wstat([1, 11],
                    f'{path}/LE_bmin5.grp',
                    f'{path}/LE_phabkg20s_g0_0-94.pha',
                    f'{path}/LE_rsp.rsp')
    ME = data_wstat([8, 35],
                    f'{path}/ME_bmin5.grp',
                    f'{path}/ME_phabkg20s_g0_0-53.pha',
                    f'{path}/ME_rsp.rsp')
    HE = data_wstat([18, 250],
                    f'{path}/HE_bmin5.grp',
                    f'{path}/HE_phabkg20s_g0_0-12.pha',
                    f'{path}/HE_rsp.rsp')
    beta = 1#/np.log(np.sum([len(inst['channel']) for inst in [LE, ME, HE]]))
    with pm.Model() as model:
        nH = pt.constant(2.79)
        alpha = pm.Uniform('alpha', lower=1, upper=3)
        # Ecut = pm.Uniform('Ecut', lower=0, upper=2000)
        norm = pm.Uniform('norm', lower=0, upper=5)

        for inst in [LE, ME, HE]:
            wabs = Wabs(inst['ebins_ph'])(nH).eval()
            # cpl = wabs*norm*CutoffPowerlaw(inst['ebins_ph'])(alpha, Ecut)
            pl = wabs*norm*Powerlaw(inst['ebins_ph'])(alpha)
            # src = cpl
            # pl = pm.math.exp(norm)*Powerlaw(inst['ebins_ph'])(alpha)
            WStat(pl,
                  inst['response'],
                  inst['Non'],
                  inst['Noff'],
                  inst['Ton'],
                  inst['Toff'],
                  inst['name'],
                  inst['channel'],
                  beta)
        # p_map = pm.find_MAP(return_raw=True)
        idata = pm.sample(10000,
                          tune=2000,
                          #idata_kwargs={'log_likelihood': True},
                          #chains=4,
                          #mp_ctx='forkserver'
                          )
        az.plot_trace(idata, var_names=['alpha', 'norm'])
        for inst in [LE, ME, HE]:
            loglike = idata.log_likelihood
            loglike[inst['name']]=loglike[inst['name']+'_Non']+loglike[inst['name']+'_Noff']
        loglike['all_channel'] = xr.DataArray(
            np.concatenate([loglike[i['name']] for i in [LE, ME, HE]],axis=-1),
            coords={
                'chain': loglike.coords.variables['chain'],
                'draw': loglike.coords.variables['draw'],
                'channel': np.concatenate([i['channel'] for i in [LE, ME, HE]])
            }
        )
        loglike['all'] = loglike['all_channel'].sum('channel')
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    # az.waic(idata, var_name='all_channel', scale='deviance')
    # -2 * idata.log_likelihood['all'].mean()
    import corner
    corner.corner(
        data=idata,
        #var_names=['alpha', 'norm'],
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
