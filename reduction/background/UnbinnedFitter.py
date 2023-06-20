import sys
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.integrate import quad
from tqdm import tqdm

__all__ = ['EventRateFitter']


class EventRateFitter:
    """
    Class for performing a maximum likelihood fit on time-tagged event data,
    treating each energy channel separately.

    Parameters
    ----------
    event : (n,) array_like
        Arrival time of events.
    channel : (n,) array_like
        Channel number of events.
    bound : None, (2,) or (m, 2) array_like
        Events inside `bound` will be used to fit. If `bound` is None, all
        events are used in fit (the default).
    exposure : None, float, or (m,) array_like
        The exposure of given `bound`, equals to length of `bound` if
        `exposure` is None (the default), or equals to time span of events
        if both `bound` and `exposure` are None.

    Methods
    -------
    fit :
        Fit the event data with log linear model of given order.
    interpolate:
        Interpolate the log linear model over given bound.

    Notes
    -----
    The exposure correction in `fit` and `interpolate` assumes the dead time of
    a detection process is not significantly varies with time within the given
    `bound`. If the assumption does not hold, a narrow enough `bound` should be
    given to avoid bias.

    """
    def __init__(self, event, channel, bound=None, exposure=None):
        # check input begin
        if (le := len(event)) != (lc := len(channel)):
            raise ValueError(
                f'size of `event` ({le}) and `channel` ({lc}) is not matched'
            )

        if bound is None:
            if exposure is not None and type(exposure) not in {float, int}:
                raise ValueError(
                    'when `bound` is not given, only None and positive number '
                    'are supported for `exposure`, while the input type is '
                    f'{type(exposure)}'
                )
            self._b = np.atleast_2d([min(event), max(event)])
        else:
            self._b = np.atleast_2d(bound)
            if self._b.shape[1] != 2:
                raise ValueError(
                    'shape of `bound` must be (2,) or (m, 2), while the input'
                    f'is {np.shape(bound)}'
                )
            if exposure is not None and type(exposure) not in {float, int}:
                if (le := len(exposure)) != self._b.shape[0]:
                    raise ValueError(
                        f'`exposure` shape ({le},) is not matched with `bound`'
                        f'shape ({self._b.shape})'
                    )

        # compute exposure
        if exposure is None:
            exposure = np.squeeze(np.diff(self._b, axis=1), axis=1)
        else:
            exposure = np.atleast_1d(exposure)

        # initialize some variable
        self._res = []

        # store the data
        self._b = np.array(self._b, dtype=np.float64, order='C')
        self._t = np.array(event, dtype=np.float64, order='C')
        self._c = np.array(channel, dtype=np.int64, order='C')
        argsort = self._t.argsort()
        self._t = self._t[argsort]
        self._c = self._c[argsort]

        # compute the alive time ratio
        self._a = exposure/np.squeeze(np.diff(self._b, axis=1), axis=1)

        # filter event if given bound
        if bound is not None:
            _t = np.expand_dims(self._t, axis=1)
            mask = (self._b[:, 0] <= _t) & (_t <= self._b[:, 1])
            mask = np.any(mask, axis=1)
            self._t = self._t[mask]
            self._c = self._c[mask]

        self._unq_c = np.unique(self._c)

        # shift and scale the time of event to avoid overflow
        # the shift to align time with 0
        self._s = -(self._b.min() + self._b.max())/2.0
        # scale time so that extrapolation is numerical stable to some level
        scale = 0.5
        self._f = scale*2.0/(self._b.max() - self._b.min())

        # now shift and scale the event
        self._t += self._s
        self._t *= self._f
        self._b += self._s
        self._b *= self._f

    def fit(self, order=1, gtol=1e-4, maxiter=1000, progress=True, desc=''):
        if order < 0:
            raise ValueError('`order` must be non-negative integer')
        if gtol > 0.01 or gtol < 0.0:
            raise ValueError('`gtol` must be a float betweem 0.0 and 0.01')
        if maxiter is not None and maxiter < 0.0:
            raise ValueError('`maxiter` must be a positive integer')

        self._order = order
        self._res = []
        if progress:
            channel = tqdm(self._unq_c, desc, file=sys.stdout)
        else:
            channel = self._unq_c
        for ch in channel:
            t = self._t[self._c == ch]
            basis = t ** np.arange(order + 1)[:, None]
            coeff_init = np.zeros(order + 1)
            res = minimize(
                fun=self._compute_lnL,
                x0=coeff_init,
                args=(basis,),
                method='BFGS',
                # bounds=[(-100, 100) for i in range(order + 1)],
                jac='3-point',
                options={
                    'gtol': gtol,
                    'maxiter': maxiter
                }
            )
            if not res.success:
                print(
                    f'WARNING: channel {ch} fit ended with message '
                    f'"{res.message}"'
                )
            self._res.append(res)

    def interpolate(self, bound, exposure=None, return_count=True, progress=True, desc=''):
        if len(self._res) == 0:
            raise ValueError('you must perform fit before interpolation')

        bound = np.array(bound, dtype=np.float64, order='C')
        if exposure not in {None, False} \
                and type(exposure) not in {float, int}:
            if (le := len(exposure)) != (lb := len(bound)) - 1:
                raise ValueError(
                    f'`exposure` size ({le}) is not matched with bin number '
                    f'(lb)'
                )
        if exposure is None:
            exposure = np.squeeze(np.diff(bound, axis=1))
        else:
            exposure = np.array(exposure, dtype=np.float64, order='C')

        bound = self._f * (bound + self._s)

        model = np.empty((self._unq_c.size, len(bound)))
        error = np.empty_like(model)

        f = lambda coeffs, t1, t2: self._integrate_rate(coeffs, t1, t2)
        if progress:
            channel_res = enumerate(tqdm(self._res, desc, file=sys.stdout))
        else:
            channel_res = enumerate(self._res)
        for c, res in channel_res:
            coeffs = res.x
            covar = res.hess_inv
            m = [
                f(coeffs, *b)
                for b in bound
            ]
            model[c] = np.array(m, dtype=np.float64, order='C')
            dm = [
                approx_derivative(f, coeffs, args=b)
                for b in bound
            ]
            dm = np.array(dm, dtype=np.float64, order='C')
            error[c] = np.sqrt(
                np.squeeze(
                    dm[:, None, :] @ covar @ dm[:, :, None],
                    axis=(1, 2)
                )
            )

        if not return_count:
            model /= exposure
            error /= exposure

        return model, error


    def _compute_lnL(self, coeffs, basis):
        lnL = 0.0
        lnL += self._ln_rate(coeffs, basis).sum()
        for b, a in zip(self._b, self._a):
            lnL -= a*self._integrate_rate(coeffs, b[0], b[1])
        return -lnL

    @staticmethod
    @njit('float64[::1](float64[::1], float64[:,::1])', cache=True)
    def _ln_rate(coeffs, basis):
        # if len(coeffs) > 1:
        #     poly = coeffs[1:] @ basis[1:]
        #     return np.log(np.exp(poly) + np.exp(coeffs[0]))
        # else:
            return coeffs@basis

    @staticmethod
    @njit('float64[::1](float64[::1], float64[:,::1])', cache=True)
    def _rate(coeffs, basis):
        # if len(coeffs) > 1:
        #     poly = coeffs[1:] @ basis[1:]
        #     return np.exp(poly) + np.exp(coeffs[0])
        # else:
            return np.exp(coeffs@basis)

    def _integrate_rate(self, coeffs, tstart, tstop):
        f = lambda t: self._rate(coeffs, t**np.arange(self._order + 1)[:,None])
        return quad(f, tstart, tstop)[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numba as nb

    @nb.njit(nb.float64(nb.float64, nb.float64[:]))
    def poly_rate(t, args):
        r = 0.0
        v = 1.0
        for i in range(len(args)):
            r += args[i]*v
            v *= t

        return r

    @nb.njit(
        nb.float64[:](
            nb.types.FunctionType(nb.float64(nb.float64, nb.float64[:])),
            nb.float64[:],
            nb.float64,
            nb.float64,
            nb.float64
        )
    )
    def simulate_events(ft, args, ftmax, t0, t1):
        """Algorithm 6"""
        np.random.seed(42)
        t = t0
        events = []
        while t <= t1:
            t = t - np.log(np.random.rand()) / ftmax
            if np.random.rand() <= ft(t, args) / ftmax:
                events.append(t)
        return np.array(events)

    coeff = np.array([2000, -11., 0.1])
    event = simulate_events(poly_rate, coeff, 10000, 0, 100)

    tstart = 0
    tstop = 100
    tbins = np.linspace(tstart, tstop, 101)
    lc = np.histogram(event, tbins)[0] / np.diff(tbins)
    plt.step(tbins, np.append(lc, lc[-1]), where='post')

    channel = np.zeros_like(event, dtype=np.int64)
    bound = np.column_stack([tbins[:-1], tbins[1:]])
    fitter = EventRateFitter(event, channel, bound)
    fitter.fit(2)
    print(fitter._res[0].fun)
    model, error = fitter.interpolate(bound, return_count=False)

    plt.step(tbins, np.append(lc, lc[-1]), where='post')
    plt.step(tbins, np.append(model[0,:], model[0,-1]), where='post', zorder=10)
    plt.errorbar((tbins[:-1]+tbins[1:])/2, model[0], error[0], fmt='. ')


# class EventRateFitter:
#     """
#     Class for performing a maximum likelihood fit on time-tagged event data,
#     treating each energy channel separately.
#
#     Parameters
#     ----------
#     event : (n,) array_like
#         Arrival time of events.
#     channel : (n,) array_like
#         Channel number of events.
#     bound : None, (2,) or (m, 2) array_like
#         Events inside `bound` will be used to fit. If `bound` is None, all
#         events are used in fit (the default).
#     exposure : None, float, or (m,) array_like
#         The exposure of given `bound`, equals to length of `bound` if
#         `exposure` is None (the default), or equals to time span of events
#         if both `bound` and `exposure` are None.
#
#     Methods
#     -------
#     fit :
#         Fit the event data with log linear model of given order.
#     interpolate:
#         Interpolate the log linear model over given bound.
#
#     Notes
#     -----
#     The exposure correction in `fit` and `interpolate` assumes the dead time of
#     a detection process is not significantly varies with time within the given
#     `bound`. If the assumption does not hold, a narrow enough `bound` should be
#     given to avoid bias.
#
#     """
#     def __init__(self, event, channel, bound=None, exposure=None):
#         # check input begin
#         if (le := len(event)) != (lc := len(channel)):
#             raise ValueError(
#                 f'size of `event` ({le}) and `channel` ({lc}) is not matched'
#             )
#
#         if bound is None:
#             if exposure is not None or type(exposure) not in {float, int}:
#                 raise ValueError(
#                     'when `bound` is not given, only None and positive number '
#                     'are supported for `exposure`, while the input type is '
#                     f'{type(exposure)}'
#                 )
#             self._b = np.atleast_2d([min(event), max(event)])
#         else:
#             self._b = np.atleast_2d(bound)
#             if self._b.shape[1] != 2:
#                 raise ValueError(
#                     'shape of `bound` must be (2,) or (m, 2), while the input'
#                     f'is {np.shape(bound)}'
#                 )
#             if exposure is not None and type(exposure) not in {float, int}:
#                 if (le := len(exposure)) != self._b.shape[0]:
#                     raise ValueError(
#                         f'`exposure` shape ({le},) is not matched with `bound`'
#                         f'shape ({self._b.shape})'
#                     )
#
#         # compute exposure
#         if exposure is None:
#             exposure = np.squeeze(np.diff(self._b, axis=1), axis=1)
#         else:
#             exposure = np.atleast_1d(exposure)
#
#         # initialize some variable
#         self._res = []
#
#         # store the data
#         self._b = np.array(self._b, dtype=np.float64, order='C')
#         self._t = np.array(event, dtype=np.float64, order='C')
#         self._c = np.array(channel, dtype=np.int64, order='C')
#         argsort = self._t.argsort()
#         self._t = self._t[argsort]
#         self._c = self._c[argsort]
#
#         # compute the alive time ratio
#         self._a = exposure/np.squeeze(np.diff(self._b, axis=1), axis=1)
#
#         # filter event if given bound
#         if bound is not None:
#             _t = np.expand_dims(self._t, axis=1)
#             mask = (self._b[:, 0] <= _t) & (_t <= self._b[:, 1])
#             mask = np.any(mask, axis=1)
#             self._t = self._t[mask]
#             self._c = self._c[mask]
#
#         self._unq_c = np.unique(self._c)
#
#         # shift and scale the time of event to avoid overflow
#         # the shift to align time with 0
#         self._s = -(self._b.min() + self._b.max())/2.0
#         # the factor to scale time into [-0.001, 0.001],
#         # so that extrapolation is numerical stable to some level
#         scale = 0.5
#         self._f = scale*2.0/(self._b.max() - self._b.min())
#
#         # now shift and scale the event
#         self._t += self._s
#         self._t *= self._f
#         self._b += self._s
#         self._b *= self._f
#
#     def fit(self, order=1, gtol=1e-4, maxiter=1000, progress=True, desc=''):
#         if order < 0:
#             raise ValueError('`order` must be non-negative integer')
#         if gtol > 0.01 or gtol < 0.0:
#             raise ValueError('`gtol` must be a float betweem 0.0 and 0.01')
#         if maxiter is not None and maxiter < 0.0:
#             raise ValueError('`maxiter` must be a positive integer')
#
#         scaled_exposure = np.sum(self._a*np.diff(self._b, axis=1))
#         self._res = []
#         if progress:
#             channel = tqdm(self._unq_c, desc, file=sys.stdout)
#         else:
#             channel = self._unq_c
#         for c in channel:
#             t = self._t[self._c == c]
#             n = len(t)
#             if n:
#                 coeff_init = np.zeros(order + 1)
#                 if order:
#                     coeff_init[0] = np.log(n/scaled_exposure)
#                 res = minimize(
#                     fun=self._compute_lnL,
#                     x0=coeff_init,
#                     args=(t,),
#                     method='BFGS',
#                     # method='L-BFGS-B',
#                     # bounds=[(-100, 100) for i in range(order + 1)],
#                     jac='3-point',
#                     options={
#                         'gtol': gtol,
#                         'maxiter': maxiter
#                     }
#                 )
#                 if not res.success:
#                     print(
#                         f'WARNING: channel {c} fit ended with message '
#                         f'"{res.message}"'
#                     )
#             else:
#                 res = None
#             self._res.append(res)
#
#     def interpolate(self, bound, exposure=None, return_count=True, progress=True, desc=''):
#         if len(self._res) == 0:
#             raise ValueError('you must perform fit before interpolation')
#
#         bound = np.array(bound, dtype=np.float64, order='C')
#         if exposure not in {None, False} \
#                 and type(exposure) not in {float, int}:
#             if (le := len(exposure)) != (lb := len(bound)) - 1:
#                 raise ValueError(
#                     f'`exposure` size ({le}) is not matched with bin number '
#                     f'(lb)'
#                 )
#         if exposure is None:
#             exposure = np.squeeze(np.diff(bound, axis=1))
#         else:
#             exposure = np.array(exposure, dtype=np.float64, order='C')
#
#         bound = self._f * (bound + self._s)
#
#         model = np.empty((self._unq_c.size, len(bound)))
#         error = np.empty_like(model)
#
#         f = lambda coeffs, t1, t2: self._integrate_rate(coeffs, t1, t2)
#         if progress:
#             channel_res = enumerate(tqdm(self._res, desc, file=sys.stdout))
#         else:
#             channel_res = enumerate(self._res)
#         for c, res in channel_res:
#             coeffs = res.x
#             covar = res.hess_inv
#             m = [
#                 f(coeffs, *b)
#                 for b in bound
#             ]
#             model[c] = np.array(m, dtype=np.float64, order='C')
#             dm = [
#                 approx_derivative(f, coeffs, args=b)
#                 for b in bound
#             ]
#             dm = np.array(dm, dtype=np.float64, order='C')
#             error[c] = np.sqrt(
#                 np.squeeze(
#                     dm[:, None, :] @ covar @ dm[:, :, None],
#                     axis=(1, 2)
#                 )
#             )
#
#         if not return_count:
#             model /= exposure
#             error /= exposure
#
#         return model, error
#
#
#     def _compute_lnL(self, coeffs, t):
#         lnL = 0.0
#         lnL += self._ln_rate(coeffs, t).sum()
#         for b, a in zip(self._b, self._a):
#             lnL -= a*self._integrate_rate(coeffs, b[0], b[1])
#         return -lnL
#
#     def _ln_rate(self, coeffs, t):
#         return np.polynomial.polynomial.polyval(t, coeffs)
#
#     def _rate(self, coeffs, t):
#         return np.exp(self._ln_rate(coeffs, t))
#
#     def _integrate_rate(self, coeffs, tstart, tstop):
#         return quad(lambda t: self._rate(coeffs, t), tstart, tstop)[0]
#     #
#     # def _compute_lnL(self, coeffs, t):
#     #     lnL = 0.0
#     #     lnL += self._ln_rate(coeffs, t).sum()
#     #     for b, a in zip(self._b, self._a):
#     #         lnL -= a*self._integrate_rate(coeffs, b[0], b[1])
#     #     return -lnL
#     #
#     # def _ln_rate(self, coeffs, t):
#     #     if len(coeffs) > 1:
#     #         poly = np.polynomial.polynomial.polyval(t, np.append(0,coeffs[1:]))
#     #         return np.log(np.exp(poly) + np.exp(coeffs[0]))
#     #     else:
#     #         return np.full_like(t, coeffs[0])
#     #
#     # def _rate(self, coeffs, t):
#     #     if len(coeffs) > 1:
#     #         poly = np.polynomial.polynomial.polyval(t, np.append(0,coeffs[1:]))
#     #         return np.exp(poly) + np.exp(coeffs[0])
#     #     else:
#     #         return np.full_like(t, np.exp(coeffs[0]))
#     #
#     # def _integrate_rate(self, coeffs, tstart, tstop):
#     #     return quad(lambda t: self._rate(coeffs, t), tstart, tstop)[0]
#
#
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     tstart = 0 + 131903046.67
#     tstop = 100 + 131903046.67
#     rate = 0.5
#     tbins = np.linspace(tstart, tstop, 501)
#     exposure = tstop - tstart
#     N = np.random.poisson(exposure*rate)
#     event = np.sort(np.random.uniform(tstart, tstop, N))
#     channel = np.zeros(N, dtype=np.int64)
#     bound = np.column_stack([tbins[:-1], tbins[1:]])
#     fitter = EventRateFitter(event, channel, bound)
#     fitter.fit(2)
#     model, error = fitter.interpolate(bound, return_count=False)
#     plt.step(tbins, np.append(model[0,:], model[0,-1]), where='post', zorder=10)
#     plt.errorbar((tbins[:-1]+tbins[1:])/2, model[0], error[0], fmt='. ')