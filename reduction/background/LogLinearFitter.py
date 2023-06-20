import sys
import warnings
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from scipy.integrate import quad
from tqdm import trange


class LogLinearFitter:
    r"""
    Class for performing a maximum likelihood fit on Time-Energy count data,
    treating each energy channel separately.

    Parameters
    ----------
    counts : (c, t) array_like
        Count data of each channel and time bins.
    tstart : (t,) array_like
        The left edge of time bins.
    tstop : (t,) array_like
        The right edge of time bins.
    exposure : None or (t,) array_like, optional
        The exposure (or live time) of time bins, equals to `tstop - tstart`
        if None. The default is None.
    """

    def __init__(self, counts, tstart, tstop, exposure=None, extra=0.0):
        self._counts = np.atleast_2d(np.array(counts, dtype=np.float64))
        self._nchan, self._ntime = self._counts.shape

        idx = np.flatnonzero([i == 0.0 for i in self._counts.sum(1)])
        if idx.size:
            idx = idx.astype(str)
            warnings.warn(
                f'these channels have zero count: {", ".join(idx)}.\n',
                ZeroCountWarning,
                2
            )

        if len(tstart) != self._ntime:
            raise ValueError('length of `tstart` and `counts` not match')
        else:
            tstart = np.array(tstart, dtype=np.float64)

        if len(tstop) != self._ntime:
            raise ValueError('length of `tstart` and `counts` not match')
        else:
            tstop = np.array(tstop, dtype=np.float64)

        if exposure is not None:
            if len(exposure) != self._ntime:
                raise ValueError('length of `exposure` and `counts` not match')
            exposure = np.array(exposure, dtype=np.float64)
        else:
            exposure = tstop - tstart

        self._shift = -0.5*(tstart.min() + tstop.max())
        self._scale = 0.01 * 2/(tstop.max() - tstart.min())
        self._tstart = self._scale * (tstart + self._shift)
        self._tstop = self._scale * (tstop + self._shift)
        self._sexpo = self._scale*np.sum(exposure)
        self._factor = exposure / (tstop - tstart)

        self._extra = None
        self._order = None
        self._res = []

    def fit(
        self, order, extra=0.0, bounds=None, gtol=1e-4, desc='', progress=True
    ):
        r"""
        Fit the data given the order and extra rate

        Parameters
        ----------
        order : int, optional
            ...
        extra : float, (c,) or (c, t) array_like, optional
            An extra count component added in fit. The exposure correction for
            extra will be done in the fit. The default is 0.0.
        gtol : float, optional
            Absolute tolerance of gradient used in optimization. The default is
            1e-4.

        Returns
        -------

        """
        if type(order) is not int or order < 0:
            raise ValueError('`order` must be non-negative integer')
        if type(gtol) is not float or gtol > 0.01 or gtol < 0.0:
            raise ValueError('``gtol`` must be a float betweem 0.0 and 0.01')

        extra = np.array(extra, dtype=np.float64)
        if extra.shape == ():
            extra = np.full_like(self._counts, extra, dtype=np.float64)
        elif (l1 := len(extra)) == self._nchan:
            if extra.shape == (self._nchan,):
                extra = extra[:, None].repeat(self._ntime, axis=1)
            elif (s1 := extra.shape) != (s2 := (self._nchan, self._ntime)):
                raise ValueError(
                    f'shape of `extra` ({s1}) should be ({s2})'
                )
        else:
            raise ValueError(
                f'length of `extra` ({l1}) not match with channel number '
                f'({self._nchan})'
            )

        tstart = self._tstart/self._scale - self._shift
        tstop = self._tstop/self._scale - self._shift
        dt = tstop - tstart

        self._extra = self._factor * extra
        self._order = order
        self._res = []

        if order == 0: # zero-order has analytical solution
            expo = self._sexpo/self._scale
            counts = np.sum(self._counts, axis=1)
            model = (counts - self._extra.sum(1))/expo
            mask = model < 0.0
            model[mask] = 0.0
            error = np.sqrt(counts)/expo
            error[mask] = 0.0
            self._res = [(m, e) for m, e in zip(model, error)]
            model = model[:, None].repeat(self._ntime, axis=1)
            error = error[:, None].repeat(self._ntime, axis=1)
            model = model + extra/dt
            return np.atleast_2d(model), np.atleast_2d(error)

        pinit = np.full((self._nchan, self._order + 1), 0.0)
        p0_guess = np.sum(self._counts - extra, axis=1) / self._sexpo
        mask = p0_guess > 0.0
        pinit[mask, 0] = np.log(p0_guess[mask])
        pinit[~mask, 0] = -42

        pars_bound = [(None, None) for i in range(self._order + 1)]
        if bounds is not None:
            for b in bounds:
                ipar, ilb, iub = b
                pars_bound[ipar-1] = (ilb, iub)
                if ilb is None:
                    pinit[:, ipar-1] = iub - 0.1*np.abs(iub)
                elif iub is None:
                    pinit[:, ipar - 1] = ilb + 0.1 * np.abs(ilb)
                else:
                    pinit[:, ipar - 1] = (ilb + iub)/2.0

        flag = not progress
        desc = desc or 'Fit'
        for i in trange(self._nchan, desc=desc, disable=flag, file=sys.stdout):
            if self._counts[i].sum() > 0.0:
                res = minimize(
                    fun=self._compute_lnL,
                    x0=pinit[i],
                    args=(i,),
                    method='BFGS' if bounds is None else 'L-BFGS-B',
                    bounds=None if bounds is None else pars_bound,
                    jac='3-point',
                    options={'gtol': gtol}
                )
                if not res.success:
                    jac_norm = np.linalg.norm(res.jac)
                    msg = f'channel {i} fit ended with gradient={jac_norm:.1e}'
                    msg += f' and message "{res.message}"\n'
                    warnings.warn(msg, ConvergenceWarning, 2)
                if type(res.hess_inv) != np.ndarray:
                    res.hess_inv = res.hess_inv.todense()
            else:
                res = None
            self._res.append(res)

        model, error = self.interpolate(
            tstart, tstop, extra=extra, desc='Estimate Error', progress=True
        )
        model /= dt
        error /= dt
        return model, error


    def interpolate(
        self, tstart, tstop, exposure=None, extra=None, desc='', progress=True
    ):
        if len(self._res) == 0:
            raise ValueError('fit must be performed before interpolation')

        tstart = np.array(np.atleast_1d(tstart), dtype=np.float64)
        tstop = np.array(np.atleast_1d(tstop), dtype=np.float64)

        if (l1:=len(tstart)) != (l2:=len(tstop)):
            raise ValueError(
                f'length of `tstart` ({l1}) and `tstop` ({l2}) should match'
            )

        if exposure is not None and type(exposure) not in (float, int):
            if (l3 := len(exposure)) != l1:
                raise ValueError(
                    f'`exposure` size ({l3}) is not matched with bin number '
                    f'({l1})'
                )

        if extra is None:
            if self._extra.sum() > 0.0:
                warnings.warn(
                    'an extra counts component is added in fit, while no extra'
                    ' counts component is provided for interpolation. A '
                    'consistent interpolation should be obtained by adding a '
                    'proper extra counts component.\n',
                    InterpolationWarning,
                    2
                )
            extra = np.zeros((self._nchan, l1))
        else:
            extra = np.array(extra, dtype=np.float64)
            if extra.shape == ():
                extra = np.full((self._nchan, l1), extra, dtype=np.float64)
            elif (l4 := len(extra)) == self._nchan:
                if extra.shape == (self._nchan,):
                    extra = extra[:, None].repeat(l1, axis=1)
                elif (s1 := extra.shape) != (s2 := (self._nchan, l1)):
                    raise ValueError(
                        f'shape of `extra` ({s1}) should be ({s2})'
                    )
            else:
                raise ValueError(
                    f'length of `extra` ({l4}) not match with channel number '
                    f'({self._nchan})'
                )

        if exposure is None:
            exposure = tstop - tstart
            factor = 1.0
        else:
            factor = np.array(exposure, dtype=np.float64)/(tstop - tstart)

        model = np.empty((self._nchan, l1))
        error = np.empty_like(model)

        if self._order == 0:
            for i in range(self._nchan):
                m, e = self._res[i]
                model[i] = m * exposure + extra[i]*factor
                error[i] = e * exposure

            return model, error

        tstart = self._scale * (tstart + self._shift)
        tstop = self._scale * (tstop + self._shift)

        flag = not progress
        desc = desc or 'Interpolate'
        for i in trange(self._nchan, desc=desc, disable=flag, file=sys.stdout):
            model[i], error[i] = self._interp_channel(tstart, tstop, i)

        model += extra
        model *= factor
        error *= factor

        return model, error


    def _interp_channel(self, tstart, tstop, channel):
        n = len(tstart)
        res = self._res[channel]
        if res is None:
            return np.zeros(n), np.zeros(n)

        pars = res.x
        covar = res.hess_inv
        model = np.empty(n)
        dmodel = np.empty((n, len(pars)))

        for i in range(n):
            t1 = tstart[i]
            t2 = tstop[i]
            f = lambda p: self._integrate_rate(p, t1, t2)
            model[i] = f(pars)
            dmodel[i] = approx_derivative(f, pars)

        error = dmodel[:, None, :] @ covar @ dmodel[:, :, None]

        return model, np.sqrt(np.squeeze(error, axis=(1, 2)))


    def _compute_lnL(self, pars, channel):
        lnL = 0.0
        counts = self._counts[channel]
        extra = self._extra[channel]
        for i in range(self._ntime):
            fi = self._factor[i]
            t1 = self._tstart[i]
            t2 = self._tstop[i]
            mi = fi * self._integrate_rate(pars, t1, t2)
            di = counts[i]
            lnL -= mi
            if di > 0.0:
                lnL += di*np.log(mi + extra[i])
        return -lnL


    def _integrate_rate(self, pars, tstart, tstop):
        return quad(self._rate, tstart, tstop, (pars,))[0]


    @staticmethod
    @njit('float64(float64, float64[::1])', cache=True, nogil=True)
    def _rate(t, pars):
        n = pars.size
        basis = np.empty(n)
        v = 1.0
        basis[0] = v
        for i in range(1, n):
            v *= t
            basis[i] = v
        return np.exp(pars.dot(basis))


class ZeroCountWarning(UserWarning):
    r"""
    Issued by initialization when zero count occurs in some channel.
    """
    pass


class ConvergenceWarning(UserWarning):
    r"""
    Issued by fit if optimization is not success.
    """
    pass


class InterpolationWarning(UserWarning):
    r"""
    Issued by interpolation if extra component is given in fit, while not in
    interpolation.
    """
    pass


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

    coeff = np.array([2000, -11., 0.1])[:1]
    event = simulate_events(poly_rate, coeff, 10000, 0, 100)

    tstart = 0
    tstop = 100
    tbins = np.linspace(tstart, tstop, 501)
    counts = np.histogram(event, tbins)[0]
    lc = counts / np.diff(tbins)
    plt.step(tbins, np.append(lc, lc[-1]), where='post')

    fitter = LogLinearFitter(counts, tbins[:-1], tbins[1:], np.diff(tbins))
    model, error = fitter.fit(1, extra=1000*np.diff(tbins)[0])
    # model, error = fitter.interpolate(tbins[:-1], tbins[1:], extra=1000*np.diff(tbins)[0])/np.diff(tbins)
    plt.step(tbins, np.append(lc, lc[-1]), where='post')
    plt.step(tbins, np.append(model[0,:], model[0,-1]), where='post', zorder=10)
    plt.errorbar((tbins[:-1]+tbins[1:])/2, model[0], error[0], fmt='. ')