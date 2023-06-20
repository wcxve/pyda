import numpy as np
from numpy import exp, log
from scipy.optimize import minimize
from scipy.special import dawsn
from scipy.stats import chi2, norm


class LogLinearCountFitter:
    def __init__(self, counts, t, factor=1.0, extra=0.0):
        # extra is unfactored counts
        # self.__floor is the additional counts
        self.__floor = 0.0e-5 / (np.max(t) - np.min(t))
        self._shift = (np.min(t) + np.max(t)) / 2.0
        self._scale = 1.0 / np.abs(t - self._shift).max()

        self._counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
        self._t = np.asarray(self._scale * (t - self._shift), dtype=np.float64)
        self._factor = np.asarray(factor, dtype=np.float64)
        if type(extra) is float and extra == 0.0:
            self._extra = np.full_like(self._counts, self.__floor)
        else:
            self._extra = np.atleast_2d(extra) + self.__floor
        self._nchans, self._ntimes = self._counts.shape

        # self._TIME = time.time()
        self._basis = None
        self._basis_product = None
        self._dof = None
        self._order = None
        self._res = []

    def fit(self, order=1):
        # TODO: 全 0 数组拟合修正
        if order < 0:
            raise ValueError('Order must be non-negative')
        self._order = order
        self._basis = self._eval_basis(self._t)
        # basis product, shape (order + 1, order + 1, ntimes)
        self._basis_product = self._basis[:, None, :] * self._basis[None, :, :]

        for n_chan in range(self._nchans):
            self.__pars = None
            init = np.zeros(self._order + 1)
            best = self._counts[n_chan].sum() / self._factor.sum()
            _extra = np.mean(self._extra[n_chan])
            _adjust = -_extra if best > _extra else 0
            init[0] = log(best + _adjust)
            res = minimize(
                fun=self._objective_function,
                x0=init,
                method='trust-exact',
                args=(n_chan,),
                jac=self._jacobian,
                hess=self._hessian,
                # callback=lambda x: print(x, time.time()-self._TIME, flush=True),
                # options={'gtol': 1e-8}
                # terminate if gradient norm is less than gtol, this option will
                # somehow cause failure if gtol is too small and the data have very
                # low counts
            )
            self._res.append(res)
            if not res.success:
                print(
                    f'WARNING: Channel {n_chan} background fit ended with '
                    f'message "{res.message}"'
                )

    def interpolate(self, t, factor=1.0, extra=0.0):
        basis = self._eval_basis(self._scale * (t - self._shift))
        model = np.empty((self._nchans, basis.shape[1]))
        variance = np.empty((self._nchans, basis.shape[1]))
        if type(extra) is float and extra == 0.0:
            _extra = np.full_like(model, self.__floor)
        else:
            _extra = np.atleast_2d(extra) + self.__floor

        for nchan in range(self._nchans):
            pars = self._res[nchan].x
            covar = np.linalg.inv(self._res[nchan].hess)
            tmp = exp(pars @ basis)
            model[nchan] = factor * (tmp + _extra[nchan])
            variance[nchan] = factor * factor * tmp * tmp * np.squeeze(
                basis.T[:, None, :] @ covar @ basis.T[:, :, None],
                axis=(1, 2)
            )

        return model, np.sqrt(variance)

    def _eval_basis(self, t):
        """
        Evaluate the basis functions for each time bin.

        Parameters
        ----------
        t : (N,) array_like
            The middle of time bins.

        Returns
        -------
        basis : (``order`` + 1, N) ndarray
            The basis functions for each time bin.

        """
        basis = np.array([t ** i for i in range(self._order + 1)])
        return np.atleast_2d(basis)

    def _eval_model(self, pars, basis, factor=1.0, extra=0.0):
        """
        Evaluate the model values over the given time bins.

        Parameters
        ----------
        pars : (``order`` + 1,) array_like
            The parameters of log-linear model.
        basis : (``order`` + 1, N) array_like
            The basis functions for each time bin.
        factor : (N,) array_like or float, optional
            The exposure (or live time) factor of time bins. The default is 1.
        extra : (N,) array_like or float, optional
            The extra component to add to model. The default is 0.

        Returns
        -------
        model : (N,) ndarray
            The model values of each channel over the given time bins.

        """
        if np.any(pars != self.__pars):
            self.__pars = pars
            self.__back = factor * exp(pars @ basis)
            self.__total = self.__back + factor * extra
            # self.__total = np.max(
            #    [self.__total, np.full_like(self.__total, 1e-2)],
            #    axis=0
            # ) # avoid zero model value
            # self.__total[self.__total < 1e-20] = 1e-20
            # print('EVAL', pars, time.time()-self._TIME, flush=True)

    def _objective_function(self, pars, n_chan):
        """This is the negative log Poisson likelihood"""
        self._eval_model(pars, self._basis, self._factor, self._extra[n_chan])
        # print('OBJ', pars, time.time()-self._TIME, flush=True)
        return np.sum(self.__total - self._counts[n_chan] * log(self.__total))

    def _jacobian(self, pars, n_chan):
        """This is the Jacobian of the objective function"""
        self._eval_model(pars, self._basis, self._factor, self._extra[n_chan])
        jac = np.sum(
            (1 - self._counts[
                n_chan] / self.__total) * self.__back * self._basis,
            axis=-1
        )
        # print('JAC', pars, time.time()-self._TIME, flush=True)
        return jac

    def _hessian(self, pars, n_chan):
        """This is the Hessian of the objective function"""
        self._eval_model(pars, self._basis, self._factor, self._extra[n_chan])
        hess = np.sum(
            (self._counts[n_chan] * (self.__back / self.__total) / self.__total
             + 1 - self._counts[n_chan] / self.__total) * self.__back \
            * self._basis_product,
            axis=-1
        )
        if np.isnan(hess).any() or np.isinf(hess).any():
            raise ValueError(pars, n_chan, hess, self.__back, self.__total)
        # print('HESS', pars, time.time()-self._TIME, flush=True)
        return hess


class LogLinearRateFitter:
    """
    Class for performing a maximum likelihood fit on Time-Energy histogram
    count data, treating each energy channel separately.

    Parameters
    ----------
    counts : (``nchans``, ``ntimes``) array_like
        Count data of each channel over time bins.
    tstart : (``ntimes``,) array_like
        The left edge of time bins.
    tstop : (``ntimes``,) array_like
        The right edge of time bins.
    exposure : (``ntimes``,) array_like
        The exposure (or live time) of time bins.
    known_component : scalar or (``nchans``, ``ntimes``) array_like, optional
        A known count rate component in ``counts`` data. The default is 0.0.
    """

    def __init__(self, counts, tstart, tstop, exposure, known_component=0.0):
        self._counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
        self._nchans, self._ntimes = self._counts.shape

        if len(tstart) != self._ntimes:
            raise ValueError('length of ``tstart`` and ``counts`` not match')
        if len(tstop) != self._ntimes:
            raise ValueError('length of ``tstart`` and ``counts`` not match')
        if len(exposure) != self._ntimes:
            raise ValueError('length of ``exposure`` and ``counts`` not match')

        # self.__floor is the minimum count rate in ``counts`` data
        self.__floor = 1e-100
        self._shift = (min(tstart) + max(tstop)) / 2.0
        self._scale = 0.5 / np.abs(np.r_[tstart, tstop] - self._shift).max()
        self._tstart = np.asarray(tstart, dtype=np.float64)
        self._tstop = np.asarray(tstop, dtype=np.float64)
        self._exposure = np.asarray(exposure, dtype=np.float64)
        self._factor = self._exposure / (self._tstop - self._tstart)
        self._component_c = self._eval_count(known_component, self._exposure)
        self._component_r = known_component

    def fit(
            self, order=1, gtol=1e-4, maxiter=None, ltol=1e-9, ptol=1e-9,
            eta=0.15, initial_trust_radius=1.0, max_trust_radius=1000.0
    ):
        if order not in [0, 1, 2]:
            raise ValueError('Order must be 0, 1, or 2')
        if gtol > 0.01 or gtol < 0.0:
            raise ValueError('``gtol`` must be a float betweem 0.0 and 0.01')
        if maxiter is not None and maxiter < 0.0:
            raise ValueError('``maxiter`` must be a positive integer')
        if ltol > 0.1 or ltol <= 0.0:
            raise ValueError('``ltol`` must be a float betweem 0.0 and 0.1')
        if ptol > 0.1 or ptol <= 0.0:
            raise ValueError('``ptol`` must be a float betweem 0.0 and 0.1')

        self._gtol = gtol
        self._maxiter = maxiter
        self._ltol = ltol
        self._ptol = ptol
        self._eta = eta
        self._initial_trust_radius = initial_trust_radius
        self._max_trust_radius = max_trust_radius

        self._order = order
        self._t1, self._basis1 = self._eval_basis(self._tstart)
        self._t2, self._basis2 = self._eval_basis(self._tstop)
        self._res = []

        scaled_exposure = self._scale * np.sum(self._exposure)
        init0_guess = np.sum(self._counts, axis=1) / scaled_exposure
        init0_adjust = np.sum(self._component_c, axis=1) / scaled_exposure
        init0_adjust[init0_guess <= init0_adjust] = 0.0
        init = np.full((self._nchans, self._order + 1), 0.0)
        mask = init0_guess > 0.0  # for non-zero counts
        init[mask, 0] = log(init0_guess[mask] - init0_adjust[mask])

        for nchan in range(self._nchans):
            self.__pars = np.nan
            self.__obj = np.inf
            if self._counts[nchan].sum() != 0.0:
                res = self._minimize(init[nchan], nchan)
            else:
                res = None
            self._res.append(res)

        model, error = self.interpolate(
            self._tstart, self._tstop, self._exposure, self._component_r
        )

        mask = (self._counts.sum(1) != 0.0)
        C = self._gof(model[mask], self._counts[mask])
        self._gof_C = np.zeros((self._nchans, self._ntimes))
        self._gof_C[mask] = (C[0] - C[1]) / np.sqrt(C[2])
        self._gof_Ctime = (C[0].sum(0) - C[1].sum(0)) / np.sqrt(C[2].sum(0))
        self._gof_Cchan = (C[0].sum(1) - C[1].sum(1)) / np.sqrt(C[2].sum(1))
        self._gof_Csum = (C[0].sum() - C[1].sum()) / np.sqrt(C[2].sum())
        self._pvalue_C = norm.sf(np.abs(self._gof_Csum))
        self._dof = mask.sum() * (self._ntimes - self._order - 1)
        self._gof_chi = (self._counts - model) / np.sqrt(model)
        self._gof_chi_red = (self._gof_chi ** 2).sum() / self._dof
        self._gof_chi_time = (self._counts.sum(0) - model.sum(0)) / np.sqrt(
            model.sum(0))
        self._pvalue_chi = chi2(self._dof).sf(self._gof_chi_red * self._dof)
        return model, error

    def interpolate(self, tstart, tstop, exposure, extra_component=0.0):
        tstart = np.atleast_1d(tstart)
        tstop = np.atleast_1d(tstop)
        exposure = np.atleast_1d(exposure)
        if not len(tstart) == len(tstop) == len(exposure):
            raise ValueError(
                'length of ``tstart``, ``tstop``, and ``exposure`` not match'
            )
        extra = self._eval_count(extra_component, exposure)
        factor = exposure / (tstop - tstart)
        t1, basis1 = self._eval_basis(tstart)
        t2, basis2 = self._eval_basis(tstop)
        interp = np.empty((self._nchans, len(exposure)), dtype=float)
        uncert = np.empty((self._nchans, len(exposure)), dtype=float)
        for n in range(self._nchans):
            if self._res[n] is None:
                interp[n] = 0.0
                uncert[n] = 0.0
                continue
            p = self._res[n].x
            c = extra[n]
            p_covar = np.linalg.inv(self._res[n].hess)
            m, Dm = self._model(p, t1, t2, basis1, basis2, factor, c, 1, 0)
            Dm = np.transpose(Dm)
            interp[n] = m
            uncert[n] = np.sqrt(np.squeeze(
                Dm[:, None, :] @ p_covar @ Dm[:, :, None],
                axis=(1, 2)
            ))
        return interp, uncert

    def _gof(self, mu, k):
        mu = np.asarray(mu, dtype=np.float64)
        if np.any(mu <= 0.0):
            raise ValueError('``mu`` must be positive!')
        k = np.asarray(k, dtype=np.float64)
        if np.any(k < 0.0):
            raise ValueError('``k`` must be non-negative!')

        zero_mask = (k == 0.0)
        C = 2 * (mu - k + np.piecewise(
            k,
            [zero_mask, ~zero_mask],
            [0.0, lambda k: k * log(k / mu[~zero_mask])]
        ))

        Ce_cond = [
            (0.0 <= mu) & (mu <= 0.5),
            (0.5 < mu) & (mu <= 2.0),
            (2.0 < mu) & (mu <= 5.0),
            (5.0 < mu) & (mu <= 10.0),
            mu > 10.0
        ]
        Ce_func = [
            lambda mu: -0.25 * mu ** 3 + 1.38 * mu ** 2 - 2 * mu * log(mu),
            lambda mu: -0.00335 * mu ** 5 + 0.04259 * mu ** 4 \
                       - 0.27331 * mu ** 3 + 1.381 * mu ** 2 - 2 * mu * log(
                mu),
            lambda mu: 1.019275 + 0.1345 * mu ** (0.461 - 0.9 * log(mu)),
            lambda mu: 1.00624 + 0.604 / (mu ** 1.68),
            lambda mu: 1.0 + 0.1649 / mu + 0.226 / (mu ** 2),
        ]
        Ce = np.piecewise(mu, Ce_cond, Ce_func)

        Cv_cond = [
            (0.0 <= mu) & (mu <= 0.1),
            (0.1 < mu) & (mu <= 0.2),
            (0.2 < mu) & (mu <= 0.3),
            (0.3 < mu) & (mu <= 0.5),
            (0.5 < mu) & (mu <= 1.0),
            (1.0 < mu) & (mu <= 2.0),
            (2.0 < mu) & (mu <= 3.0),
            (3.0 < mu) & (mu <= 5.0),
            (5.0 < mu) & (mu <= 10.0),
            mu > 10
        ]
        Cv_func = [
            lambda mu: 4 * sum(
                exp(-mu) * mu ** k / np.math.factorial(k) * \
                (mu - k + (k * log(k / mu) if k else 0.0)) ** 2
                for k in range(5)
            ) - Ce[Cv_cond[0]] ** 2,
            lambda
                mu: -262 * mu ** 4 + 195 * mu ** 3 - 51.24 * mu ** 2 + 4.34 * mu + 0.77005,
            lambda mu: 4.23 * mu ** 2 - 2.8254 * mu + 1.12522,
            lambda
                mu: -3.7 * mu ** 3 + 7.328 * mu ** 2 - 3.6926 * mu + 1.20641,
            lambda
                mu: 1.28 * mu ** 4 - 5.191 * mu ** 3 + 7.666 * mu ** 2 - 3.5446 * mu + 1.15431,
            lambda
                mu: 0.1125 * mu ** 4 - 0.641 * mu ** 3 + 0.859 * mu ** 2 + 1.0914 * mu - 0.05748,
            lambda
                mu: 0.089 * mu ** 3 - 0.872 * mu ** 2 + 2.8422 * mu - 0.67539,
            lambda mu: 2.12336 + 0.012202 * mu ** (5.717 - 2.6 * log(mu)),
            lambda mu: 2.05159 + 0.331 * mu ** (1.343 - log(mu)),
            lambda mu: 12 / (mu ** 3) + 0.79 / (mu ** 2) + 0.6747 / mu + 2
        ]
        Cv = np.piecewise(mu, Cv_cond, Cv_func)
        return C, Ce, Cv

    def _minimize(self, p0, nchan):
        loop = True
        ncall_lm = 0
        nfev = 0
        njev = 0
        nhev = 0
        nit = 0
        nit_lm = 0
        p = p0
        while loop:
            res = minimize(
                fun=self._objective_function,
                x0=p,
                method='trust-exact',
                args=(nchan,),
                jac=self._jacobian,
                hess=self._hessian,
                options={
                    'initial_trust_radius': self._initial_trust_radius,
                    'max_trust_radius': self._max_trust_radius,
                    'eta': self._eta,
                    'gtol': self._gtol,
                    'maxiter': self._maxiter
                }
            )
            res.jac_norm = np.linalg.norm(res.jac)
            if res.success:
                loop = False
            else:
                lm_res = self._lm_minimize(self.__pars_prev, nchan)
                ncall_lm += 1
                nit_lm += lm_res['nit_lm']
                if lm_res['success']:
                    for key in lm_res:
                        setattr(res, key, lm_res[key])
                    loop = False
                else:
                    if np.all(res.x == lm_res['x']) or ncall_lm > 20:
                        raise ValueError(
                            f'Channel {nchan} fit failed!\n{res}\n\n{lm_res}\n'
                        )
                    else:
                        p = res['x']
            nfev += res.nfev
            njev += res.njev
            nhev += res.nhev
            nit += res.nit
        res.ncall_lm = ncall_lm
        res.nfev = nfev
        res.njev = njev
        res.nhev = nhev
        res.nit = nit
        res.nit_lm = nit_lm
        return res

    def _lm_minimize(self, p0, nchan):
        """This is the modified Levenberg-Marquardt algorithm"""
        d = self._counts[nchan]
        c = self._component_c[nchan]
        t1 = self._t1
        t2 = self._t2
        basis1 = self._basis1
        basis2 = self._basis2
        factor = self._factor

        p = np.reshape(p0, -1)
        m, Dm = self._model(p, t1, t2, basis1, basis2, factor, c, 1, 0)
        DmDm = Dm[:, None, :] * Dm[None, :, :]
        l, Dl, DDl = self._loss(d, m)
        alpha = np.sum(DDl * DmDm, axis=-1)
        beta = -np.sum(Dl * Dm, axis=-1)
        mu = 0.001
        iter_count = 0
        pnum = len(p)
        maxiter = self._maxiter or 200 * pnum
        enrom_g = np.linalg.norm(beta)
        flags = np.array([1, enrom_g >= self._gtol, 1, 1], dtype=bool)
        mu_error = False
        while all(flags):
            loop_mu = True
            # loop over all possible mu to find a p_new which reduces the loss
            while loop_mu and flags[0]:
                a_new = alpha + mu * np.eye(pnum)
                delta = np.linalg.inv(a_new) @ beta
                p_new = p + delta
                m_new = self._model(
                    p_new, t1, t2, basis1, basis2, factor, c, 0, 0
                )[0]
                l_new = self._loss(d, m_new)[0]
                delta_l = (l - l_new)
                rho = delta_l / (beta @ delta - delta @ a_new @ delta / 2.0)
                if rho > 0.0:
                    p = p_new
                    m = m_new
                    l = l_new
                    Dm = self._model(
                        p, t1, t2, basis1, basis2, factor, c, 1, 0
                    )[1]
                    DmDm = Dm[:, None, :] * Dm[None, :, :]
                    l, Dl, DDl = self._loss(d, m)
                    alpha = np.sum(DDl * DmDm, axis=-1)
                    beta = -np.sum(Dl * Dm, axis=-1)
                    mu = max(0.1 * mu, 1e-30)
                    loop_mu = False
                else:
                    mu *= 2.0
                    loop_mu = True

                iter_count += 1
                flags[0] = iter_count < maxiter
                if mu > 1e75:
                    mu_error = True
                    break
            if mu_error:
                break
            enorm_delta = np.linalg.norm(delta)
            enrom_g = np.linalg.norm(beta)
            flags[1] = enrom_g >= self._gtol
            flags[2] = delta_l <= 0.0 or delta_l > self._ltol
            flags[3] = enorm_delta > self._ptol
        DDm = self._model(p, t1, t2, basis1, basis2, factor, c, 0, 1)[1]
        status = [-1] if mu_error else list(np.where(~flags)[0])
        message = {
            -1: 'A large mu (> 1e75) encountered.',
            0: 'Maximum number of iterations has been exceeded.',
            1: 'Gradient condition satisfied.',
            2: 'Loss condition satisfied.',
            3: 'Parameter condition satisfied.',
        }
        result = {
            'fun': l,
            'hess': np.sum(DDl * DmDm + Dl * DDm, axis=-1),
            'jac': -beta,
            'jac_norm': enrom_g,
            'message': 'LM: ' + ' '.join([message[i] for i in status]),
            'nit_lm': iter_count,
            'status': 0 if all(np.r_[status] > 0) else -1,
            'success': True if all(np.r_[status] > 0) else False,
            'x': p
        }
        return result

    def _objective_function(self, pars, nchan):
        """This is the negative log Poisson likelihood, a.k.a. C-statistics."""
        self._fit_eval(pars, nchan)
        return self.__obj

    def _jacobian(self, pars, nchan):
        """This is the Jacobian of the objective function."""
        self._fit_eval(pars, nchan)
        return self.__jac

    def _hessian(self, pars, nchan):
        """This is the Hessian of the objective function."""
        self._fit_eval(pars, nchan)
        return self.__hess

    def _fit_eval(self, pars, nchan):
        if np.allclose(pars, self.__pars):
            return None

        p = pars
        d = self._counts[nchan]
        component = self._component_c[nchan]
        t1 = self._t1
        t2 = self._t2
        basis1 = self._basis1
        basis2 = self._basis2
        factor = self._factor

        m, Dm, DDm = self._model(p, t1, t2, basis1, basis2, factor, component)
        DmDm = Dm[:, None, :] * Dm[None, :, :]
        l, Dl, DDl = self._loss(d, m)

        if l < self.__obj:
            self.__pars_prev = np.atleast_1d(self.__pars)
        self.__pars = p
        self.__obj = l
        self.__jac = np.sum(Dl * Dm, axis=-1)
        self.__hess = np.sum(DDl * DmDm + Dl * DDm, axis=-1)

    def _model(self, p, t1, t2, basis1, basis2, factor, component, D1=1, D2=1):
        Dany = D1 + D2
        if self._order == 0:
            b0 = p
            e0 = exp(b0) * (t2 - t1)

            m = factor * e0 + component
            Dm = factor * np.array([e0]) if D1 else None
            DDm = factor * np.array([
                [e0]
            ]) if D2 else None

        elif self._order == 1:
            b0, b1 = p
            if b1 != 0.0:
                rate1 = exp(p @ basis1)
                rate2 = exp(p @ basis2)
                x1 = b1 * t1 - 1.0 if Dany else None
                x2 = b1 * t2 - 1.0 if Dany else None
                e0 = (rate2 - rate1) / b1
                e1 = (rate2 * x2 - rate1 * x1) / b1 ** 2.0 if Dany else None
                e2 = (rate2 * (x2 * x2 + 1.0)
                      - rate1 * (x1 * x1 + 1.0)) / b1 ** 3.0 if D2 else None
            else:
                e0 = exp(b0) * (t2 - t1)
                e1 = exp(b0) * (t2 * t2 - t1 * t1) / 2.0 if Dany else None
                e2 = exp(b0) * (
                            t2 * t2 * t2 - t1 * t1 * t1) / 3.0 if D2 else None

            m = factor * e0 + component
            Dm = factor * np.array([e0, e1]) if D1 else None
            DDm = factor * np.array([
                [e0, e1],
                [e1, e2]
            ]) if D2 else None

        elif self._order == 2:
            b0, b1, b2 = p
            if b2 != 0.0:
                sqrt_b2 = np.sqrt(b2, dtype=float if b2 > 0.0 else complex)
                rate1 = exp(p @ basis1)
                rate2 = exp(p @ basis2)
                x1 = 2.0 * b2 * t1 - b1
                x2 = 2.0 * b2 * t2 - b1
                dawson1 = np.real(
                    dawsn((2.0 * b1 + x1) / (2.0 * sqrt_b2)) / sqrt_b2
                )
                dawson2 = np.real(
                    dawsn((2.0 * b1 + x2) / (2.0 * sqrt_b2)) / sqrt_b2
                )
                e0 = (rate2 * dawson2
                      - rate1 * dawson1)
                e1 = (rate2 * (1.0 - b1 * dawson2)
                      - rate1 * (1.0 - b1 * dawson1)) \
                     / (2.0 * b2)
                e2 = (rate2 * (x2 + (b1 ** 2.0 - 2.0 * b2) * dawson2)
                      - rate1 * (x1 + (b1 ** 2.0 - 2.0 * b2) * dawson1)) \
                     / (4.0 * b2 ** 2.0)
                e3 = (rate2 * ((b1 ** 2.0 - 4.0 * b2 + x2 * (b1 + x2))
                               + (b1 ** 3.0 - 6.0 * b1 * b2) * dawson2)
                      - rate1 * ((b1 ** 2.0 - 4.0 * b2 + x1 * (b1 + x1))
                                 + (b1 ** 3.0 - 6.0 * b1 * b2) * dawson1)) \
                     / (8.0 * b2 ** 3.0)
                e4 = (rate2 * (
                            (4.0 * b1 * b2 + x2 * (2.0 * b1 ** 2.0 - 6.0 * b2
                                                   + x2 * (2.0 * b1 + x2)))
                            + (
                                        b1 ** 4.0 - 12.0 * b1 ** 2.0 * b2 + 12.0 * b2 ** 2.0)
                            * dawson2)
                      - rate1 * ((4.0 * b1 * b2 + x1 * (
                                    2.0 * b1 ** 2.0 - 6.0 * b2
                                    + x1 * (2.0 * b1 + x1)))
                                 + (
                                             b1 ** 4.0 - 12.0 * b1 ** 2.0 * b2 + 12.0 * b2 ** 2.0)
                                 * dawson1)) \
                     / (16.0 * b2 ** 4.0)
            elif b1 != 0.0:
                rate1 = exp(p @ basis1)
                rate2 = exp(p @ basis2)
                x1 = b1 * t1 - 1.0 if Dany else None
                x2 = b1 * t2 - 1.0 if Dany else None
                e0 = (rate2 - rate1) / b1
                e1 = (rate2 * x2 - rate1 * x1) / b1 ** 2.0 if Dany else None
                e2 = (rate2 * (x2 * x2 + 1.0)
                      - rate1 * (x1 * x1 + 1.0)) / b1 ** 3.0 if Dany else None
                e3 = (rate2 * (-2.0 + x2 * (3.0 + x2 * x2))
                      - rate1 * (-2.0 + x1 * (3.0 + x1 * x1))) / b1 ** 4.0 \
                    if D2 else None
                e4 = (rate2 * (9.0 + x2 * (-8.0 + x2 * (6.0 + x2 * x2)))
                      - rate1 * (9.0 + x1 * (-8.0 + x1 * (6.0 + x1 * x1)))) \
                     / b1 ** 5.0 if D2 else None
            else:
                e0 = exp(b0) * (t2 - t1)
                e1 = exp(b0) * (t2 * t2 - t1 * t1) / 2.0 if Dany else None
                e2 = exp(b0) * (
                            t2 * t2 * t2 - t1 * t1 * t1) / 3.0 if Dany else None
                e3 = exp(b0) * (t2 ** 4.0 - t1 ** 4.0) / 4.0 if D2 else None
                e4 = exp(b0) * (t2 ** 5.0 - t1 ** 5.0) / 5.0 if D2 else None
            m = factor * e0 + component
            Dm = factor * np.array([e0, e1, e2]) if D1 else None
            DDm = factor * np.array([
                [e0, e1, e2],
                [e1, e2, e3],
                [e2, e3, e4]
            ]) if D2 else None

        return [i for i in [m, Dm, DDm] if i is not None]

    def _loss(self, data, model):
        l = np.sum(model - data * np.log(model))
        Dl = 1.0 - data / model
        DDl = data / model / model
        if np.any(np.isnan(DDl)):
            print(model)
        return l, Dl, DDl

    def _eval_count(self, rate, exposure):
        """
        Evaluate the expected counts given the rate and exposure, including a
        contribution from the minimum count rate (``self.__floor``).

        Parameters
        ----------
        rate : scalar or (``nchans``, N) array_like
            The rate.
        exposure : (N,) array_like
            The exposure.

        Returns
        -------
        counts : (``nchans``, N) array_like
            The expected counts.

        """
        if type(rate) is float:
            counts = np.full(
                (self._nchans, len(exposure)),
                (rate + self.__floor) * exposure
            )
        else:
            rate = np.atleast_2d(np.asarray(rate, dtype=np.float64))
            if rate.shape != (self._nchans, len(exposure)):
                raise ValueError(
                    f'the shape of ``rate`` should be ({self._nchans}, '
                    f'{len(exposure)})'
                )
            counts = (rate + self.__floor) * exposure
        return counts

    def _eval_basis(self, t):
        """
        Evaluate the basis functions for each time bin.

        Parameters
        ----------
        t : (N,) array_like
            The middle of time bins.

        Returns
        -------
        basis : (``order`` + 1, N) ndarray
            The basis functions for each time bin.

        """
        t_adj = self._scale * (t - self._shift)
        basis = np.array([t_adj ** i for i in np.arange(self._order + 1.0)])
        return t_adj, np.atleast_2d(basis)