# -*- coding: utf-8 -*-
"""
@author: xuewc
"""

import warnings
import numpy as np


class PolynomialFitter:
    def __init__(self, counts, tstart, tstop, exposure, rank_warn=True):
        self._counts = np.atleast_2d(np.asarray(counts, dtype=np.float64))
        self._t0 = tstart[0]
        self._tstart = np.asarray(tstart, dtype=np.float64) - self._t0
        self._tstop = np.asarray(tstop, dtype=np.float64) - self._t0
        self._exposure = np.asarray(exposure, dtype=np.float64)
        self._nchans, self._ntimes = self._counts.shape
        
        self._fit_method = '2pass'
        self._pchi2 = None
        self._dof = None
        self._order = None
        self._coeffs = None
        self._covars = None
        self._rank_warn = rank_warn
    
    
    @property
    def fit_method(self):
        return self._fit_method
    
    
    @fit_method.setter
    def fit_method(self, method):
        method_list = [
            '2pass',
            'mle',
        ]
        if method in method_list:
            self._fit_method = method
        else:
            raise ValueError(
                f'\nmethod ``{method}`` not supported, available choices are:'
                '\n'
                '+-----------+---------------------------------------+\n'
                '| ``2pass`` | two-pass fitting procedure            |\n'
                '+-----------+---------------------------------------+\n'
                '|  ``mle``  | maximum Poisson likelihood estimation |\n'
                '+-----------+---------------------------------------+'
            )
    
    
    @property
    def test_method(self):
        return 'pchi2'
    
    
    @property
    def test_statistic(self):
        return self._pchi2
    
    
    @property
    def dof(self):
        return self._dof
    
    
    @property
    def rank_warn(self):
        return self._rank_warn
    
    
    @rank_warn.setter
    def rank_warn(self, boolean):
        if type(boolean) is bool:
            self._rank_warn = boolean
        else:
            raise ValueError(
                '``rank_warn`` must be ``True`` or ``False``'
            )
    
    
    def fit(self, order=1):
        """
        Fit the count data with a polynomial.

        Parameters
        ----------
        order : int, optional
            The order of the polynomial. The default is 1.

        Returns
        -------
        model : (``nchans``, ``ntimes``) ndarray
            The model values of each channel over the given time bins.
        model_uncert : (``nchans``, ``ntimes``) ndarray
            The model uncertainties of each channel over the given time bins.

        Warns
        -----
        RankWarning
            The rank of the coefficient matrix in the least-squares fit is
            deficient.
        RuntimeWarning
            The background model has negative value.

        """
        # check polynomial order
        if order < 0:
            raise ValueError('polynomial order must be non-negative')
        self._order = order
        
        # fit
        if self.fit_method == '2pass':
            self._fit_2pass()
        elif self.fit_method == 'mle':
            self._fit_2pass()
            self._fit_mle(self._coeffs)
        elif self.fit_method == 'irls':
            self._fit_irls()
        
        # evaluate model
        model = self._eval_model(self._tstart, self._tstop, None)

        # check if model has negative value
        negative = np.any(model < 0.0, axis=1)
        if np.any(negative):
            err_chans = np.where(negative)[0].astype(str)
            warnings.warn(
                f'background model has negative value in following channel(s) '
                f'(starting from 0): {", ".join(err_chans)}\n\n'
                f'This error maybe eliminated by reducing the order of '
                f'polynomial (the current is {self._order}).',
                RuntimeWarning
            )
        
        # evaluate model uncertainty
        model_uncert = self._eval_uncertainty(self._tstart, self._tstop, None)

        # evaluate goodness-of-fit   
        self._pchi2, self._dof = self._eval_gof(model)
        
        return model, model_uncert
    
    
    def interpolate(self, tstart, tstop, exposure=None):
        """
        Interpolate the fitted model over the given time bins.

        Parameters
        ----------
        tstart : (N,) array_like
            The left edge of time bins.
        tstop : (N,) array_like
            The right edge of time bins.
        exposure : (N,) array_like or None, optional
            The exposure (or live time) of time bins. Equals to (``tstart`` -
            ``tstop``) if None. The default is None.

        Returns
        -------
        interp : (``nchans``, N) ndarray
            The model values of each channel over the given time
            bins.
        interp_uncert : (``nchans``, N) ndarray
            The uncertainties of model of each channel over
            the given time bins.

        """
        tstart = tstart - self._t0
        tstop = tstop - self._t0
        interp = self._eval_model(tstart, tstop, exposure)
        interp_uncert = self._eval_uncertainty(tstart, tstop, exposure)
        return interp, interp_uncert
    
    
    def _fit_2pass(self):
        """
        Model variances are used for chi-squared via two fitting passes.
        Adapted from RMfit polynomial fitter.
        """
        self._coeffs = np.empty((self._nchans, self._order+1))
        self._covars = np.empty((self._nchans, self._order+1, self._order+1))
        
        # two-pass fitting
        X = self._eval_basis(self._tstart, self._tstop, self._exposure)
        y = self._counts
        
        # first pass uses the weights calculated from data count
        zero = (self._counts == 0)
        w = np.piecewise(
            self._counts,
            condlist=[zero, ~zero],
            funclist=[lambda c: 0.0, lambda c: np.sqrt(1/c)]
        )
        
        for i in range(self._nchans):
            self._coeffs[i] = self._weighted_leastsq(X, y[i], w[i], False)
        
        # second pass uses the weights calculated from model count
        model = self._eval_model(self._tstart, self._tstop, self._exposure)
        
        # check if model has negative value
        negative = np.any(model < 0.0, axis=1)
        if np.any(negative):
            err_chans = np.where(negative)[0].astype(str)
            warnings.warn(
                f'background model has negative value in following channel(s) '
                f'(starting from 0): {", ".join(err_chans)}\n\n'
                f'This error maybe eliminated by reducing the order of '
                f'polynomial (the current is {self._order}).',
                RuntimeWarning
            )
        
        positive = (model > 0.0)
        w = np.piecewise(
            model,
            condlist=[~positive, positive],
            funclist=[lambda m: 0.0, lambda m: np.sqrt(1/m)]
        )
        
        for i in range(self._nchans):
            self._coeffs[i], self._covars[i] = \
                self._weighted_leastsq(X, y[i], w[i])
    
    
    def _fit_iwls(self):
        """
        Iterated weighted least-squares fit.
        """
        self._coeffs = np.empty((self._nchans, self._order+1))
        self._covars = np.empty((self._nchans, self._order+1, self._order+1))
        
        X = self._eval_basis(self._tstart, self._tstop, self._exposure)
        y = self._counts
        var = y
        w = # initial weights set to one
        
        for i in range(self._nchans):
            self._coeffs[i] = self._weighted_leastsq(X, y[i], w[i], False)
        
    
    
    def _eval_model(self, tstart, tstop, exposure=None):
        """
        Evaluate the model values over the given time bins.

        Parameters
        ----------
        tstart : (N,) array_like
            The left edge of time bins.
        tstop : (N,) array_like
            The right edge of time bins.
        exposure : (N,) array_like or None, optional
            The exposure (or live time) of time bins. Equals to (``tstart`` -
            ``tstop``) if None. The default is None.

        Returns
        -------
        model : (``nchans``, N) ndarray
            The model values of each channel over the given time bins.

        """
        basis = self._eval_basis(tstart, tstop, exposure)
        model = self._coeffs @ basis.T
        return model
    
    
    def _eval_uncertainty(self, tstart, tstop, exposure=None):
        """
        Evaluate the uncertainty of the model values over the given time bins,
        based on the covariance matrix of the model coefficients.

        Parameters
        ----------
        tstart : (N,) array_like
            The left edge of time bins.
        tstop : (N,) array_like
            The right edge of time bins.
        exposure : (N,) array_like or None, optional
            The exposure (or live time) of time bins. Equals to (``tstart`` -
            ``tstop``) if None. The default is None.

        Returns
        -------
        uncertainty : (``nchans``, N) ndarray
            The model uncertainties of each channel over the given time bins.

        """
        X = self._eval_basis(tstart, tstop, exposure)
        # expand 1st X to shape(N, 1, 1, ``order`` + 1)
        # expand 2nd X to shape(N, 1, ``order`` + 1, 1)
        # matmul of X1, covars (`nchans`, ``order`` + 1, ``order`` + 1)
        # and X2, gives var (N, ``nchans``, 1, 1)
        var = X[:, None, None, :] @ self._covars @ X[:, None, :, None]
        # remove some axes from var matrix, from shape (N, ``nchans``, 1, 1)
        # to shape (N, ``nchans``), then sqrt(var).T gives the uncertainty
        uncertainty = np.sqrt(np.squeeze(var, axis=(2, 3))).T
        return uncertainty
    
    
    def _eval_gof(self, model):
        """
        Evaluate goodness-of-fit for the fitted model. Pearson chi-squared is
        used here.

        Parameters
        ----------
        model : (``nchans``, ``ntimes``) array_like
            The fitted model.

        Returns
        -------
        chi2 : (``nchans``,) ndarray
            Pearson chi-squared.
        dof : (``nchans``,) ndarray
            Degrees-of-freedom of each fitted channel.

        """
        mask = (model > 0.0) # (self._counts > 0.0)
        chi2 = np.array([
            np.sum(
                (self._counts[i, mask[i]] - model[i, mask[i]])**2 \
                    / model[i, mask[i]]
            )
            for i in range(self._nchans)
        ])
        dof = np.sum(mask, axis=1) - (self._order + 1.0)
        return chi2, dof
    
    
    def _eval_basis(self, tstart, tstop, exposure=None):
        """
        Evaluate the basis functions for each time bin, i.e. polynomials of
        different orders integrated over each time bin. Exposure factor may be
        applied.

        Parameters
        ----------
        tstart : (N,) array_like
            The left edge of time bins.
        tstop : (N,) array_like
            The right edge of time bins.
        exposure : (N,) array_like or None, optional
            The exposure (or live time) of time bins. Equals to (``tstart`` -
            ``tstop``) if None. The default is None.

        Returns
        -------
        basis : (N, ``order`` + 1) ndarray
            The basis functions for each time bin.

        """
        factor = 1 if exposure is None else exposure / (tstop - tstart)
        basis = np.array([
            (tstop**(i + 1.0) - tstart**(i + 1.0)) * factor
            for i in range(self._order + 1)
        ]).T
        return np.atleast_2d(basis)
    
    
    def _weighted_leastsq(self, X, y, w, return_cov=True):
        """
        Return the least-squares solution to a linear matrix equation, which
        minimizes the weighted Euclidean 2-norm :math:`||W^{1/2} (y - bX)||`.
        This function is adapted from `numpy` polynomial fitter.
        
        Parameters
        ----------
        X : (M, N) array_like
            "Coefficient" matrix.
        y : (M,) array_like
            Ordinate or "dependent variable" values.
        w : (M,) array_like
            Weights to apply to the y-coordinates of the sample points. For
            gaussian uncertainties, use 1/sigma (not 1/sigma**2).
        return_cov : bool, optional
            Whether to return covariance. The default is True.

        Returns
        -------
        b : (N,) ndarray
            Weighted least-squares solution.
        cov : (N, N) ndarray
            The covariance matrix of the parameter estimates. The diagonal of
            this matrix are the variance estimates for each parameter.
        
        Warns
        -----
        RankWarning
            The rank of the coefficient matrix in the least-squares fit is
            deficient.
        
        """
        order = X.shape[1]
        
        # return if all values of y are zeros
        if all(y == 0):
            b = np.full(order, 0.0)
            if return_cov:
                cov = np.full((order, order), 0.0)
                return b, cov
            else:
                return b
        
        # set up least squares equation with weight
        WX = w[:, None] * X
        Wy = w * y
        
        # scale WX to improve condition number and solve
        scale = np.sqrt(np.square(WX).sum(axis=0))
        scale[scale == 0] = 1
        b, resids, rank, s = np.linalg.lstsq(WX / scale, Wy, rcond=None)
        b = b / scale
        
        # warn on rank reduction, which indicates an ill conditioned matrix
        if self._rank_warn and rank != order:
            msg = 'The fit may be poorly conditioned'
            warnings.warn(msg, RankWarning, stacklevel=2)
        
        if return_cov:
            # scale the covariance matrix, to reduce the potential bias of
            # weights
            fac = resids / (X.shape[0] - order)
            scaled_cov = fac * np.linalg.inv(WX.T @ WX)
            return b, scaled_cov
        else:
            return b
    
    
    def _iwls(self, X, y, return_cov=True, delta=1e-5):
        order = X.shape[1]
        
        # return if all values of y are zeros
        if all(y == 0.0):
            b = np.full(order, 0.0)
            if return_cov:
                cov = np.full((order, order), 0.0)
                return b, cov
            else:
                return b
        
        # initialize the weight
        w = np.zeros_like(var, dtype=float)
        idx = var > 0.0
        w[idx] = np.sqrt(1/var[idx])
        w = np.ones_like(1.0/var, dtype=np.float64)
        
        # set up least squares equation with weight
        WX = w[:, None] * X
        Wy = w * y
        
        # scale WX to improve condition number and solve
        scale = np.sqrt(np.square(WX).sum(axis=0))
        scale[scale == 0] = 1
        b, resids, rank, s = np.linalg.lstsq(WX / scale, Wy, rcond=None)
        b = b / scale
        
        # 收敛判据：||delta p|| <= 1e5 or deviance <= 1e6
        # edm = jac_i.T @ inv(hess_i) @ jac_i
        # vtest = np.mean(
        #    (diag(inv(hess_i)) - diag(inv(hess_{i-1}))) / diag(inv(hess_{i-1}))
        # )
        # convergence: (edm<eps and vtest < veps) or (edm<eps*1.0e-5)



class RankWarning(UserWarning):
    """Issued by fit when the design matrix is rank deficient."""
    pass


class RuntimeWarning(UserWarning):
    """"Issued by fit when the background model has negative value."""
    pass