#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:00:16 2023

@author: xuewc
"""

import torch
from torch.autograd import gradcheck

def d_igamma_dp_series_expansion(p: torch.Tensor, x: torch.Tensor, eps: float = 1e-5, n_max: int = 100) -> torch.Tensor:
    """
    Algorithm AS 187: Derivatives of the Incomplete Gamma Integral
    Author(s): R. J. Moore
    Stable URL: http://www.jstor.org/stable/2348014

    Calculate d igamma(p,x) / dp using a series expansion (cf. Moore, 1982)
    This is valid for p <= x <= 1 and also for x < p.
    Notation from Moore 1982
    Args:
        p: actual observations
        x: model prediction
        eps: the minimal accepted accuracy when the series expansion is
        n_max: max number of iteration
    Returns:
        Calculate d igamma(p,x) valid for p <= x <= 1
    """
    log_x = torch.log(x)
    p_plus_1 = p + 1.0
    log_f = p * log_x - torch.lgamma(p_plus_1) - x
    f = log_f.exp()
    df_dp = f * (log_x - torch.polygamma(0, p_plus_1))

    C_old = torch.ones_like(p)
    d_C_old_dp = torch.zeros_like(p)
    S = C_old
    dS_dp = d_C_old_dp

    idx_notconv = torch.arange(0, p.numel(), dtype=torch.long)

    converged = False

    for n in range(1, n_max):
        ppn = torch.reciprocal(p + n)
        C_new = x * ppn * C_old
        d_C_new_dp = C_new * (torch.reciprocal(C_old) * d_C_old_dp - ppn)
        # update indices
        idx_notconv = torch.where(torch.abs(C_new) > S * eps)[0]

        if idx_notconv.numel() > 0:
            S[idx_notconv] = (S + C_new)[idx_notconv]
            dS_dp[idx_notconv] = (dS_dp + d_C_new_dp)[idx_notconv]
        else:
            converged = True
            break

        C_old = C_new
        d_C_old_dp = d_C_new_dp

    return S * df_dp + f * dS_dp


def d_igamma_dp_cf_expansion(p: torch.Tensor, x: torch.Tensor, eps: float = 1e-5, n_max: int = 100) -> torch.Tensor:
    """
    Algorithm AS 187: Derivatives of the Incomplete Gamma Integral
    Author(s): R. J. Moore
    Stable URL: http://www.jstor.org/stable/2348014

    Calculate d igamma(p,x) / dp using a continued fraction expansion (cf. Moore, 1982)
    This is valid for the entire input domain outside of { {p <= x <= 1} U {x < p} }.
    Notation from Moore 1982
    Args:
        p: actual observations
        x: model prediction
        eps: the minimal accepted accuracy when the series expansion is
        n_max: max number of iteration
    Returns:
        Calculate d igamma(p,x) valid for p <= x <= 1
    """
    log_x = torch.log(x)
    log_f = p * log_x - torch.lgamma(p) - x
    f = log_f.exp()
    df_dp = f * (log_x - torch.polygamma(0, p))

    A0 = torch.ones_like(p)
    B0 = x
    A1 = x + 1.0
    B1 = x * (2.0 - p + x)

    # init derivatives
    dA0_dp = torch.zeros_like(p)
    dA1_dp = torch.zeros_like(p)
    dB0_dp = torch.zeros_like(p)
    dB1_dp = -x

    S = torch.zeros_like(p)
    S_old = torch.zeros_like(p)
    S_temp = torch.zeros_like(p)
    dS_dp = torch.zeros_like(p)

    # tensor indices that haven't converged yet (initially all of them)
    idx_notconv = torch.arange(0, p.numel(), dtype=torch.long)

    converged = False

    for n in range(2, n_max):
        a, b = (n - 1) * (p - n), 2.0 * n - p + x
        A2 = b * A1 + a * A0
        B2 = b * B1 + a * B0
        dA2_dp = b * dA1_dp - A1 + a * dA0_dp + (n - 1) * A0
        dB2_dp = b * dB1_dp - B1 + a * dB0_dp + (n - 1) * B0

        S_old[idx_notconv] = (A1 / B1)[idx_notconv]
        S_temp[idx_notconv] = (A2 / B2)[idx_notconv]

        # update indices
        idx_notconv = torch.where(torch.abs(S_temp - S_old) > S_temp * eps)[0]

        if idx_notconv.numel() > 0:
            S[idx_notconv] = S_temp[idx_notconv]
            # TODO: refactor in the future - this might be numerically unstable
            dS_dp[idx_notconv] = (dA2_dp / B2 - S_temp * (dB2_dp / B2))[idx_notconv]
        else:
            converged = True
            break

        # update intermediates and partials
        A0, B0 = A1, B1
        A1, B1 = A2, B2
        dA0_dp, dB0_dp = dA1_dp, dB1_dp
        dA1_dp, dB1_dp = dA2_dp, dB2_dp

    return -S * df_dp - f * dS_dp


def d_igamma_dp(p: torch.Tensor, x: torch.Tensor, eps: float = 1e-5, n_max: int = 200) -> torch.Tensor:
    """
    Algorithm AS 187: Derivatives of the Incomplete Gamma Integral
    Author(s): R. J. Moore
    Stable URL: http://www.jstor.org/stable/2348014

    Calculate the derivative of the incomplete gamma function as the series expansion
    Notation from Moore 1982
    Args:
        p: actual observations
        x: model prediction
        eps: the minimal accepted accuracy when the series expansion is
        n_max: max number of iteration
    Returns:
        derivative of the incomplete gamma function
    """
    # uses the notation in (Moore, 1982)
    d_igamma = torch.zeros_like(p)
    idx = torch.logical_or(torch.logical_and(p <= x, x <= 1), x <= p)
    idx_series = idx.nonzero(as_tuple=True)[0]
    idx_continued_fraction = (~idx).nonzero(as_tuple=True)[0]
    if idx_series.numel() > 0:
        p_s, x_s = p[idx_series], x[idx_series]
        d_igamma[idx_series] = d_igamma_dp_series_expansion(p_s, x_s, eps, n_max)
    if idx_continued_fraction.numel() > 0:
        p_cf, x_cf = p[idx_continued_fraction], x[idx_continued_fraction]
        d_igamma[idx_continued_fraction] = d_igamma_dp_cf_expansion(p_cf, x_cf, eps, n_max)
    return d_igamma


class CustomIGamma(torch.autograd.Function):
    """
    PyTorch IGamma function implementation
    https://pytorch.org/docs/stable/distributions.html
    """

    def forward(self, a: torch.Tensor, z: torch.Tensor):  # type: ignore # pylint: disable=W0221
        cdf_ = torch.igamma(a, z)
        self.save_for_backward(a, z)
        return cdf_

    def backward(self, grad_output):  # pylint: disable=W0221
        a, z = self.saved_tensors
        d_igamma_a = d_igamma_dp(a, z)
        # The exact formula can be found:
        # https://www.wolframalpha.com/input/?i=d+GammaRegularized%5Ba%2C+0%2C+z%5D+%2F+dz
        d_igamma_z = (-z + (a - 1.0) * torch.log(z) - torch.lgamma(a)).exp()
        return (grad_output * d_igamma_a, grad_output * d_igamma_z)


class GammaCDF(torch.autograd.Function):
    """
    PyTorch Gamma distribution CDF implementation. This implementation solves the following issue raised on the PyTorch forum:
    https://github.com/pytorch/pytorch/issues/41637
    """

    def forward(self, x: torch.Tensor, k: torch.Tensor, psi: torch.Tensor):  # type: ignore # pylint: disable=W0221
        cdf_ = torch.igamma(k, x / psi)
        self.save_for_backward(k, x, psi)
        return cdf_

    def backward(self, grad_output, eps=1e-5):  # pylint: disable=W0221
        k, x, psi = self.saved_tensors
        a = k

        # Numerical stability
        x[x < eps] = eps
        z = x / psi

        d_igamma_a = d_igamma_dp(a, z, eps)
        d_igamma_z = (-z + a * torch.log(z) - torch.lgamma(a)).exp()
        return (grad_output * d_igamma_z / x, grad_output * d_igamma_a, -grad_output * d_igamma_z / psi)
