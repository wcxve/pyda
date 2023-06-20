#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 06:47:53 2023

@author: xuewc
"""

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numba import njit

from time import sleep

class Powerlaw(pt.Op):
    itypes = [pt.dscalar]
    otypes = [pt.dvector]

    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64, order='C')
        self._grad = PowerlawGrad(self.ebins)
        # self.info = [[-2, np.nan]]

    def perform(self, node, inputs, outputs):
        # return value
        outputs[0][0] = self._perform(*inputs, self.ebins)

    def grad(self, inputs, outputs):
        # return grad Op, in backward mode
        g = self._grad(*inputs)
        return [pt.dot(outputs[0], g)]

    @staticmethod
    # @njit
    def _perform(alpha, ebins):
        # info.append([info[-1][0]+1, alpha])
        # print('FUNC', *info[-1])
        # sleep(1)
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
        # self.info = [[0, np.nan]]

    def perform(self, node, inputs, outputs):
        # return value
        outputs[0][0] = self._perform(*inputs, self.ebins, self.log_ebins)

    @staticmethod
    # @njit
    def _perform(alpha, ebins, log_ebins):
        # info.append([info[-1][0]+1, alpha])
        # print('GRAD', *info[-1], '\n\n')
        # sleep(0.1)
        if alpha != 1.0:
            v1 = 1.0 - alpha
            v2 = ebins ** v1
            dalpha = v2 * (1 - v1*log_ebins) / (v1*v1)
        else:
            dalpha = - log_ebins*log_ebins / 2.0

        return dalpha[1:] - dalpha[:-1]


class Powerlaw2(pt.Op):
    itypes = [pt.dscalar]
    otypes = [pt.dvector]

    def __init__(self, ebins):
        self.ebins = np.asarray(ebins, dtype=np.float64, order='C')
        self._grad = PowerlawGrad(self.ebins)
        # self.info = [[-2, np.nan]]

    def perform(self, node, inputs, outputs):
        # return value
        # print(dir(node), node, inputs, '\n\n')
        # print(node.clone_with_new_inputs())
        outputs[0][0] = self._perform(*inputs, self.ebins)

    def grad(self, inputs, outputs):
        # return grad Op, in backward mode
        # forward difference approximation
        print(print(dir(self)), self.default_output)
        raise
        h = pt.constant(1e-8)
        g = (self(inputs[0] + h) - self(inputs[0])) / h
        return [pt.dot(outputs[0], g)]

    @staticmethod
    # @njit
    def _perform(alpha, ebins):
        # info.append([info[-1][0]+1, alpha])
        # print('FUNC', *info[-1])
        # sleep(1)
        if alpha != 1.0:
            NE = ebins**(1.0 - alpha) / (1.0 - alpha)
        else:
            NE = np.log(ebins)

        return (NE[1:] - NE[:-1])
    
def flux(alpha, ebins):
    return np.diff(ebins**(1.0 - alpha) / (1.0 - alpha))
    

ebins = np.geomspace(1, 10, 101)
t = 10
src_true = t*np.exp(2.0)*flux(1.5, ebins)
data = np.random.poisson(src_true)
with pm.Model() as model:
    PhoIndex = pm.Flat('PhoIndex')
    norm = pt.exp(pm.Flat('norm'))
    pl = Powerlaw2(ebins)(PhoIndex)
    pl = norm * pl
    src = pl
    loglike = pm.Poisson('N', mu=src*t, observed=data)
    idata = pm.sample(50000, target_accept=0.95, random_seed=42, chains=1, progressbar=1)

# az.plot_trace(idata)

# with pm.Model() as model:
#     PhoIndex = pm.Flat('PhoIndex')
#     norm = pt.exp(pm.Flat('norm'))
#     pl = Powerlaw(ebins)(PhoIndex)
#     pl = norm * pl
#     src = pl
#     loglike = pm.Poisson('N', mu=src*t, observed=data)
#     idata = pm.sample(50000, target_accept=0.95, random_seed=42, chains=4, progressbar=1)

# az.plot_trace(idata)


