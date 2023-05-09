"""
Created at 02:14:28 on 2023-05-09

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import numpy as np
import pytensor.tensor as pt

from numba import njit
from pytensor.gradient import grad_not_implemented
from pytensor.ifelse import ifelse

from pyda.numerics.specfun import cutoffpl

class SpectralModel(pt.Op):
    def __init__(self, *args, numeric_grad='f'):
        # difference approximation: c=central, f=forward, and b=backward
        self._args = args
        grad_eps = 1e-7 if numeric_grad != 'c' else 1e-4
        self._numeric_grad = numeric_grad
        self._eps = pt.constant(grad_eps, dtype=float)

    def __call__(self, *args, **kwargs):
        return self._wrapped_call(*args, **kwargs)

    def __add__(self, other):
        return self._excute_operation(operator='__add__', other=other)

    def __radd__(self, other):
        return self._excute_operation(operator='__radd__', other=other)

    def __sub__(self, other):
        return self._excute_operation(operator='__sub__', other=other)

    def __rsub__(self, other):
        return self._excute_operation(operator='__rsub__', other=other)

    def __mul__(self, other):
        return self._excute_operation(operator='__mul__', other=other)

    def __rmul__(self, other):
        return self._excute_operation(operator='__rmul__', other=other)

    def __matmul__(self, other):
        return self._excute_operation(operator='__matmul__', other=other)

    def __rmatmul__(self, other):
        return self._excute_operation(operator='__rmatmul__', other=other)

    def __truediv__(self, other):
        return self._excute_operation(operator='__truediv__', other=other)

    def __rtruediv__(self, other):
        return self._excute_operation(operator='__rtruediv__', other=other)

    def __pow__(self, other):
        return self._excute_operation(operator='__pow__', other=other)

    def __rpow__(self, other):
        return self._excute_operation(operator='__rpow__', other=other)

    def __neg__(self):
        return self._excute_operation(operator='__neg__', other=None)

    def __pos__(self):
        return self._excute_operation(operator='__pos__', other=None)

    def _excute_operation(self, operator, other):
        new_model = SpectralModel()
        new_model._wrap_call(model=self, operator=operator, other=other)
        return new_model

    def _wrap_call(self, model, operator, other):
        if isinstance(other, SpectralModel):
            func = lambda ebins: getattr(model(ebins), operator)(other(ebins))
        elif other is not None:
            func = lambda ebins: getattr(model(ebins), operator)(other)
        else:
            func = lambda ebins: getattr(model(ebins), operator)()

        self._wrapped_call = func

    def _wrapped_call(self, *args, **kwargs):
        if not kwargs.pop('call_from_grad', False):
            self.ebins = np.asarray(args[0], dtype=np.float64, order='C')
            return super().__call__(*self._args, *args[1:], **kwargs)
        else:
            return super().__call__(*args, **kwargs)

    def perform(self, node, inputs, output_storage, params=None):
        # return value
        output_storage[0][0] = self._perform(self.ebins, *inputs)

    def grad(self, inputs, output_grads):
        # return grad Op, in backward mode
        self._flux = self(*inputs, call_from_grad=True)
        return [
            pt.dot(output_grads[0], self._grad(inputs, i))
            for i in range(len(inputs))
        ]
        # return [grad_not_implemented(self, i, v) for i,v in enumerate(inputs)]

    @staticmethod
    def _perform(ebins, *inputs):
        raise NotImplementedError

    def _grad(self, inputs, idx):
        # https://en.wikipedia.org/wiki/Finite_difference
        if self._numeric_grad == 'f':
            # forward difference approximation
            args = inputs[:]
            args[idx] = args[idx] + self._eps
            flux_eps = self(*args, call_from_grad=True)
            return (flux_eps - self._flux)/self._eps
        if self._numeric_grad == 'c':
            # central difference approximation, useful when hessian is needed
            args = inputs[:]
            args[idx] = args[idx] + self._eps
            flux_peps = self(*args, call_from_grad=True)
            args = inputs[:]
            args[idx] = args[idx] - self._eps
            flux_meps = self(*args, call_from_grad=True)
            return (flux_peps - flux_meps) / (2.0 * self._eps)


class Powerlaw(SpectralModel):
    itypes = [pt.dscalar]
    otypes = [pt.dvector]

    @staticmethod
    @njit('float64[::1](float64[::1], Array(float64, 0, "C"))', cache=True)
    def _perform(ebins, PhoIndex):
        one_minus_PhoIndex = 1.0 - PhoIndex
        if one_minus_PhoIndex != 0.0:
            N = ebins**one_minus_PhoIndex / one_minus_PhoIndex
        else:
            N = np.log(ebins)
        return N[1:] - N[:-1]

class CutoffPowerlaw(SpectralModel):
    itypes = [pt.dscalar, pt.dscalar]
    otypes = [pt.dvector]

    @staticmethod
    def _perform(ebins, PhoIndex, Ecut):
        return cutoffpl(PhoIndex, Ecut, ebins)


if __name__ == '__main__':
    import pymc as pm
    ph_ebins = np.geomspace(1, 10, 101)
    t = 10
    src_true = (t*np.exp(2.0)*Powerlaw(pt.constant(1.5,dtype=float)))(ph_ebins)
    np.random.seed(42)
    data = np.random.poisson(src_true.eval())
    with pm.Model() as model:
        # PhoIndex = pm.Flat('PhoIndex')
        # norm = pt.exp(pm.Flat('norm'))
        PhoIndex = pm.Uniform('PhoIndex', lower=0, upper=5)
        norm = pm.Uniform('norm', lower=0, upper=30)
        pl = norm * Powerlaw(PhoIndex, numeric_grad='c')
        src = pl(ph_ebins)
        loglike = pm.Poisson('N', mu=(src * t), observed=data)
        # hess = pm.hessian(model.observedlogp, model.continuous_value_vars)
        # pi_J = pt.log(pm.math.det(hess)) / 2.0
        # pm.Potential('pi_J', pi_J)
        idata = pm.sample(50000, target_accept=0.95, random_seed=42,
                          chains=4, progressbar=True)
        # pmap = pm.find_MAP()
    pm.plot_trace(idata)