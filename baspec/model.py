"""
Created at 02:14:28 on 2023-05-09

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import numpy as np
import pytensor.tensor as pt
from pytensor.gradient import grad_not_implemented


class SpectralModel(pt.Op):
    otypes = [pt.dvector]
    def __init__(self, *args, numeric_grad='f'):
        # difference approximation: n=no, c=central, f=forward, and b=backward
        self._args = [
            i if type(i) == pt.TensorVariable else pt.constant(i, dtype=float)
            for i in args
        ]

        self._numeric_grad = numeric_grad
        if numeric_grad != 'n':
            grad_eps = 1e-7 if numeric_grad != 'c' else 1e-4
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
            if isinstance(self, XspecModel):
                self.ebins = self.ebins.tolist()
            return super().__call__(*self._args, *args[1:], **kwargs)
        else:
            return super().__call__(*args, **kwargs)

    def perform(self, node, inputs, output_storage, params=None):
        # return value
        output_storage[0][0] = self._perform(*inputs, self.ebins)

    def grad(self, inputs, output_grads):
        # return grad Op, in backward mode
        if self._numeric_grad != 'n':
            if self._numeric_grad != 'c':
                self._flux = self(*inputs, call_from_grad=True)
            return [
                pt.dot(output_grads[0], self._grad(inputs, i))
                for i in range(len(inputs))
            ]
        else:
            return [
                grad_not_implemented(self, i, v)
                for i, v in enumerate(inputs)
            ]

    def _perform(self, ebins, *inputs):
        raise NotImplementedError

    def _grad(self, inputs, idx):
        # https://en.wikipedia.org/wiki/Finite_difference
        # https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
        if self._numeric_grad == 'f':
            # forward difference approximation
            args = inputs[:]
            args[idx] = args[idx] + self._eps
            flux_eps = self(*args, call_from_grad=True)
            return (flux_eps - self._flux)/self._eps
        elif self._numeric_grad == 'c':
            # central difference approximation, accurate when eval hessian
            args = inputs[:]
            args[idx] = args[idx] + self._eps
            flux_peps = self(*args, call_from_grad=True)
            args = inputs[:]
            args[idx] = args[idx] - self._eps
            flux_meps = self(*args, call_from_grad=True)
            return (flux_peps - flux_meps) / (2.0 * self._eps)
        elif self._numeric_grad == 'b':
            # backward difference approximation
            args = inputs[:]
            args[idx] = args[idx] - self._eps
            flux_eps = self(*args, call_from_grad=True)
            return (self._flux - flux_eps)/self._eps
        else:
            raise ValueError(
                f'wrong input ({self._numeric_grad}) for `numeric_grad`, '
                'supported difference approximation types are "c" for central,'
                ' "f" for forward, and "b" for backward'
            )


class BlackBody(SpectralModel):
    from pyda.numerics.specfun import bbody as _perform
    itypes = [pt.dscalar]


class BlackBodyRad(SpectralModel):
    from pyda.numerics.specfun import bbodyrad as _perform
    itypes = [pt.dscalar]


class CutoffPowerlaw(SpectralModel):
    from pyda.numerics.specfun import cutoffpl as _perform
    itypes = [pt.dscalar, pt.dscalar]


class Powerlaw(SpectralModel):
    from pyda.numerics.specfun import powerlaw as _perform
    itypes = [pt.dscalar]


class XspecModel(SpectralModel):
    try:
        from mxspec._pymXspec import callModFunc as _callModFunc
    except:
        raise ImportError('Xspec is not installed')

    def __init__(self, xsmodel, *args, **kwargs):
        self._xsmodel = xsmodel
        self.itypes = [pt.dscalar for i in args]
        return super().__init__(*args, **kwargs)

    def _perform(self, *args):
        *inputs, ebins = args
        res = []
        self._callModFunc(self._xsmodel, ebins, inputs, res, [], 1, '')
        return np.asarray(res, dtype=np.float64, order='C')


if __name__ == '__main__':
    import pymc as pm
    ph_ebins = np.geomspace(1, 10, 101)
    t = 100
    # src_true = (t*np.exp(2.0)*Powerlaw(1.5))(ph_ebins)
    src_true = CutoffPowerlaw(1.5, 4.0)(ph_ebins) * 50 * t
    np.random.seed(42)
    data = np.random.poisson(src_true.eval())
    with pm.Model() as model:
        # PhoIndex = pm.Flat('PhoIndex')
        # norm = pt.exp(pm.Flat('norm'))
        PhoIndex = pm.Uniform('PhoIndex', lower=0, upper=5)
        Ecut = pm.Uniform('Ecut', lower=0, upper=20)
        norm = pm.Uniform('norm', lower=0, upper=100)
        m = norm * CutoffPowerlaw(PhoIndex, Ecut, numeric_grad='f')
        # m = norm * XspecModel('powerlaw', PhoIndex)#Powerlaw(PhoIndex)
        src = m(ph_ebins) * t
        loglike = pm.Poisson('N', mu=src, observed=data)
        # hess = pm.hessian(model.observedlogp, model.continuous_value_vars)
        # pi_J = pt.log(pm.math.det(hess)) / 2.0
        # pm.Potential('pi_J', pi_J)
        idata = pm.sample(10000, target_accept=0.95, random_seed=42,
                          chains=4, progressbar=True)
        # pmap = pm.find_MAP()
    pm.plot_trace(idata)