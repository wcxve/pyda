"""
Created at 00:24:56 on 2023-05-09

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import numpy as np
import pytensor.tensor as pt
from pytensor.ifelse import ifelse

class SpectralModel:
    def __call__(self, ebins):
        return self._flux(ebins)
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
        spectral_model = SpectralModel()
        spectral_model._wrap_flux(model=self, operator=operator, other=other)
        return spectral_model


    def _wrap_flux(self, model, operator, other):
        if isinstance(other, SpectralModel):
            flux = lambda ebins: getattr(model(ebins), operator)(other(ebins))
        elif other is not None:
            flux = lambda ebins: getattr(model(ebins), operator)(other)
        else:
            flux = lambda ebins: getattr(model(ebins), operator)()

        self._flux = flux


    def _flux(self, ebins):
        raise NotImplementedError


class Powerlaw(SpectralModel):
    def __init__(self, PhoIndex: pt.TensorVariable):
        self._one_minus_PhoIndex = 1.0 - PhoIndex

    def _flux(self, ebins: np.ndarray):
        ebins = np.asarray(ebins, dtype=np.float64, order='C')
        ebins_tensor = pt.constant(ebins)
        log_ebins_tensor = pt.log(pt.constant(ebins))
        N = ifelse(
            pt.neq(self._one_minus_PhoIndex, 0.0),
            ebins**self._one_minus_PhoIndex/self._one_minus_PhoIndex,
            log_ebins_tensor
        )
        return N[1:] - N[:-1]


class BBodyRad(SpectralModel):
    # TODO: this class needs to be rewritten
    def __init__(self, kT: pt.TensorVariable):
        self._kT = kT

    def _flux(self, ebins: np.ndarray):
        # this is from xspec
        N = len(ebins)
        ebins_tensor = pt.constant(ebins)
        flux = pt.tensor(dtype='float64', shpae=(N - 1,))

        el = ebins[0]
        x = el / kT
        if pm.math.leq(x, 1.0e-4):
            nl = el * kT  # limit_{el/kT->1} el*el/(exp(el/kT)-1) = el*kT
        elif pm.math.gt(x, 60.0):
            flux[:] = 0.0
            return flux
        else:
            nl = el * el / (pm.math.exp(x) - 1.0)

        # norm of 2-point approximation to integral
        norm = 1.0344e-3 / 2.0  # BBodyRad
        # kT2 = kT*kT
        # norm = 8.0525 / (kT2*kT2) / 2.0 # BBody

        for i in range(N - 1):
            eh = ebins[i + 1]
            x = eh / kT
            if pm.math.leq(x, 1.0e-4):
                nh = eh * kT
            elif pm.math.gt(x, 60.0):
                flux[i:] = 0.0
                break
            else:
                nh = eh * eh / (pm.math.exp(x) - 1.0)
            flux[i] = norm * (nl + nh) * (eh - el)
            el = eh
            nl = nh

        return flux

if __name__ == '__main__':
    import pymc as pm
    ph_ebins = np.geomspace(1, 10, 101)
    t = 10
    src_true = (t*np.exp(2.0)*Powerlaw(1.5))(ph_ebins)
    np.random.seed(42)
    data = np.random.poisson(src_true.eval())
    with pm.Model() as model:
        # PhoIndex = pm.Flat('PhoIndex')
        # norm = pt.exp(pm.Flat('norm'))
        PhoIndex = pm.Uniform('PhoIndex', lower=0.0, upper=5)
        norm = pm.Uniform('norm', lower=1e-5, upper=30)
        pl = norm * Powerlaw(PhoIndex)
        src = pl(ph_ebins)
        loglike = pm.Poisson('N', mu=(src * t), observed=data)
        # hess = pm.hessian(model.observedlogp, model.continuous_value_vars)
        # pi_J = pt.log(pm.math.det(hess)) / 2.0
        # pm.Potential('pi_J', pi_J)
        # pmap = pm.find_MAP()
        idata = pm.sample(50000, target_accept=0.95, random_seed=42,
                          chains=4, progressbar=True
                          )
    # import arviz as az
    # az.plot_trace(idata)