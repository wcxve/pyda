import warnings
import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.gradient import grad_not_implemented

try:
    import xspec_models_cxc as xsmodel
    HAS_XSMODEL = True
except:
    HAS_XSMODEL = False

__all__ = ['SpectralModel', 'SpectralComponent']

if HAS_XSMODEL:
    __all__.extend(xsmodel.list_models())

pytensor.config.floatX = 'float64'

class SpectralComponent(pt.Op):
    def __init__(self, pars, optype, grad_method='f', eps=1e-7):
        if optype not in ['add', 'mul', 'con']:
            raise ValueError(
                f'wrong value ({optype}) for `optype`, supported are "add", '
                '"mul", and "con"'
            )

        self._pars = [
            p if type(p) == pt.TensorVariable
                else pt.constant(p, dtype='floatX')
            for p in pars
        ]
        self._npars = len(pars)
        self._optype = optype
        self.grad_method = grad_method
        self.eps = eps
        self.otypes = [pt.TensorType('floatX', shape=(None,))]

    def __call__(self, ebins, flux=None):
        if type(ebins) != pt.TensorVariable:
            ebins = pt.constant(ebins, dtype='floatX')

        if self.optype != 'con':
            return super().__call__(*self._pars, ebins)
        else:
            if flux is None:
                raise ValueError('`flux` is required for convolution model')
            if type(flux) != pt.TensorVariable:
                flux = pt.constant(flux, dtype='floatX')

            return super().__call__(*self._pars, ebins, flux)

    @property
    def eps(self):
        return self._eps.value

    @eps.setter
    def eps(self, value):
        self._eps = pt.constant(value, dtype='floatX')

    @property
    def grad_method(self):
        return self._grad_method

    @grad_method.setter
    def grad_method(self, value):
        if value not in ['b', 'c', 'f', 'n']:
            raise ValueError(
                f'wrong value ({value}) for `grad_method`, supported '
                'difference approximation types are "c" for central, '
                '"f" for forward, "b" for backward, and "n" for no gradient'
            )
        else:
            self._grad_method = value

    @property
    def optype(self):
        return self._optype

    @property
    def npars(self):
        return self._npars

    def perform(self, node, inputs, output_storage, params=None):
        # the last element of inputs is ebins
        # returns model value
        output_storage[0][0] = self._perform(*inputs)

    def grad(self, inputs, output_grads):
        # the last element of inputs is ebins
        # returns grad Op in backward mode
        if self.grad_method not in ['c', 'n'] \
                or (self.grad_method != 'n' and self.optype == 'con'):
            self._tensor_output = self._make_tensor(*inputs)

        return [
            self._grad_for_inputs(inputs, index, output_grads[0])
            for index in range(len(inputs))
        ]

    def _perform(self, *args):
        raise NotImplementedError

    def _make_tensor(self, *args):
        # in this case, pars is included in args
        return super().__call__(*args)

    def _grad_for_inputs(self, inputs, index, output_grad):
        # https://en.wikipedia.org/wiki/Finite_difference
        # https://www.dam.brown.edu/people/alcyew/handouts/numdiff.pdf
        if index == self.npars:  # case for input is ebins
            return grad_not_implemented(self, index, inputs[index])
        elif index == self.npars + 1:  # case for input is flux
            if self.grad_method != 'n':
                # TODO: numeric gradient for convolution is the hardest part
                warnings.warn(
                    'gradient for convolution component is not implemented',
                    GradientWarning
                )
                return grad_not_implemented(self, index, inputs[index])
            else:
                return grad_not_implemented(self, index, inputs[index])

        pars = inputs[:self.npars]
        others = inputs[self.npars:]  # ebins, and possibly flux if "con" model

        if self.grad_method == 'f':
            # forward difference approximation
            pars[index] = pars[index] + self._eps
            flux_eps = self._make_tensor(*pars, *others)
            g = (flux_eps - self._tensor_output) / self._eps
            return pt.dot(output_grad, g)
        elif self.grad_method == 'c':
            # central difference approximation, accurate when compute hessian
            par_i = pars[index]
            pars[index] = par_i + self._eps
            flux_peps = self._make_tensor(*pars, *others)
            pars[index] = par_i - self._eps
            flux_meps = self._make_tensor(*pars, *others)
            g = (flux_peps - flux_meps) / (2.0 * self._eps)
            return pt.dot(output_grad, g)
        elif self.grad_method == 'b':
            # backward difference approximation
            pars[index] = pars[index] - self._eps
            flux_eps = self._make_tensor(*pars, *others)
            g = (self._tensor_output - flux_eps) / self._eps
            return pt.dot(output_grad, g)
        else:
            return grad_not_implemented(self, index, inputs[index])

class XspecModelOp(SpectralComponent):
    def __init__(self, modname, pars, settings='', grad_method='f', eps=1e-7):
        if not HAS_XSMODEL:
            raise ImportError('Xspec models not found')

        if modname not in xsmodel.list_models():
            raise ValueError(f'Model "{modname}" not found')

        if settings:
            # for xsect and abund settings
            exec(settings)

        super().__init__(pars,
                         xsmodel.info(modname).modeltype.name.lower(),
                         grad_method,
                         eps)

        xsfunc = getattr(xsmodel, modname)
        language = xsmodel.info(modname).language

        if pytensor.config.floatX != 'float64' and language != 'F77Style4':
            self._xsfunc = lambda *args: np.float32(xsfunc(*args))
        else:
            self._xsfunc = xsfunc

        self.itypes = [
            pt.TensorType('floatX', shape=())
            for _ in pars
        ]
        self.itypes.append(
            pt.TensorType('floatX', shape=(None,))  # ebins
        )

        if self.optype == 'con':
            self.itypes.append(
                pt.TensorType('floatX', shape=(None,))  # flux
            )

    def _perform(self, *inputs):
        pars = np.array(inputs[:self.npars])
        return self._xsfunc(pars, *inputs[self.npars:])


class SpectralModel(object):
    def __init__(self, op, mtype):
        if mtype not in ['add', 'mul', 'con']:
            raise ValueError(
                f'wrong value ({mtype}) for `mtype`, supported are "add", '
                '"mul", and "con"'
            )

        self._op = op
        self._mtype = mtype
        self._func = None
        self._NE = None
        self._ENE = None
        self._EENE = None
        self._CE = None

    def __call__(self, *args, **kwargs):
        return self._create_tensor(*args, **kwargs)

    @property
    def mtype(self):
        return self._mtype

    @property
    def func(self):
        if self._func is None:
            func_input = []
            for p in self._op._pars:
                if type(p) == pt.TensorVariable:
                    func_input.append(p)
            if self.mtype != 'con':
                ebins = pt.vector('ph_ebins')
                func_tensor = self._create_tensor(ebins)
                func_input.append(ebins)
            else:
                ebins = pt.vector('flux')
                flux = pt.vector('flux')
                func_tensor = self._create_tensor(ebins, flux)
                func_input.extend([ebins, flux])

            self._func = pytensor.function(func_input, func_tensor)
            self._func_tensor = func_tensor
            self._func_input = func_input

        return self._func

    @property
    def NE(self):
        if self.mtype != 'add':
            raise TypeError(f'NE is undefined for "{self.mtype}" type model')

        if self._func is None:
            self.func

        if self._NE is None:
            *pars, ebins = self._func_input
            NE_tensor = self._func_tensor/(ebins[1:] - ebins[:-1])
            self._NE = pytensor.function(self._func_input, NE_tensor)
            self._NE_tensor = NE_tensor
            self._NE_input = self._func_input

        return self._NE

    @property
    def ENE(self):
        if self.mtype != 'add':
            raise TypeError(f'ENE is undefined for "{self.mtype}" type model')

        if self._NE is None:
            self.NE

        if self._ENE is None:
            *pars, ebins = self._func_input
            ENE_tensor = pt.sqrt(ebins[:-1]*ebins[1:])*self._NE_tensor
            self._ENE = pytensor.function(self._func_input, ENE_tensor)
            self._ENE_tensor = ENE_tensor
            self._ENE_input = self._func_input

        return self._ENE

    @property
    def EENE(self):
        if self.mtype != 'add':
            raise TypeError(f'EENE is undefined for "{self.mtype}" type model')

        if self._NE is None:
            self.NE

        if self._EENE is None:
            *pars, ebins = self._func_input
            EENE_tensor = ebins[:-1]*ebins[1:]*self._NE_tensor
            self._EENE = pytensor.function(self._func_input, EENE_tensor)
            self._EENE_tensor = EENE_tensor
            self._EENE_input = self._func_input

        return self._EENE

    @property
    def CE(self):
        if self.mtype != 'add':
            raise TypeError(f'CE is undefined for "{self.mtype}" type model')

        if self._NE is None:
            self.NE

        if self._CE is None:
            ch_emin = pt.vector('ch_emin')
            ch_emax = pt.vector('ch_emax')
            rsp = pt.matrix('resp_matrix')
            CE_input = self._func_input + [ch_emin, ch_emax, rsp]
            CE_tensor = pt.dot(self._func_tensor, rsp)/(ch_emax - ch_emin)
            self._CE = pytensor.function(CE_input, CE_tensor)
            self._CE_tensor = CE_tensor
            self._CE_input = CE_input

        return self._CE

    def _create_tensor(self, *args, **kwargs):
        return self._op(*args, **kwargs)

    def __add__(self, other):
        if self.mtype != 'add':
            raise TypeError('model is not additive')

        if isinstance(other, SpectralModel):
            if other.mtype != 'add':
                raise TypeError('model is not additive')

            op = lambda ebins: self(ebins) + other(ebins)
            op._pars = self._op._pars + other._op._pars
        else:
            op = lambda ebins: self(ebins) + other
            if type(other) != pt.TensorVariable:
                op._pars = self._op._pars
            else:
                op._pars = self._op._pars + [other]

        model = SpectralModel(op, 'add')

        return model

    def __mul__(self, other):
        if isinstance(other, SpectralModel):
            if self.mtype == 'add':
                if other.mtype == 'add':
                    raise TypeError('model is not multiplicative')
                elif other.mtype == 'mul':
                    op = lambda ebins: other(ebins) * self(ebins)
                    mtype = 'add'
                else:  # other is con
                    op = lambda ebins: other(ebins, self(ebins))
                    mtype = 'add'
            elif self.mtype == 'mul':
                if other.mtype == 'add':
                    op = lambda ebins: self(ebins) * other(ebins)
                    mtype = 'add'
                elif other.mtype == 'mul':
                    op = lambda ebins: self(ebins) * other(ebins)
                    mtype = 'mul'
                else:  # other is con
                    op = lambda ebins, flux: self(ebins) * other(ebins, flux)
                    mtype = 'con'
            else:  # self is con
                if other.mtype == 'add':
                    op = lambda ebins: self(ebins, other(ebins))
                    mtype = 'add'
                elif other.mtype == 'mul':
                    op = lambda ebins, flux: self(ebins, other(ebins) * flux)
                    mtype = 'con'
                else:  # other is con
                    op = lambda ebins, flux: self(ebins, other(ebins, flux))
                    mtype = 'con'
            op._pars = self._op._pars + other._op._pars
        else:
            if self.mtype != 'con':
                op = lambda ebins: self(ebins) * other

            else:
                op = lambda ebins, flux: self(ebins, flux) * other

            if type(other) != pt.TensorVariable:
                op._pars = self._op._pars
            else:
                op._pars = self._op._pars + [other]

            mtype = str(self.mtype)

        model = SpectralModel(op, mtype)

        return model

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        if isinstance(other, SpectralModel):
            return other*self
        else:
            return self*other


class_template = """
class {modname}(SpectralModel):
    def __init__(self, *pars, settings='', grad_method='f', eps=1e-7):
        op = XspecModelOp('{modname}', pars, settings, grad_method, eps)
        super().__init__(op, op.optype)
"""
if HAS_XSMODEL:
    for m in xsmodel.list_models():
        exec(class_template.format(modname=m))

class GradientWarning(Warning):
    """
    issued by no implementation of gradient
    """


if __name__ == '__main__':
    ebins = np.geomspace(1, 10, 1025)
    flux = lambda E, PhoIndex: E ** (1 - PhoIndex) / (1 - PhoIndex)
    pflux = flux(10, 1.2) - flux(1, 1.2)

    # pl = powerlaw(1.2)
    # tbabs = TBabs(0.356)
    # src = tbabs*pl
    # m = src(ebins)
    # flux_val = m.eval()
    rng = np.random.default_rng(42)
    # flux_data = rng.poisson(flux_val)

    # import matplotlib.pyplot as plt
    # plt.loglog(ebins[:-1], flux_val)
    flux_data = rng.poisson(10, ebins.size-1)

    import pymc as pm
    with pm.Model() as model:
        norm = pm.HalfFlat('norm')
        PhoIndex = pm.Flat('PhoIndex')
        # ebins = pt.vector('ebins')
        pl = norm*powerlaw(PhoIndex)
        # pl = powerlaw(PhoIndex)
        # tbabs = TBabs(0.356)
        # src = tbabs * pl
        m = pl(ebins)
        pm.Poisson('d', m, observed=flux_data)

        idata = pm.sample(random_seed=42,
                          init='jitter+adapt_diag_grad',
                          target_accept=0.95,
                          )
        # mle = pm.find_MAP(return_raw=True)