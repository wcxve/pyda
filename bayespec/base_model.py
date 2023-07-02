import warnings
import pytensor
import pytensor.tensor as pt

from pytensor.gradient import grad_not_implemented


__all__ = ['SpectralModel', 'NumericGradOp']

pytensor.config.floatX = 'float64'


class NumericGradOp(pt.Op):
    def __init__(self, pars, optype, grad_method='c', eps=1e-7):
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
        if type(ebins) not in [pt.TensorVariable,
                               pt.sharedvar.TensorSharedVariable]:
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
            self._tensor_output = self._create_tensor(*inputs)

        return [
            self._grad_for_inputs(inputs, index, output_grads[0])
            for index in range(len(inputs))
        ]

    def _perform(self, *args):
        raise NotImplementedError

    def _create_tensor(self, *args):
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
            flux_eps = self._create_tensor(*pars, *others)
            g = (flux_eps - self._tensor_output) / self._eps
            return pt.dot(output_grad, g)
        elif self.grad_method == 'c':
            # central difference approximation, accurate when compute hessian
            par_i = pars[index]
            pars[index] = par_i + self._eps
            flux_peps = self._create_tensor(*pars, *others)
            pars[index] = par_i - self._eps
            flux_meps = self._create_tensor(*pars, *others)
            g = (flux_peps - flux_meps) / (2.0 * self._eps)
            return pt.dot(output_grad, g)
        elif self.grad_method == 'b':
            # backward difference approximation
            pars[index] = pars[index] - self._eps
            flux_eps = self._create_tensor(*pars, *others)
            g = (self._tensor_output - flux_eps) / self._eps
            return pt.dot(output_grad, g)
        else:
            return grad_not_implemented(self, index, inputs[index])


class AutoGradOp:
    def __init__(self, pars, optype, integral_method='trapz'):
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
        self._optype = optype
        self.integral_method = integral_method

    def __call__(self, ebins, flux=None):
        if type(ebins) not in [pt.TensorVariable,
                               pt.sharedvar.TensorSharedVariable]:
            ebins = pt.constant(ebins, dtype='floatX')

        if self.optype == 'add':
            return self._eval_flux(ebins)
        elif self.optype == 'mul':
            return self._eval(ebins)
        else:
            if flux is None:
                raise ValueError('`flux` is required for convolution model')
            if type(flux) != pt.TensorVariable:
                flux = pt.constant(flux, dtype='floatX')

            return self._eval(ebins, flux)

    @property
    def integral_method(self):
        return self._method

    @integral_method.setter
    def integral_method(self, value):
        if value not in ['trapz', 'simpson']:
            raise ValueError(
                f'wrong value ({value}) for `integral_method`, supported '
                'are "trapz" and "simpson"'
            )
        else:
            self._method = value

    @property
    def optype(self):
        return self._optype

    def _eval_flux(self, ebins):
        if self.optype == 'add':
            if self.integral_method == 'trapz':
                dE = ebins[1:] - ebins[:-1]
                NE = self._NE(ebins)
                flux = (NE[:-1] + NE[1:])/2.0 * dE
            else:  # simpson's 1/3 rule
                dE = ebins[1:] - ebins[:-1]
                E_mid = (ebins[:-1] + ebins[1:])/2.0
                NE = self._NE(ebins)
                NE_mid = self._NE(E_mid)
                flux = dE/6.0 * (NE[:-1] + 4.0*NE_mid + NE[1:])

        return flux

    def _NE(self, ebins):
        raise NotImplementedError('NE is not defined')

    def _eval(self, ebins, flux=None):
        raise NotImplementedError('eval is not defined')


class SpectralModel:
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
            if self.mtype != 'con':
                ebins = pt.vector('ph_ebins')
                func_tensor = self._create_tensor(ebins)
                func_args = [ebins]
            else:
                ebins = pt.vector('ph_ebins')
                flux = pt.vector('flux')
                func_tensor = self._create_tensor(ebins, flux)
                func_args = [ebins, flux]

            func_input = []
            stack = [func_tensor]
            while stack:
                var = stack.pop(0)
                if var.owner is not None:
                    for input in var.owner.inputs:
                        stack.append(input)
                # TODO: other cases?
                if var.name and var.name not in ['ph_ebins', 'flux'] \
                        and type(var) == pt.TensorVariable \
                        and var not in func_input:
                    func_input.append(var)

            func_input.extend(func_args)

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


# from pyda.numerics import specfun as _specfuc
# class Cutoffpl(SpectralModel):
#     def __init__(self, PhoIndex, HighECut, grad_method='c', eps=1e-7):
#         op = NumericGradOp([PhoIndex, HighECut], 'add', grad_method, eps)
#         op.itypes = [pt.TensorType('floatX', shape=())] * 2
#         op.itypes.append(pt.TensorType('floatX', shape=(None,)))
#         op._perform = _specfuc.cutoffpl
#         super().__init__(op, op.optype)


class GradientWarning(Warning):
    """
    issued by no implementation of gradient
    """
    pass
