import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.gradient import disconnected_grad, grad_not_implemented

try:
    import xspec
    from mxspec._pymXspec import callModFunc
    HAS_XSPEC = True
except:
    HAS_XSPEC = False


class SpectralModel:
    # name = ''
    type = ''
    def __init__(self, *pars, type='', method='trapz'):
        # if name != '':
        #     self.name = name
        # elif self.name == '':
        #     raise ValueError('model name is not specified')

        if type != '':
            if type not in ['add', 'mul', 'con']:
                raise ValueError('type must be add, mul, or con')
            self.type = type
        elif self.type == '':
            raise TypeError('model type is not specified')

        if method not in ['trapz', 'simpson']:
            raise ValueError('`method` should be "trapz" or "simpson"')
        self.method = method

        self.pars = pars

    def __call__(self, ebins):
        if self.type == 'add':
            return self._eval_flux(ebins)
        elif self.type == 'mul':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def NE(self, energy):
        raise NotImplementedError('NE is not defined')

    def ENE(self, energy):
        if self.type != 'add':
            raise NotImplementedError('ENE is not defined')

        return energy*self.NE(energy)

    def E2NE(self, energy):
        if self.type != 'add':
            raise NotImplementedError('E2NE is not defined')

        return energy*energy*self.NE(energy)

    def eval_ebins(self, ebins):
        if self.type == 'add':
            return self._eval_flux(ebins)
        elif self.type == 'mul':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def eval_energy(self, energy):
        if self.type == 'add':
            return self.NE(energy)
        elif self.type == 'mul':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _eval_flux(self, ebins):
        if self.method == 'trapz':
            dE = ebins[1:] - ebins[:-1]
            NE = self.NE(ebins)
            flux = (NE[:-1] + NE[1:])/2.0 * dE
        else:  # simpson's 1/3 rule
            dE = ebins[1:] - ebins[:-1]
            E_mid = (ebins[:-1] + ebins[1:])/2.0
            NE = self.NE(ebins)
            NE_mid = self.NE(E_mid)
            flux = dE/6.0 * (NE[:-1] + 4.0*NE_mid + NE[1:])

        return flux

    def __add__(self, other):
        if self.type != 'add':
            raise TypeError('model is not additive')

        if isinstance(other, SpectralModel) and other.type != 'add':
            raise TypeError('model is not additive')

        if isinstance(other, SpectralModel):
            _eval_flux = lambda ebins: self(ebins) + other(ebins)
            NE = lambda energy: self.NE(energy) + other.NE(energy)
            # name = f'({self.name}+{other.name})'
        else:
            _eval_flux = lambda ebins: self(ebins) + other
            NE = lambda energy: self.NE(energy) + other
            # name = f'({self.name}+{other})'

        model = SpectralModel(type='add')
        model.NE = NE
        model._eval_flux = _eval_flux

        return model

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, SpectralModel):
            _eval_flux = lambda ebins: self(ebins) * other(ebins)
            NE = lambda energy: self.NE(energy) * other.eval_energy(energy)
            # name = f'{self.name}*{other.name}'
        else:
            _eval_flux = lambda ebins: self(ebins) * other
            NE = lambda energy: self.NE(energy) * other
            # name = f'{self.name}*{other}'

        model = SpectralModel(type='add')
        model.NE = NE
        model._eval_flux = _eval_flux

        return model

    def __rmul__(self, other):
        return self.__mul__(other)


    def __matmul__(self, other):
        raise NotImplementedError


class Powerlaw(SpectralModel):
    # name = 'PL'
    type = 'add'
    def NE(self, energy):
        PhoIndex, = self.pars
        E = pt.constant(energy, dtype=pytensor.config.floatX)
        return pt.pow(E, -PhoIndex)

    def _eval_flux(self, ebins):
        PhoIndex, = self.pars
        alpha = 1.0 - PhoIndex
        ebins = pt.constant(ebins, dtype=pytensor.config.floatX)
        integral = pt.pow(ebins, alpha) / alpha
        return integral[1:] - integral[:-1]

class Cutoffpl(SpectralModel):
    # name = 'CPL'
    type = 'add'
    def NE(self, energy):
        PhoIndex, HighECut = self.pars
        E = pt.constant(energy, dtype=pytensor.config.floatX)
        return pt.pow(E, -PhoIndex) * pt.exp(-E / HighECut)


class Bbody(SpectralModel):
    # name = 'BB'
    type = 'add'
    def NE(self, energy):
        kT, = self.pars
        E = pt.constant(energy, dtype=pytensor.config.floatX)
        return 8.0525 * E * E / (kT * kT * kT * kT * (pt.exp(E / kT) - 1.0))

class Bbodyrad(SpectralModel):
    # name = 'BBr'
    type = 'add'
    def NE(self, energy):
        kT, = self.pars
        E = pt.constant(energy, dtype=pytensor.config.floatX)
        return 1.0344e-3 * E * E / (pt.exp(E / kT) - 1.0)

class Band(SpectralModel):
    type = 'add'
    def NE(self, energy):
        alpha, beta, Ec = self.pars
        E = pt.constant(energy, dtype=pytensor.config.floatX)
        Epiv = 100.0
        amb = alpha - beta
        Eb = Ec*amb
        return pt.switch(
            pt.lt(E, Eb),
            (E/Epiv)**alpha * pt.exp(-E/Ec),
            (Eb/Epiv)**amb * pt.exp(-amb) * (E/Epiv)**beta
        )

    # def _eval_flux(self, ebins):
    #     alpha, beta, Ec = self.pars
    #     ebins = pt.constant(ebins, dtype=pytensor.config.floatX)
    #     if self.method == 'trapz':
    #         ...
    #     else:
    #         ...


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mxspec._pymXspec import callModFunc
    ebins = np.geomspace(1, 10, 1025)
    flux1 = []
    # callModFunc('cutoffpl', ebins.tolist(), [-1.5, 7.0], flux1, [], 1, '')
    # callModFunc('powerlaw', ebins.tolist(), [1.5], flux1, [], 1, '')
    callModFunc('grbm', ebins.tolist(), [-1, -2, 300], flux1, [], 1, '')
    # cpl = Cutoffpl(-1.5, 7)
    # flux2 = cpl(ebins).eval()
    m = Band(-1, -2, 300)
    flux2 = m(ebins).eval()
    plt.semilogx(ebins[:-1], (flux2-flux1)/flux1)

    from pytensor import function
    a = pt.dscalar()
    pl = function([a], Powerlaw(a)(ebins))
    b = pt.dscalar()
    Ec = pt.dscalar()
    m = function([a, b, Ec], Band(a, b, Ec)(ebins))

