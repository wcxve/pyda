import pytensor.tensor as pt

from pyda.bayespec.base_model import AutoGradOp, SpectralModel

__all__ = [
    'Band', 'BandEp',
    'BlackBody', 'BlackBodyRad',
    'Comptonized', 'CutoffPowerlaw',
    'OOTB',
    'Powerlaw'
]


class BandEpOp(AutoGradOp):
    def __init__(self, pars, integral_method='trapz'):
        super().__init__(pars, 'add', integral_method)

    def _NE(self, E):
        alpha, beta, Epeak = self._pars
        Epiv = 100.0
        Ec = Epeak / (2.0 + alpha)
        Ebreak = (alpha - beta) * Ec
        amb = alpha - beta
        return pt.exp(pt.switch(
            pt.lt(E, Ebreak),
            alpha * pt.log(E / Epiv) - E / Ec,
            amb * pt.log(amb * Ec / Epiv) - amb + beta * pt.log(E / Epiv)
        ))


class BandOp(AutoGradOp):
    def __init__(self, pars, integral_method='trapz'):
        super().__init__(pars, 'add', integral_method)

    def _NE(self, E):
        alpha, beta, Ec = self._pars
        Epiv = 100.0
        amb = alpha - beta
        Ebreak = Ec*amb
        return pt.exp(pt.switch(
            pt.lt(E, Ebreak),
            alpha * pt.log(E / Epiv) - E / Ec,
            amb * pt.log(amb * Ec / Epiv) - amb + beta * pt.log(E / Epiv)
        ))


class BlackBodyOp(AutoGradOp):
    def __init__(self, pars, integral_method='trapz'):
        super().__init__(pars, 'add', integral_method)

    def _NE(self, E):
        kT, = self._pars
        return 8.0525 * E * E / (kT * kT * kT * kT * pt.expm1(E / kT))


class BlackBodyRadOp(AutoGradOp):
    def __init__(self, pars, integral_method='trapz'):
        super().__init__(pars, 'add', integral_method)

    def _NE(self, E):
        kT, = self._pars
        return 1.0344e-3 * E * E / pt.expm1(E / kT)


class ComptonizedOp(AutoGradOp):
    def __init__(self, pars, integral_method='trapz'):
        super().__init__(pars, 'add', integral_method)

    def _NE(self, E):
        PhoIndex, Epeak = self._pars
        return pt.exp(-E * (2.0 + PhoIndex) / Epeak + PhoIndex * pt.log(E))


class CutoffPowerlawOp(AutoGradOp):
    def __init__(self, pars, integral_method='trapz'):
        super().__init__(pars, 'add', integral_method)

    def _NE(self, E):
        PhoIndex, HighECut = self._pars
        return pt.pow(E, -PhoIndex) * pt.exp(-E / HighECut)


class OOTBOp(AutoGradOp):
    def __init__(self, pars, integral_method='trapz'):
        super().__init__(pars, 'add', integral_method)

    def _NE(self, E):
        kT, = self._pars
        Epiv = 1.0
        return pt.exp((Epiv - E) / kT) * Epiv / E


class PowerlawOp(AutoGradOp):
    def __init__(self, pars):
        super().__init__(pars, 'add', 'trapz')

    def _NE(self, E):
        PhoIndex, = self._pars
        return pt.pow(E, -PhoIndex)

    def _eval_flux(self, ebins):
        PhoIndex, = self._pars
        alpha = 1.0 - PhoIndex
        # integral = ifelse(
        #     pt.eq(alpha, 0.0),
        #     pt.log(ebins),
        #     pt.pow(ebins, alpha) / alpha,
        # )
        integral = pt.pow(ebins, alpha) / alpha

        return integral[1:] - integral[:-1]


class Band(SpectralModel):
    name = 'Band'
    def __init__(self, alpha, beta, Epeak, integral_method='trapz'):
        op = BandOp([alpha, beta, Epeak], integral_method)
        super().__init__(op, op.optype)


class BandEp(SpectralModel):
    name = 'Band_Ep'
    def __init__(self, alpha, beta, Ecut, integral_method='trapz'):
        op = BandEpOp([alpha, beta, Ecut], integral_method)
        super().__init__(op, op.optype)


class BlackBody(SpectralModel):
    name = 'BB'
    def __init__(self, kT, integral_method='trapz'):
        op = BlackBodyOp([kT], integral_method)
        super().__init__(op, op.optype)


class BlackBodyRad(SpectralModel):
    name = 'BBrad'
    def __init__(self, kT, integral_method='trapz'):
        op = BlackBodyRadOp([kT], integral_method)
        super().__init__(op, op.optype)


class Comptonized(SpectralModel):
    name = 'Comp'
    def __init__(self, PhoIndex, Epeak, integral_method='trapz'):
        op = ComptonizedOp([PhoIndex, Epeak], integral_method)
        super().__init__(op, op.optype)


class CutoffPowerlaw(SpectralModel):
    name = 'CPL'
    def __init__(self, PhoIndex, Ecut, integral_method='trapz'):
        op = CutoffPowerlawOp([PhoIndex, Ecut], integral_method)
        super().__init__(op, op.optype)


class OOTB(SpectralModel):
    name = 'OOTB'
    def __init__(self, kT, integral_method='trapz'):
        op = OOTBOp([kT], integral_method)
        super().__init__(op, op.optype)


class Powerlaw(SpectralModel):
    name = 'PL'
    def __init__(self, PhoIndex):
        op = PowerlawOp([PhoIndex])
        super().__init__(op, op.optype)
