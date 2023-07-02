from pyda.bayespec.xsmodel import XspecNumericGradOp
from pyda.bayespec.base_model import SpectralModel


class SSS_ice(SpectralModel):
    pars_name = ('clumps',)
    pars_range = ((0.0, 10.0),)
    pars_default = (0.0,)
    pars_frozen = (False,)
    def __init__(self, clumps, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('SSS_ice', [clumps], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class TBabs(SpectralModel):
    pars_name = ('nH',)
    pars_range = ((0.0, 1000000.0),)
    pars_default = (1.0,)
    pars_frozen = (False,)
    def __init__(self, nH, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('TBabs', [nH], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class TBfeo(SpectralModel):
    pars_name = ('nH', 'O', 'Fe', 'redshift')
    pars_range = ((0.0, 1000000.0), (-1e+38, 1e+38), (-1e+38, 1e+38), (-1.0, 10.0))
    pars_default = (1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, True, True, True)
    def __init__(self, nH, O, Fe, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('TBfeo', [nH, O, Fe, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class TBgas(SpectralModel):
    pars_name = ('nH', 'redshift')
    pars_range = ((0.0, 1000000.0), (-1.0, 10.0))
    pars_default = (1.0, 0.0)
    pars_frozen = (False, True)
    def __init__(self, nH, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('TBgas', [nH, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class TBgrain(SpectralModel):
    pars_name = ('nH', 'h2', 'rho', 'amin', 'amax', 'PL')
    pars_range = ((0.0, 1000000.0), (0.0, 1.0), (0.0, 5.0), (0.0, 0.25), (0.0, 1.0), (0.0, 5.0))
    pars_default = (1.0, 0.2, 1.0, 0.025, 0.25, 3.5)
    pars_frozen = (False, True, True, True, True, True)
    def __init__(self, nH, h2, rho, amin, amax, PL, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('TBgrain', [nH, h2, rho, amin, amax, PL], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class TBpcf(SpectralModel):
    pars_name = ('nH', 'pcf', 'redshift')
    pars_range = ((0.0, 1000000.0), (0.0, 1.0), (-1.0, 10.0))
    pars_default = (1.0, 0.5, 0.0)
    pars_frozen = (False, False, True)
    def __init__(self, nH, pcf, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('TBpcf', [nH, pcf, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class TBrel(SpectralModel):
    pars_name = ('nH', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Cl', 'Ar', 'Ca', 'Cr', 'Fe', 'Co', 'Ni', 'H2', 'rho', 'amin', 'amax', 'PL', 'H_dep', 'He_dep', 'C_dep', 'N_dep', 'O_dep', 'Ne_dep', 'Na_dep', 'Mg_dep', 'Al_dep', 'Si_dep', 'S_dep', 'Cl_dep', 'Ar_dep', 'Ca_dep', 'Cr_dep', 'Fe_dep', 'Co_dep', 'Ni_dep', 'redshift')
    pars_range = ((-1000000.0, 1000000.0), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1e+38), (0.0, 1.0), (0.0, 5.0), (0.0, 0.25), (0.0, 1.0), (0.0, 5.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-1.0, 10.0))
    pars_default = (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 0.025, 0.25, 3.5, 1.0, 1.0, 0.5, 1.0, 0.6, 1.0, 0.25, 0.2, 0.02, 0.1, 0.6, 0.5, 1.0, 0.003, 0.03, 0.3, 0.05, 0.04, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, H2, rho, amin, amax, PL, H_dep, He_dep, C_dep, N_dep, O_dep, Ne_dep, Na_dep, Mg_dep, Al_dep, Si_dep, S_dep, Cl_dep, Ar_dep, Ca_dep, Cr_dep, Fe_dep, Co_dep, Ni_dep, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('TBrel', [nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, H2, rho, amin, amax, PL, H_dep, He_dep, C_dep, N_dep, O_dep, Ne_dep, Na_dep, Mg_dep, Al_dep, Si_dep, S_dep, Cl_dep, Ar_dep, Ca_dep, Cr_dep, Fe_dep, Co_dep, Ni_dep, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class TBvarabs(SpectralModel):
    pars_name = ('nH', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Cl', 'Ar', 'Ca', 'Cr', 'Fe', 'Co', 'Ni', 'H2', 'rho', 'amin', 'amax', 'PL', 'H_dep', 'He_dep', 'C_dep', 'N_dep', 'O_dep', 'Ne_dep', 'Na_dep', 'Mg_dep', 'Al_dep', 'Si_dep', 'S_dep', 'Cl_dep', 'Ar_dep', 'Ca_dep', 'Cr_dep', 'Fe_dep', 'Co_dep', 'Ni_dep', 'Redshift')
    pars_range = ((0.0, 1000000.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 5.0), (0.0, 0.25), (0.0, 1.0), (0.0, 5.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 1.0, 0.025, 0.25, 3.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, H2, rho, amin, amax, PL, H_dep, He_dep, C_dep, N_dep, O_dep, Ne_dep, Na_dep, Mg_dep, Al_dep, Si_dep, S_dep, Cl_dep, Ar_dep, Ca_dep, Cr_dep, Fe_dep, Co_dep, Ni_dep, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('TBvarabs', [nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, H2, rho, amin, amax, PL, H_dep, He_dep, C_dep, N_dep, O_dep, Ne_dep, Na_dep, Mg_dep, Al_dep, Si_dep, S_dep, Cl_dep, Ar_dep, Ca_dep, Cr_dep, Fe_dep, Co_dep, Ni_dep, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class absori(SpectralModel):
    pars_name = ('PhoIndex', 'nH', 'Temp_abs', 'xi', 'Redshift', 'Fe_abund')
    pars_range = ((0.0, 4.0), (0.0, 100.0), (10000.0, 1000000.0), (0.0, 5000.0), (-0.999, 10.0), (0.0, 1000000.0))
    pars_default = (2.0, 1.0, 30000.0, 1.0, 0.0, 1.0)
    pars_frozen = (True, False, True, False, True, True)
    def __init__(self, PhoIndex, nH, Temp_abs, xi, Redshift, Fe_abund, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('absori', [PhoIndex, nH, Temp_abs, xi, Redshift, Fe_abund], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class acisabs(SpectralModel):
    pars_name = ('Tdays', 'norm', 'tauinf', 'tefold', 'nC', 'nH', 'nO', 'nN')
    pars_range = ((0.0, 10000.0), (0.0, 1.0), (0.0, 1.0), (1.0, 10000.0), (0.0, 50.0), (1.0, 50.0), (0.0, 50.0), (0.0, 50.0))
    pars_default = (850.0, 0.00722, 0.582, 620.0, 10.0, 20.0, 2.0, 1.0)
    pars_frozen = (True, True, True, True, True, True, True, True)
    def __init__(self, Tdays, norm, tauinf, tefold, nC, nH, nO, nN, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('acisabs', [Tdays, norm, tauinf, tefold, nC, nH, nO, nN], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class agauss(SpectralModel):
    pars_name = ('LineE', 'Sigma')
    pars_range = ((0.0, 1000000.0), (0.0, 1000000.0))
    pars_default = (10.0, 1.0)
    pars_frozen = (False, False)
    def __init__(self, LineE, Sigma, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('agauss', [LineE, Sigma], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class agnsed(SpectralModel):
    pars_name = ('mass', 'dist', 'logmdot', 'astar', 'cosi', 'kTe_hot', 'kTe_warm', 'Gamma_hot', 'Gamma_warm', 'R_hot', 'R_warm', 'logrout', 'Htmax', 'reprocess', 'redshift')
    pars_range = ((1.0, 10000000000.0), (0.01, 1000000000.0), (-10.0, 2.0), (-1.0, 0.998), (0.05, 1.0), (10.0, 300.0), (0.1, 0.5), (1.3, 3.0), (2.0, 10.0), (6.0, 500.0), (6.0, 500.0), (-3.0, 7.0), (6.0, 10.0), (0.0, 1.0), (0.0, 1.0))
    pars_default = (10000000.0, 100.0, -1.0, 0.0, 0.5, 100.0, 0.2, 1.7, 2.7, 10.0, 20.0, -1.0, 10.0, 1.0, 0.0)
    pars_frozen = (True, True, False, True, True, True, False, False, False, False, False, True, True, True, True)
    def __init__(self, mass, dist, logmdot, astar, cosi, kTe_hot, kTe_warm, Gamma_hot, Gamma_warm, R_hot, R_warm, logrout, Htmax, reprocess, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('agnsed', [mass, dist, logmdot, astar, cosi, kTe_hot, kTe_warm, Gamma_hot, Gamma_warm, R_hot, R_warm, logrout, Htmax, reprocess, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class agnslim(SpectralModel):
    pars_name = ('mass', 'dist', 'logmdot', 'astar', 'cosi', 'kTe_hot', 'kTe_warm', 'Gamma_hot', 'Gamma_warm', 'R_hot', 'R_warm', 'logrout', 'rin', 'redshift')
    pars_range = ((1.0, 10000000000.0), (0.01, 1000000000.0), (-10.0, 3.0), (0.0, 0.998), (0.05, 1.0), (10.0, 300.0), (0.1, 0.5), (1.3, 3.0), (2.0, 10.0), (2.0, 500.0), (2.0, 500.0), (-3.0, 7.0), (-1.0, 100.0), (0.0, 5.0))
    pars_default = (10000000.0, 100.0, 1.0, 0.0, 0.5, 100.0, 0.2, 2.4, 3.0, 10.0, 20.0, -1.0, -1.0, 0.0)
    pars_frozen = (True, True, False, True, True, True, False, False, False, False, False, True, True, True)
    def __init__(self, mass, dist, logmdot, astar, cosi, kTe_hot, kTe_warm, Gamma_hot, Gamma_warm, R_hot, R_warm, logrout, rin, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('agnslim', [mass, dist, logmdot, astar, cosi, kTe_hot, kTe_warm, Gamma_hot, Gamma_warm, R_hot, R_warm, logrout, rin, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class apec(SpectralModel):
    pars_name = ('kT', 'Abundanc', 'Redshift')
    pars_range = ((0.008, 64.0), (0.0, 5.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 0.0)
    pars_frozen = (False, True, True)
    def __init__(self, kT, Abundanc, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('apec', [kT, Abundanc, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bapec(SpectralModel):
    pars_name = ('kT', 'Abundanc', 'Redshift', 'Velocity')
    pars_range = ((0.008, 64.0), (0.0, 5.0), (-0.999, 10.0), (0.0, 10000.0))
    pars_default = (1.0, 1.0, 0.0, 0.0)
    pars_frozen = (False, True, True, True)
    def __init__(self, kT, Abundanc, Redshift, Velocity, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bapec', [kT, Abundanc, Redshift, Velocity], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bbody(SpectralModel):
    pars_name = ('kT',)
    pars_range = ((0.0001, 200.0),)
    pars_default = (3.0,)
    pars_frozen = (False,)
    def __init__(self, kT, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bbody', [kT], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bbodyrad(SpectralModel):
    pars_name = ('kT',)
    pars_range = ((0.0001, 200.0),)
    pars_default = (3.0,)
    pars_frozen = (False,)
    def __init__(self, kT, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bbodyrad', [kT], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bexrav(SpectralModel):
    pars_name = ('Gamma1', 'breakE', 'Gamma2', 'foldE', 'rel_refl', 'cosIncl', 'abund', 'Fe_abund', 'Redshift')
    pars_range = ((-10.0, 10.0), (0.1, 1000.0), (-10.0, 10.0), (1.0, 1000000.0), (0.0, 10.0), (0.05, 0.95), (0.0, 1000000.0), (0.0, 1000000.0), (-0.999, 10.0))
    pars_default = (2.0, 10.0, 2.0, 100.0, 0.0, 0.45, 1.0, 1.0, 0.0)
    pars_frozen = (False, False, False, False, False, True, True, True, True)
    def __init__(self, Gamma1, breakE, Gamma2, foldE, rel_refl, cosIncl, abund, Fe_abund, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bexrav', [Gamma1, breakE, Gamma2, foldE, rel_refl, cosIncl, abund, Fe_abund, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bexriv(SpectralModel):
    pars_name = ('Gamma1', 'breakE', 'Gamma2', 'foldE', 'rel_refl', 'Redshift', 'abund', 'Fe_abund', 'cosIncl', 'T_disk', 'xi')
    pars_range = ((-10.0, 10.0), (0.1, 1000.0), (-10.0, 10.0), (1.0, 1000000.0), (0.0, 1000000.0), (-0.999, 10.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.05, 0.95), (10000.0, 1000000.0), (0.0, 5000.0))
    pars_default = (2.0, 10.0, 2.0, 100.0, 0.0, 0.0, 1.0, 1.0, 0.45, 30000.0, 1.0)
    pars_frozen = (False, False, False, False, False, True, True, True, True, True, False)
    def __init__(self, Gamma1, breakE, Gamma2, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bexriv', [Gamma1, breakE, Gamma2, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bkn2pow(SpectralModel):
    pars_name = ('PhoIndx1', 'BreakE1', 'PhoIndx2', 'BreakE2', 'PhoIndx3')
    pars_range = ((-3.0, 10.0), (0.0, 1000000.0), (-3.0, 10.0), (0.0, 1000000.0), (-3.0, 10.0))
    pars_default = (1.0, 5.0, 2.0, 10.0, 3.0)
    pars_frozen = (False, False, False, False, False)
    def __init__(self, PhoIndx1, BreakE1, PhoIndx2, BreakE2, PhoIndx3, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bkn2pow', [PhoIndx1, BreakE1, PhoIndx2, BreakE2, PhoIndx3], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bknpower(SpectralModel):
    pars_name = ('PhoIndx1', 'BreakE', 'PhoIndx2')
    pars_range = ((-3.0, 10.0), (0.0, 1000000.0), (-3.0, 10.0))
    pars_default = (1.0, 5.0, 2.0)
    pars_frozen = (False, False, False)
    def __init__(self, PhoIndx1, BreakE, PhoIndx2, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bknpower', [PhoIndx1, BreakE, PhoIndx2], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bmc(SpectralModel):
    pars_name = ('kT', 'alpha', 'log_A')
    pars_range = ((0.0001, 200.0), (0.0001, 6.0), (-8.0, 8.0))
    pars_default = (1.0, 1.0, 0.0)
    pars_frozen = (False, False, False)
    def __init__(self, kT, alpha, log_A, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bmc', [kT, alpha, log_A], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bremss(SpectralModel):
    pars_name = ('kT',)
    pars_range = ((0.0001, 200.0),)
    pars_default = (7.0,)
    pars_frozen = (False,)
    def __init__(self, kT, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bremss', [kT], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class brnei(SpectralModel):
    pars_name = ('kT', 'kT_init', 'Abundanc', 'Tau', 'Redshift', 'Velocity')
    pars_range = ((0.0808, 79.9), (0.0808, 79.9), (0.0, 10000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0), (0.0, 10000.0))
    pars_default = (0.5, 1.0, 1.0, 100000000000.0, 0.0, 0.0)
    pars_frozen = (False, False, True, False, True, True)
    def __init__(self, kT, kT_init, Abundanc, Tau, Redshift, Velocity, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('brnei', [kT, kT_init, Abundanc, Tau, Redshift, Velocity], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class btapec(SpectralModel):
    pars_name = ('kT', 'kTi', 'Abundanc', 'Redshift', 'Velocity')
    pars_range = ((0.008, 64.0), (0.008, 64.0), (0.0, 5.0), (-0.999, 10.0), (0.0, 10000.0))
    pars_default = (1.0, 1.0, 1.0, 0.0, 0.0)
    pars_frozen = (False, False, True, True, True)
    def __init__(self, kT, kTi, Abundanc, Redshift, Velocity, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('btapec', [kT, kTi, Abundanc, Redshift, Velocity], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bvapec(SpectralModel):
    pars_name = ('kT', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift', 'Velocity')
    pars_range = ((0.0808, 68.447), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0), (0.0, 10000.0))
    pars_default = (6.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, Velocity, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bvapec', [kT, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, Velocity], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bvrnei(SpectralModel):
    pars_name = ('kT', 'kT_init', 'H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Tau', 'Redshift', 'Velocity')
    pars_range = ((0.0808, 79.9), (0.0808, 79.9), (0.0, 1.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0), (0.0, 10000.0))
    pars_default = (0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100000000000.0, 0.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True)
    def __init__(self, kT, kT_init, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, Velocity, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bvrnei', [kT, kT_init, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, Velocity], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bvtapec(SpectralModel):
    pars_name = ('kT', 'kTi', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift', 'Velocity')
    pars_range = ((0.0808, 68.447), (0.0808, 68.447), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0), (0.0, 10000.0))
    pars_default = (6.5, 6.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, kTi, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, Velocity, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bvtapec', [kT, kTi, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, Velocity], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bvvapec(SpectralModel):
    pars_name = ('kT', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Redshift', 'Velocity')
    pars_range = ((0.0808, 68.447), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0), (0.0, 10000.0))
    pars_default = (6.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, Velocity, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bvvapec', [kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, Velocity], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bvvrnei(SpectralModel):
    pars_name = ('kT', 'kT_init', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Tau', 'Redshift', 'Velocity')
    pars_range = ((0.0808, 79.9), (0.0808, 79.9), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0), (0.0, 10000.0))
    pars_default = (0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100000000000.0, 0.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True)
    def __init__(self, kT, kT_init, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, Velocity, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bvvrnei', [kT, kT_init, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, Velocity], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bvvtapec(SpectralModel):
    pars_name = ('kT', 'kTi', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Redshift', 'Velocity')
    pars_range = ((0.0808, 68.447), (0.0808, 68.447), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0), (0.0, 10000.0))
    pars_default = (6.5, 6.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, kTi, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, Velocity, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bvvtapec', [kT, kTi, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, Velocity], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class bwcycl(SpectralModel):
    pars_name = ('Radius', 'Mass', 'csi', 'delta', 'B', 'Mdot', 'Te', 'r0', 'D', 'BBnorm', 'CYCnorm', 'FFnorm')
    pars_range = ((5.0, 20.0), (1.0, 3.0), (0.01, 20.0), (0.01, 20.0), (0.01, 100.0), (1e-06, 1000000.0), (0.1, 100.0), (10.0, 1000.0), (1.0, 20.0), (0.0, 100.0), (-1.0, 100.0), (-1.0, 100.0))
    pars_default = (10.0, 1.4, 1.5, 1.8, 4.0, 1.0, 5.0, 44.0, 5.0, 0.0, 1.0, 1.0)
    pars_frozen = (True, True, False, False, False, False, False, False, True, True, True, True)
    def __init__(self, Radius, Mass, csi, delta, B, Mdot, Te, r0, D, BBnorm, CYCnorm, FFnorm, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('bwcycl', [Radius, Mass, csi, delta, B, Mdot, Te, r0, D, BBnorm, CYCnorm, FFnorm], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class c6mekl(SpectralModel):
    pars_name = ('CPcoef1', 'CPcoef2', 'CPcoef3', 'CPcoef4', 'CPcoef5', 'CPcoef6', 'nH', 'abundanc', 'Redshift', 'switch')
    pars_range = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (1e-06, 1e+20), (0.0, 10.0), (-0.999, 10.0), (None, None))
    pars_default = (1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 1)
    pars_frozen = (False, False, False, False, False, False, True, True, True, True)
    def __init__(self, CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, abundanc, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('c6mekl', [CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, abundanc, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class c6pmekl(SpectralModel):
    pars_name = ('CPcoef1', 'CPcoef2', 'CPcoef3', 'CPcoef4', 'CPcoef5', 'CPcoef6', 'nH', 'abundanc', 'Redshift', 'switch')
    pars_range = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (1e-06, 1e+20), (0.0, 10.0), (-0.999, 10.0), (None, None))
    pars_default = (1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 1)
    pars_frozen = (False, False, False, False, False, False, True, True, True, True)
    def __init__(self, CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, abundanc, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('c6pmekl', [CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, abundanc, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class c6pvmkl(SpectralModel):
    pars_name = ('CPcoef1', 'CPcoef2', 'CPcoef3', 'CPcoef4', 'CPcoef5', 'CPcoef6', 'nH', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift', 'switch')
    pars_range = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (1e-06, 1e+20), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-0.999, 10.0), (None, None))
    pars_default = (1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1)
    pars_frozen = (False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('c6pvmkl', [CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class c6vmekl(SpectralModel):
    pars_name = ('CPcoef1', 'CPcoef2', 'CPcoef3', 'CPcoef4', 'CPcoef5', 'CPcoef6', 'nH', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift', 'switch')
    pars_range = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (1e-06, 1e+20), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-0.999, 10.0), (None, None))
    pars_default = (1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1)
    pars_frozen = (False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('c6vmekl', [CPcoef1, CPcoef2, CPcoef3, CPcoef4, CPcoef5, CPcoef6, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class cabs(SpectralModel):
    pars_name = ('nH',)
    pars_range = ((0.0, 1000000.0),)
    pars_default = (1.0,)
    pars_frozen = (False,)
    def __init__(self, nH, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('cabs', [nH], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class carbatm(SpectralModel):
    pars_name = ('T', 'NSmass', 'NSrad')
    pars_range = ((1.0, 4.0), (0.6, 2.8), (6.0, 23.0))
    pars_default = (2.0, 1.4, 10.0)
    pars_frozen = (False, False, False)
    def __init__(self, T, NSmass, NSrad, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('carbatm', [T, NSmass, NSrad], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class cemekl(SpectralModel):
    pars_name = ('alpha', 'Tmax', 'nH', 'abundanc', 'Redshift', 'switch')
    pars_range = ((0.01, 20.0), (0.01, 100.0), (1e-06, 1e+20), (0.0, 10.0), (-0.999, 10.0), (0.0, 1.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 0.0, 1)
    pars_frozen = (True, False, True, True, True, True)
    def __init__(self, alpha, Tmax, nH, abundanc, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('cemekl', [alpha, Tmax, nH, abundanc, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class cevmkl(SpectralModel):
    pars_name = ('alpha', 'Tmax', 'nH', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift', 'switch')
    pars_range = ((0.01, 20.0), (0.01, 100.0), (1e-06, 1e+20), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-0.999, 10.0), (0.0, 1.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1)
    pars_frozen = (True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, alpha, Tmax, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('cevmkl', [alpha, Tmax, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class cflow(SpectralModel):
    pars_name = ('slope', 'lowT', 'highT', 'Abundanc', 'redshift')
    pars_range = ((-5.0, 5.0), (0.0808, 79.9), (0.0808, 79.9), (0.0, 5.0), (1e-10, 10.0))
    pars_default = (0.0, 0.1, 4.0, 1.0, 0.1)
    pars_frozen = (False, False, False, False, True)
    def __init__(self, slope, lowT, highT, Abundanc, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('cflow', [slope, lowT, highT, Abundanc, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class cflux(SpectralModel):
    pars_name = ('Emin', 'Emax', 'lg10Flux')
    pars_range = ((0.0, 1000000.0), (0.0, 1000000.0), (-100.0, 100.0))
    pars_default = (0.5, 10.0, -12.0)
    pars_frozen = (True, True, False)
    def __init__(self, Emin, Emax, lg10Flux, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('cflux', [Emin, Emax, lg10Flux], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class cglumin(SpectralModel):
    pars_name = ('Emin', 'Emax', 'Distance', 'lg10Lum')
    pars_range = ((0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (-100.0, 100.0))
    pars_default = (0.5, 10.0, 10.0, 40.0)
    pars_frozen = (True, True, True, False)
    def __init__(self, Emin, Emax, Distance, lg10Lum, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('cglumin', [Emin, Emax, Distance, lg10Lum], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class clumin(SpectralModel):
    pars_name = ('Emin', 'Emax', 'Redshift', 'lg10Lum')
    pars_range = ((0.0, 1000000.0), (0.0, 1000000.0), (-0.999, 10.0), (-100.0, 100.0))
    pars_default = (0.5, 10.0, 0.0, 40.0)
    pars_frozen = (True, True, True, False)
    def __init__(self, Emin, Emax, Redshift, lg10Lum, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('clumin', [Emin, Emax, Redshift, lg10Lum], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class compLS(SpectralModel):
    pars_name = ('kT', 'tau')
    pars_range = ((0.001, 20.0), (0.0001, 200.0))
    pars_default = (2.0, 10.0)
    pars_frozen = (False, False)
    def __init__(self, kT, tau, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('compLS', [kT, tau], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class compPS(SpectralModel):
    pars_name = ('kTe', 'EleIndex', 'Gmin', 'Gmax', 'kTbb', 'tau_y', 'geom', 'HovR_cyl', 'cosIncl', 'cov_frac', 'rel_refl', 'Fe_ab_re', 'Me_ab', 'xi', 'Tdisk', 'Betor10', 'Rin', 'Rout', 'Redshift')
    pars_range = ((20.0, 100000.0), (0.0, 5.0), (-1.0, 10.0), (10.0, 10000.0), (0.001, 10.0), (0.05, 3.0), (-5.0, 4.0), (0.5, 2.0), (0.05, 0.95), (0.0, 1.0), (0.0, 10000.0), (0.1, 10.0), (0.1, 10.0), (0.0, 100000.0), (10000.0, 1000000.0), (-10.0, 10.0), (6.001, 10000.0), (0.0, 1000000.0), (-0.999, 10.0))
    pars_default = (100.0, 2.0, -1.0, 1000.0, 0.1, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0, 1000000.0, -10.0, 10.0, 1000.0, 0.0)
    pars_frozen = (False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kTe, EleIndex, Gmin, Gmax, kTbb, tau_y, geom, HovR_cyl, cosIncl, cov_frac, rel_refl, Fe_ab_re, Me_ab, xi, Tdisk, Betor10, Rin, Rout, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('compPS', [kTe, EleIndex, Gmin, Gmax, kTbb, tau_y, geom, HovR_cyl, cosIncl, cov_frac, rel_refl, Fe_ab_re, Me_ab, xi, Tdisk, Betor10, Rin, Rout, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class compST(SpectralModel):
    pars_name = ('kT', 'tau')
    pars_range = ((0.001, 100.0), (0.0001, 200.0))
    pars_default = (2.0, 10.0)
    pars_frozen = (False, False)
    def __init__(self, kT, tau, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('compST', [kT, tau], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class compTT(SpectralModel):
    pars_name = ('Redshift', 'T0', 'kT', 'taup', 'approx')
    pars_range = ((-0.999, 10.0), (0.001, 100.0), (2.0, 500.0), (0.01, 200.0), (0.0, 200.0))
    pars_default = (0.0, 0.1, 50.0, 1.0, 1.0)
    pars_frozen = (True, False, False, False, True)
    def __init__(self, Redshift, T0, kT, taup, approx, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('compTT', [Redshift, T0, kT, taup, approx], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class compbb(SpectralModel):
    pars_name = ('kT', 'kTe', 'tau')
    pars_range = ((0.0001, 200.0), (1.0, 200.0), (0.0, 10.0))
    pars_default = (1.0, 50.0, 0.1)
    pars_frozen = (False, True, False)
    def __init__(self, kT, kTe, tau, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('compbb', [kT, kTe, tau], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class compmag(SpectralModel):
    pars_name = ('kTbb', 'kTe', 'tau', 'eta', 'beta0', 'r0', 'A', 'betaflag')
    pars_range = ((0.2, 10.0), (0.2, 2000.0), (0.0, 10.0), (0.01, 1.0), (0.0001, 1.0), (0.0001, 100.0), (0.0, 1.0), (0.0, 2.0))
    pars_default = (1.0, 5.0, 0.5, 0.5, 0.57, 0.25, 0.001, 1.0)
    pars_frozen = (False, False, False, False, False, False, True, True)
    def __init__(self, kTbb, kTe, tau, eta, beta0, r0, A, betaflag, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('compmag', [kTbb, kTe, tau, eta, beta0, r0, A, betaflag], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class comptb(SpectralModel):
    pars_name = ('kTs', 'gamma', 'alpha', 'delta', 'kTe', 'log_A')
    pars_range = ((0.1, 10.0), (1.0, 10.0), (0.0, 400.0), (0.0, 200.0), (0.2, 2000.0), (-8.0, 8.0))
    pars_default = (1.0, 3.0, 2.0, 20.0, 5.0, 0.0)
    pars_frozen = (False, True, False, False, False, False)
    def __init__(self, kTs, gamma, alpha, delta, kTe, log_A, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('comptb', [kTs, gamma, alpha, delta, kTe, log_A], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class compth(SpectralModel):
    pars_name = ('theta', 'showbb', 'kT_bb', 'RefOn', 'tau_p', 'radius', 'g_min', 'g_max', 'G_inj', 'pairinj', 'cosIncl', 'Refl', 'Fe_abund', 'Ab_met', 'T_disk', 'xi', 'Beta', 'Rin', 'Rout', 'redshift')
    pars_range = ((1e-06, 1000000.0), (0.0, 10000.0), (1.0, 400000.0), (-2.0, 2.0), (0.0001, 10.0), (100000.0, 1e+16), (1.2, 1000.0), (5.0, 10000.0), (0.0, 5.0), (0.0, 1.0), (0.05, 0.95), (0.0, 2.0), (0.1, 10.0), (0.1, 10.0), (10000.0, 1000000.0), (0.0, 5000.0), (-10.0, 10.0), (6.001, 10000.0), (0.0, 1000000.0), (0.0, 4.0))
    pars_default = (1.0, 1.0, 200.0, -1.0, 0.1, 10000000.0, 1.3, 1000.0, 2.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1000000.0, 0.0, -10.0, 10.0, 1000.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True)
    def __init__(self, theta, showbb, kT_bb, RefOn, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('compth', [theta, showbb, kT_bb, RefOn, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class constant(SpectralModel):
    pars_name = ('factor',)
    pars_range = ((0.0, 10000000000.0),)
    pars_default = (1.0,)
    pars_frozen = (False,)
    def __init__(self, factor, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('constant', [factor], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class cpflux(SpectralModel):
    pars_name = ('Emin', 'Emax', 'Flux')
    pars_range = ((0.0, 1000000.0), (0.0, 1000000.0), (0.0, 10000000000.0))
    pars_default = (0.5, 10.0, 1.0)
    pars_frozen = (True, True, False)
    def __init__(self, Emin, Emax, Flux, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('cpflux', [Emin, Emax, Flux], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class cph(SpectralModel):
    pars_name = ('peakT', 'Abund', 'Redshift', 'switch')
    pars_range = ((0.1, 100.0), (0.0, 1000.0), (0.0, 50.0), (None, None))
    pars_default = (2.2, 1.0, 0.0, 1)
    pars_frozen = (False, True, True, True)
    def __init__(self, peakT, Abund, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('cph', [peakT, Abund, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class cplinear(SpectralModel):
    pars_name = ('energy00', 'energy01', 'energy02', 'energy03', 'energy04', 'energy05', 'energy06', 'energy07', 'energy08', 'energy09', 'log_rate00', 'log_rate01', 'log_rate02', 'log_rate03', 'log_rate04', 'log_rate05', 'log_rate06', 'log_rate07', 'log_rate08', 'log_rate09')
    pars_range = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0))
    pars_default = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    pars_frozen = (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, energy00, energy01, energy02, energy03, energy04, energy05, energy06, energy07, energy08, energy09, log_rate00, log_rate01, log_rate02, log_rate03, log_rate04, log_rate05, log_rate06, log_rate07, log_rate08, log_rate09, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('cplinear', [energy00, energy01, energy02, energy03, energy04, energy05, energy06, energy07, energy08, energy09, log_rate00, log_rate01, log_rate02, log_rate03, log_rate04, log_rate05, log_rate06, log_rate07, log_rate08, log_rate09], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class cutoffpl(SpectralModel):
    pars_name = ('PhoIndex', 'HighECut')
    pars_range = ((-3.0, 10.0), (0.01, 500.0))
    pars_default = (1.0, 15.0)
    pars_frozen = (False, False)
    def __init__(self, PhoIndex, HighECut, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('cutoffpl', [PhoIndex, HighECut], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class cyclabs(SpectralModel):
    pars_name = ('Depth0', 'E0', 'Width0', 'Depth2', 'Width2')
    pars_range = ((0.0, 100.0), (1.0, 100.0), (1.0, 100.0), (0.0, 100.0), (1.0, 100.0))
    pars_default = (2.0, 30.0, 10.0, 0.0, 20.0)
    pars_frozen = (False, False, True, True, True)
    def __init__(self, Depth0, E0, Width0, Depth2, Width2, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('cyclabs', [Depth0, E0, Width0, Depth2, Width2], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class disk(SpectralModel):
    pars_name = ('accrate', 'CenMass', 'Rinn')
    pars_range = ((0.0001, 10.0), (0.1, 20.0), (1.0, 1.04))
    pars_default = (1.0, 1.4, 1.03)
    pars_frozen = (False, True, True)
    def __init__(self, accrate, CenMass, Rinn, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('disk', [accrate, CenMass, Rinn], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class diskbb(SpectralModel):
    pars_name = ('Tin',)
    pars_range = ((0.0, 1000.0),)
    pars_default = (1.0,)
    pars_frozen = (False,)
    def __init__(self, Tin, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('diskbb', [Tin], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class diskir(SpectralModel):
    pars_name = ('kT_disk', 'Gamma', 'kT_e', 'LcovrLd', 'fin', 'rirr', 'fout', 'logrout')
    pars_range = ((0.01, 5.0), (1.001, 10.0), (1.0, 1000.0), (0.0, 10.0), (0.0, 1.0), (1.0001, 10.0), (0.0, 0.1), (3.0, 7.0))
    pars_default = (1.0, 1.7, 100.0, 0.1, 0.1, 1.2, 0.0001, 5.0)
    pars_frozen = (False, False, False, False, True, False, False, False)
    def __init__(self, kT_disk, Gamma, kT_e, LcovrLd, fin, rirr, fout, logrout, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('diskir', [kT_disk, Gamma, kT_e, LcovrLd, fin, rirr, fout, logrout], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class diskline(SpectralModel):
    pars_name = ('LineE', 'Betor10', 'Rin_M', 'Rout_M', 'Incl')
    pars_range = ((0.0, 100.0), (-10.0, 20.0), (6.0, 10000.0), (0.0, 10000000.0), (0.0, 90.0))
    pars_default = (6.7, -2.0, 10.0, 1000.0, 30.0)
    pars_frozen = (False, True, True, True, False)
    def __init__(self, LineE, Betor10, Rin_M, Rout_M, Incl, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('diskline', [LineE, Betor10, Rin_M, Rout_M, Incl], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class diskm(SpectralModel):
    pars_name = ('accrate', 'NSmass', 'Rinn', 'alpha')
    pars_range = ((0.0001, 10.0), (0.1, 20.0), (1.0, 1.04), (0.001, 20.0))
    pars_default = (1.0, 1.4, 1.03, 1.0)
    pars_frozen = (False, True, True, True)
    def __init__(self, accrate, NSmass, Rinn, alpha, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('diskm', [accrate, NSmass, Rinn, alpha], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class disko(SpectralModel):
    pars_name = ('accrate', 'NSmass', 'Rinn', 'alpha')
    pars_range = ((0.0001, 10.0), (0.1, 20.0), (1.0, 1.04), (0.001, 20.0))
    pars_default = (1.0, 1.4, 1.03, 1.0)
    pars_frozen = (False, True, True, True)
    def __init__(self, accrate, NSmass, Rinn, alpha, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('disko', [accrate, NSmass, Rinn, alpha], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class diskpbb(SpectralModel):
    pars_name = ('Tin', 'p')
    pars_range = ((0.1, 10.0), (0.5, 1.0))
    pars_default = (1.0, 0.75)
    pars_frozen = (False, False)
    def __init__(self, Tin, p, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('diskpbb', [Tin, p], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class diskpn(SpectralModel):
    pars_name = ('T_max', 'R_in')
    pars_range = ((0.0001, 200.0), (6.0, 1000.0))
    pars_default = (1.0, 6.0)
    pars_frozen = (False, False)
    def __init__(self, T_max, R_in, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('diskpn', [T_max, R_in], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class dust(SpectralModel):
    pars_name = ('Frac', 'Halosz')
    pars_range = ((0.0, 1.0), (0.0, 100000.0))
    pars_default = (0.066, 2.0)
    pars_frozen = (True, True)
    def __init__(self, Frac, Halosz, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('dust', [Frac, Halosz], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class edge(SpectralModel):
    pars_name = ('edgeE', 'MaxTau')
    pars_range = ((0.0, 100.0), (0.0, 10.0))
    pars_default = (7.0, 1.0)
    pars_frozen = (False, False)
    def __init__(self, edgeE, MaxTau, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('edge', [edgeE, MaxTau], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class eplogpar(SpectralModel):
    pars_name = ('Ep', 'beta')
    pars_range = ((1e-10, 10000.0), (-4.0, 4.0))
    pars_default = (0.1, 0.2)
    pars_frozen = (False, False)
    def __init__(self, Ep, beta, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('eplogpar', [Ep, beta], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class eqpair(SpectralModel):
    pars_name = ('l_hovl_s', 'l_bb', 'kT_bb', 'l_ntol_h', 'tau_p', 'radius', 'g_min', 'g_max', 'G_inj', 'pairinj', 'cosIncl', 'Refl', 'Fe_abund', 'Ab_met', 'T_disk', 'xi', 'Beta', 'Rin', 'Rout', 'redshift')
    pars_range = ((1e-06, 1000000.0), (0.0, 10000.0), (1.0, 400000.0), (0.0, 0.9999), (0.0001, 10.0), (100000.0, 1e+16), (1.2, 1000.0), (5.0, 10000.0), (0.0, 5.0), (0.0, 1.0), (0.05, 0.95), (0.0, 2.0), (0.1, 10.0), (0.1, 10.0), (10000.0, 1000000.0), (0.0, 5000.0), (-10.0, 10.0), (6.001, 10000.0), (0.0, 1000000.0), (0.0, 4.0))
    pars_default = (1.0, 100.0, 200.0, 0.5, 0.1, 10000000.0, 1.3, 1000.0, 2.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1000000.0, 0.0, -10.0, 10.0, 1000.0, 0.0)
    pars_frozen = (False, False, True, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True)
    def __init__(self, l_hovl_s, l_bb, kT_bb, l_ntol_h, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('eqpair', [l_hovl_s, l_bb, kT_bb, l_ntol_h, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class eqtherm(SpectralModel):
    pars_name = ('l_hovl_s', 'l_bb', 'kT_bb', 'l_ntol_h', 'tau_p', 'radius', 'g_min', 'g_max', 'G_inj', 'pairinj', 'cosIncl', 'Refl', 'Fe_abund', 'Ab_met', 'T_disk', 'xi', 'Beta', 'Rin', 'Rout', 'redshift')
    pars_range = ((1e-06, 1000000.0), (0.0, 10000.0), (1.0, 400000.0), (0.0, 0.9999), (0.0001, 10.0), (100000.0, 1e+16), (1.2, 1000.0), (5.0, 10000.0), (0.0, 5.0), (0.0, 1.0), (0.05, 0.95), (0.0, 2.0), (0.1, 10.0), (0.1, 10.0), (10000.0, 1000000.0), (0.0, 5000.0), (-10.0, 10.0), (6.001, 10000.0), (0.0, 1000000.0), (0.0, 4.0))
    pars_default = (1.0, 100.0, 200.0, 0.5, 0.1, 10000000.0, 1.3, 1000.0, 2.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1000000.0, 0.0, -10.0, 10.0, 1000.0, 0.0)
    pars_frozen = (False, False, True, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True)
    def __init__(self, l_hovl_s, l_bb, kT_bb, l_ntol_h, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('eqtherm', [l_hovl_s, l_bb, kT_bb, l_ntol_h, tau_p, radius, g_min, g_max, G_inj, pairinj, cosIncl, Refl, Fe_abund, Ab_met, T_disk, xi, Beta, Rin, Rout, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class equil(SpectralModel):
    pars_name = ('kT', 'Abundanc', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 10000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 0.0)
    pars_frozen = (False, True, True)
    def __init__(self, kT, Abundanc, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('equil', [kT, Abundanc, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class expabs(SpectralModel):
    pars_name = ('LowECut',)
    pars_range = ((0.0, 200.0),)
    pars_default = (2.0,)
    pars_frozen = (False,)
    def __init__(self, LowECut, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('expabs', [LowECut], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class expdec(SpectralModel):
    pars_name = ('factor',)
    pars_range = ((0.0, 100.0),)
    pars_default = (1.0,)
    pars_frozen = (False,)
    def __init__(self, factor, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('expdec', [factor], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class expfac(SpectralModel):
    pars_name = ('Ampl', 'Factor', 'StartE')
    pars_range = ((0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0))
    pars_default = (1.0, 1.0, 0.5)
    pars_frozen = (False, False, True)
    def __init__(self, Ampl, Factor, StartE, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('expfac', [Ampl, Factor, StartE], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class ezdiskbb(SpectralModel):
    pars_name = ('T_max',)
    pars_range = ((0.01, 100.0),)
    pars_default = (1.0,)
    pars_frozen = (False,)
    def __init__(self, T_max, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('ezdiskbb', [T_max], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class gabs(SpectralModel):
    pars_name = ('LineE', 'Sigma', 'Strength')
    pars_range = ((0.0, 1000000.0), (0.0, 20.0), (0.0, 1000000.0))
    pars_default = (1.0, 0.01, 1.0)
    pars_frozen = (False, False, False)
    def __init__(self, LineE, Sigma, Strength, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('gabs', [LineE, Sigma, Strength], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class gadem(SpectralModel):
    pars_name = ('Tmean', 'Tsigma', 'nH', 'abundanc', 'Redshift', 'switch')
    pars_range = ((0.01, 20.0), (0.01, 100.0), (1e-06, 1e+20), (0.0, 10.0), (-0.999, 10.0), (None, None))
    pars_default = (4.0, 0.1, 1.0, 1.0, 0.0, 2)
    pars_frozen = (True, False, True, True, True, True)
    def __init__(self, Tmean, Tsigma, nH, abundanc, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('gadem', [Tmean, Tsigma, nH, abundanc, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class gaussian(SpectralModel):
    pars_name = ('LineE', 'Sigma')
    pars_range = ((0.0, 1000000.0), (0.0, 20.0))
    pars_default = (6.5, 0.1)
    pars_frozen = (False, False)
    def __init__(self, LineE, Sigma, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('gaussian', [LineE, Sigma], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class gnei(SpectralModel):
    pars_name = ('kT', 'Abundanc', 'Tau', 'meankT', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 10000.0), (100000000.0, 50000000000000.0), (0.0808, 79.9), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 100000000000.0, 1.0, 0.0)
    pars_frozen = (False, True, False, False, True)
    def __init__(self, kT, Abundanc, Tau, meankT, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('gnei', [kT, Abundanc, Tau, meankT, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class grad(SpectralModel):
    pars_name = ('D', 'i', 'Mass', 'Mdot', 'TclovTef', 'refflag')
    pars_range = ((0.0, 10000.0), (0.0, 90.0), (0.0, 100.0), (0.0, 100.0), (1.0, 10.0), (-1.0, 1.0))
    pars_default = (10.0, 0.0, 1.0, 1.0, 1.7, 1.0)
    pars_frozen = (True, True, False, False, True, True)
    def __init__(self, D, i, Mass, Mdot, TclovTef, refflag, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('grad', [D, i, Mass, Mdot, TclovTef, refflag], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class grbcomp(SpectralModel):
    pars_name = ('kTs', 'gamma', 'kTe', 'tau', 'beta', 'fbflag', 'log_A', 'z', 'a_boost')
    pars_range = ((0.0, 20.0), (0.0, 10.0), (0.2, 2000.0), (0.0, 200.0), (0.0, 1.0), (0.0, 1.0), (-8.0, 8.0), (0.0, 10.0), (0.0, 30.0))
    pars_default = (1.0, 3.0, 100.0, 5.0, 0.2, 0.0, 5.0, 0.0, 5.0)
    pars_frozen = (False, False, False, False, False, True, True, True, True)
    def __init__(self, kTs, gamma, kTe, tau, beta, fbflag, log_A, z, a_boost, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('grbcomp', [kTs, gamma, kTe, tau, beta, fbflag, log_A, z, a_boost], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class grbjet(SpectralModel):
    pars_name = ('thobs', 'thjet', 'gamma', 'r12', 'p1', 'p2', 'E0', 'delta', 'index_pl', 'ecut', 'ktbb', 'model', 'redshift')
    pars_range = ((0.0, 30.0), (2.0, 20.0), (1.0, 500.0), (0.1, 100.0), (-2.0, 1.0), (1.1, 10.0), (0.1, 1000.0), (0.01, 1.5), (0.0, 1.5), (0.1, 1000.0), (0.1, 1000.0), (None, None), (0.001, 10.0))
    pars_default = (5.0, 10.0, 200.0, 1.0, 0.0, 1.5, 1.0, 0.2, 0.8, 20.0, 1.0, 1, 2.0)
    pars_frozen = (True, True, False, True, False, False, False, True, True, True, True, True, True)
    def __init__(self, thobs, thjet, gamma, r12, p1, p2, E0, delta, index_pl, ecut, ktbb, model, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('grbjet', [thobs, thjet, gamma, r12, p1, p2, E0, delta, index_pl, ecut, ktbb, model, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class grbm(SpectralModel):
    pars_name = ('alpha', 'beta', 'tem')
    pars_range = ((-10.0, 5.0), (-10.0, 10.0), (10.0, 10000.0))
    pars_default = (-1.0, -2.0, 300.0)
    pars_frozen = (False, False, False)
    def __init__(self, alpha, beta, tem, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('grbm', [alpha, beta, tem], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class gsmooth(SpectralModel):
    pars_name = ('Sig_6keV', 'Index')
    pars_range = ((0.0, 20.0), (-1.0, 1.0))
    pars_default = (1.0, 0.0)
    pars_frozen = (False, True)
    def __init__(self, Sig_6keV, Index, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('gsmooth', [Sig_6keV, Index], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class hatm(SpectralModel):
    pars_name = ('T', 'NSmass', 'NSrad')
    pars_range = ((0.5, 10.0), (0.6, 2.8), (5.0, 23.0))
    pars_default = (3.0, 1.4, 10.0)
    pars_frozen = (False, False, False)
    def __init__(self, T, NSmass, NSrad, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('hatm', [T, NSmass, NSrad], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class heilin(SpectralModel):
    pars_name = ('nHeI', 'b', 'z')
    pars_range = ((0.0, 1000000.0), (1.0, 1000000.0), (-0.001, 100000.0))
    pars_default = (1e-05, 10.0, 0.0)
    pars_frozen = (False, False, False)
    def __init__(self, nHeI, b, z, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('heilin', [nHeI, b, z], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class highecut(SpectralModel):
    pars_name = ('cutoffE', 'foldE')
    pars_range = ((0.0001, 1000000.0), (0.0001, 1000000.0))
    pars_default = (10.0, 15.0)
    pars_frozen = (False, False)
    def __init__(self, cutoffE, foldE, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('highecut', [cutoffE, foldE], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class hrefl(SpectralModel):
    pars_name = ('thetamin', 'thetamax', 'thetaobs', 'Feabun', 'FeKedge', 'Escfrac', 'covfac', 'Redshift')
    pars_range = ((0.0, 90.0), (0.0, 90.0), (0.0, 90.0), (0.0, 200.0), (7.0, 10.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0))
    pars_default = (0.0, 90.0, 60.0, 1.0, 7.11, 1.0, 1.0, 0.0)
    pars_frozen = (True, True, False, True, True, False, False, True)
    def __init__(self, thetamin, thetamax, thetaobs, Feabun, FeKedge, Escfrac, covfac, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('hrefl', [thetamin, thetamax, thetaobs, Feabun, FeKedge, Escfrac, covfac, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class ireflect(SpectralModel):
    pars_name = ('rel_refl', 'Redshift', 'abund', 'Fe_abund', 'cosIncl', 'T_disk', 'xi')
    pars_range = ((-1.0, 1000000.0), (-0.999, 10.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.05, 0.95), (10000.0, 1000000.0), (0.0, 5000.0))
    pars_default = (0.0, 0.0, 1.0, 1.0, 0.45, 30000.0, 1.0)
    pars_frozen = (False, True, True, True, True, True, False)
    def __init__(self, rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('ireflect', [rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class ismabs(SpectralModel):
    pars_name = ('H', 'He_II', 'C_I', 'C_II', 'C_III', 'N_I', 'N_II', 'N_III', 'O_I', 'O_II', 'O_III', 'Ne_I', 'Ne_II', 'Ne_III', 'Mg_I', 'Mg_II', 'Mg_III', 'Si_I', 'Si_II', 'Si_III', 'S_I', 'S_II', 'S_III', 'Ar_I', 'Ar_II', 'Ar_III', 'Ca_I', 'Ca_II', 'Ca_III', 'Fe', 'redshift')
    pars_range = ((0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (-1.0, 10.0))
    pars_default = (0.1, 0.0, 33.1, 0.0, 0.0, 8.32, 0.0, 0.0, 67.6, 0.0, 0.0, 12.0, 0.0, 0.0, 3.8, 0.0, 0.0, 3.35, 0.0, 0.0, 2.14, 0.0, 0.0, 0.25, 0.0, 0.0, 0.22, 0.0, 0.0, 3.16, 0.0)
    pars_frozen = (False, True, False, True, True, False, True, True, False, True, True, False, True, True, False, True, True, False, True, True, False, True, True, False, True, True, False, True, True, False, True)
    def __init__(self, H, He_II, C_I, C_II, C_III, N_I, N_II, N_III, O_I, O_II, O_III, Ne_I, Ne_II, Ne_III, Mg_I, Mg_II, Mg_III, Si_I, Si_II, Si_III, S_I, S_II, S_III, Ar_I, Ar_II, Ar_III, Ca_I, Ca_II, Ca_III, Fe, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('ismabs', [H, He_II, C_I, C_II, C_III, N_I, N_II, N_III, O_I, O_II, O_III, Ne_I, Ne_II, Ne_III, Mg_I, Mg_II, Mg_III, Si_I, Si_II, Si_III, S_I, S_II, S_III, Ar_I, Ar_II, Ar_III, Ca_I, Ca_II, Ca_III, Fe, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class ismdust(SpectralModel):
    pars_name = ('msil', 'mgra', 'redshift')
    pars_range = ((0.0, 100000.0), (0.0, 100000.0), (-1.0, 10.0))
    pars_default = (1.0, 1.0, 0.0)
    pars_frozen = (False, False, True)
    def __init__(self, msil, mgra, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('ismdust', [msil, mgra, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class jet(SpectralModel):
    pars_name = ('mass', 'Dco', 'log_mdot', 'thetaobs', 'BulkG', 'phi', 'zdiss', 'B', 'logPrel', 'gmin_inj', 'gbreak', 'gmax', 's1', 's2', 'z')
    pars_range = ((1.0, 10000000000.0), (1.0, 100000000.0), (-5.0, 2.0), (0.0, 90.0), (1.0, 100.0), (0.01, 100.0), (10.0, 10000.0), (0.01, 15.0), (40.0, 48.0), (1.0, 1000.0), (10.0, 10000.0), (1000.0, 1000000.0), (-1.0, 1.0), (1.0, 5.0), (0.0, 10.0))
    pars_default = (1000000000.0, 3350.6, -1.0, 3.0, 13.0, 0.1, 1275.0, 2.6, 43.3, 1.0, 300.0, 3000.0, 1.0, 2.7, 0.0)
    pars_frozen = (True, True, False, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, mass, Dco, log_mdot, thetaobs, BulkG, phi, zdiss, B, logPrel, gmin_inj, gbreak, gmax, s1, s2, z, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('jet', [mass, Dco, log_mdot, thetaobs, BulkG, phi, zdiss, B, logPrel, gmin_inj, gbreak, gmax, s1, s2, z], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class kdblur(SpectralModel):
    pars_name = ('Index', 'Rin_G', 'Rout_G', 'Incl')
    pars_range = ((-10.0, 10.0), (1.235, 400.0), (1.235, 400.0), (0.0, 90.0))
    pars_default = (3.0, 4.5, 100.0, 30.0)
    pars_frozen = (True, True, True, False)
    def __init__(self, Index, Rin_G, Rout_G, Incl, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('kdblur', [Index, Rin_G, Rout_G, Incl], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class kdblur2(SpectralModel):
    pars_name = ('Index', 'Rin_G', 'Rout_G', 'Incl', 'Rbreak', 'Index1')
    pars_range = ((-10.0, 10.0), (1.235, 400.0), (1.235, 400.0), (0.0, 90.0), (1.235, 400.0), (-10.0, 10.0))
    pars_default = (3.0, 4.5, 100.0, 30.0, 20.0, 3.0)
    pars_frozen = (True, True, True, False, True, True)
    def __init__(self, Index, Rin_G, Rout_G, Incl, Rbreak, Index1, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('kdblur2', [Index, Rin_G, Rout_G, Incl, Rbreak, Index1], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class kerrbb(SpectralModel):
    pars_name = ('eta', 'a', 'i', 'Mbh', 'Mdd', 'Dbh', 'hd', 'rflag', 'lflag')
    pars_range = ((0.0, 1.0), (-1.0, 0.9999), (0.0, 85.0), (0.0, 100.0), (0.0, 1000.0), (0.0, 10000.0), (1.0, 10.0), (None, None), (None, None))
    pars_default = (0.0, 0.0, 30.0, 1.0, 1.0, 10.0, 1.7, 1, 0)
    pars_frozen = (True, False, True, False, False, True, True, True, True)
    def __init__(self, eta, a, i, Mbh, Mdd, Dbh, hd, rflag, lflag, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('kerrbb', [eta, a, i, Mbh, Mdd, Dbh, hd, rflag, lflag], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class kerrconv(SpectralModel):
    pars_name = ('Index1', 'Index2', 'r_br_g', 'a', 'Incl', 'Rin_ms', 'Rout_ms')
    pars_range = ((-10.0, 10.0), (-10.0, 10.0), (1.0, 400.0), (0.0, 0.998), (0.0, 90.0), (1.0, 400.0), (1.0, 400.0))
    pars_default = (3.0, 3.0, 6.0, 0.998, 30.0, 1.0, 400.0)
    pars_frozen = (True, True, True, False, True, True, True)
    def __init__(self, Index1, Index2, r_br_g, a, Incl, Rin_ms, Rout_ms, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('kerrconv', [Index1, Index2, r_br_g, a, Incl, Rin_ms, Rout_ms], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class kerrd(SpectralModel):
    pars_name = ('distance', 'TcoloTeff', 'M', 'Mdot', 'Incl', 'Rin', 'Rout')
    pars_range = ((0.01, 1000.0), (1.0, 2.0), (0.1, 100.0), (0.01, 100.0), (0.0, 90.0), (1.235, 100.0), (10000.0, 100000000.0))
    pars_default = (1.0, 1.5, 1.0, 1.0, 30.0, 1.235, 100000.0)
    pars_frozen = (True, True, False, False, True, True, True)
    def __init__(self, distance, TcoloTeff, M, Mdot, Incl, Rin, Rout, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('kerrd', [distance, TcoloTeff, M, Mdot, Incl, Rin, Rout], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class kerrdisk(SpectralModel):
    pars_name = ('lineE', 'Index1', 'Index2', 'r_br_g', 'a', 'Incl', 'Rin_ms', 'Rout_ms', 'z')
    pars_range = ((0.1, 100.0), (-10.0, 10.0), (-10.0, 10.0), (1.0, 400.0), (0.01, 0.998), (0.0, 90.0), (1.0, 400.0), (1.0, 400.0), (0.0, 10.0))
    pars_default = (6.4, 3.0, 3.0, 6.0, 0.998, 30.0, 1.0, 400.0, 0.0)
    pars_frozen = (True, True, True, True, False, True, True, True, True)
    def __init__(self, lineE, Index1, Index2, r_br_g, a, Incl, Rin_ms, Rout_ms, z, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('kerrdisk', [lineE, Index1, Index2, r_br_g, a, Incl, Rin_ms, Rout_ms, z], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class kyconv(SpectralModel):
    pars_name = ('a', 'theta_o', 'rin', 'ms', 'rout', 'alpha', 'beta', 'rb', 'zshift', 'limb', 'ne_loc', 'normal')
    pars_range = ((0.0, 1.0), (0.0, 89.0), (1.0, 1000.0), (0.0, 1.0), (1.0, 1000.0), (-20.0, 20.0), (-20.0, 20.0), (1.0, 1000.0), (-0.999, 10.0), (0.0, 2.0), (3.0, 5000.0), (-1.0, 100.0))
    pars_default = (0.9982, 30.0, 1.0, 1.0, 400.0, 3.0, 3.0, 400.0, 0.0, 0.0, 100.0, 1.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, a, theta_o, rin, ms, rout, alpha, beta, rb, zshift, limb, ne_loc, normal, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('kyconv', [a, theta_o, rin, ms, rout, alpha, beta, rb, zshift, limb, ne_loc, normal], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class kyrline(SpectralModel):
    pars_name = ('a', 'theta_o', 'rin', 'ms', 'rout', 'Erest', 'alpha', 'beta', 'rb', 'zshift', 'limb')
    pars_range = ((0.0, 1.0), (0.0, 89.0), (1.0, 1000.0), (0.0, 1.0), (1.0, 1000.0), (1.0, 99.0), (-20.0, 20.0), (-20.0, 20.0), (1.0, 1000.0), (-0.999, 10.0), (0.0, 2.0))
    pars_default = (0.9982, 30.0, 1.0, 1.0, 400.0, 6.4, 3.0, 3.0, 400.0, 0.0, 1.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True)
    def __init__(self, a, theta_o, rin, ms, rout, Erest, alpha, beta, rb, zshift, limb, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('kyrline', [a, theta_o, rin, ms, rout, Erest, alpha, beta, rb, zshift, limb], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class laor(SpectralModel):
    pars_name = ('lineE', 'Index', 'Rin_G', 'Rout_G', 'Incl')
    pars_range = ((0.0, 100.0), (-10.0, 10.0), (1.235, 400.0), (1.235, 400.0), (0.0, 90.0))
    pars_default = (6.4, 3.0, 1.235, 400.0, 30.0)
    pars_frozen = (False, True, True, True, True)
    def __init__(self, lineE, Index, Rin_G, Rout_G, Incl, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('laor', [lineE, Index, Rin_G, Rout_G, Incl], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class laor2(SpectralModel):
    pars_name = ('lineE', 'Index', 'Rin_G', 'Rout_G', 'Incl', 'Rbreak', 'Index1')
    pars_range = ((0.0, 100.0), (-10.0, 10.0), (1.235, 400.0), (1.235, 400.0), (0.0, 90.0), (1.235, 400.0), (-10.0, 10.0))
    pars_default = (6.4, 3.0, 1.235, 400.0, 30.0, 20.0, 3.0)
    pars_frozen = (False, True, True, True, True, True, True)
    def __init__(self, lineE, Index, Rin_G, Rout_G, Incl, Rbreak, Index1, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('laor2', [lineE, Index, Rin_G, Rout_G, Incl, Rbreak, Index1], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class log10con(SpectralModel):
    pars_name = ('log10fac',)
    pars_range = ((-20.0, 20.0),)
    pars_default = (0.0,)
    pars_frozen = (False,)
    def __init__(self, log10fac, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('log10con', [log10fac], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class logconst(SpectralModel):
    pars_name = ('logfact',)
    pars_range = ((-20.0, 20.0),)
    pars_default = (0.0,)
    pars_frozen = (False,)
    def __init__(self, logfact, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('logconst', [logfact], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class logpar(SpectralModel):
    pars_name = ('alpha', 'beta', 'pivotE')
    pars_range = ((0.0, 4.0), (-4.0, 4.0), (None, None))
    pars_default = (1.5, 0.2, 1.0)
    pars_frozen = (False, False, True)
    def __init__(self, alpha, beta, pivotE, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('logpar', [alpha, beta, pivotE], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class lorentz(SpectralModel):
    pars_name = ('LineE', 'Width')
    pars_range = ((0.0, 1000000.0), (0.0, 20.0))
    pars_default = (6.5, 0.1)
    pars_frozen = (False, False)
    def __init__(self, LineE, Width, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('lorentz', [LineE, Width], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class lsmooth(SpectralModel):
    pars_name = ('Sig_6keV', 'Index')
    pars_range = ((0.0, 20.0), (-1.0, 1.0))
    pars_default = (1.0, 0.0)
    pars_frozen = (False, True)
    def __init__(self, Sig_6keV, Index, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('lsmooth', [Sig_6keV, Index], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class lyman(SpectralModel):
    pars_name = ('n', 'b', 'z', 'ZA')
    pars_range = ((0.0, 1000000.0), (1.0, 1000000.0), (-0.001, 100000.0), (1.0, 2.0))
    pars_default = (1e-05, 10.0, 0.0, 1.0)
    pars_frozen = (False, False, False, False)
    def __init__(self, n, b, z, ZA, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('lyman', [n, b, z, ZA], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class meka(SpectralModel):
    pars_name = ('kT', 'nH', 'Abundanc', 'Redshift')
    pars_range = ((0.001, 100.0), (1e-06, 1e+20), (0.0, 1000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, True, True, True)
    def __init__(self, kT, nH, Abundanc, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('meka', [kT, nH, Abundanc, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class mekal(SpectralModel):
    pars_name = ('kT', 'nH', 'Abundanc', 'Redshift', 'switch')
    pars_range = ((0.0808, 79.9), (1e-06, 1e+20), (0.0, 1000.0), (-0.999, 10.0), (0.0, 1.0))
    pars_default = (1.0, 1.0, 1.0, 0.0, 1)
    pars_frozen = (False, True, True, True, True)
    def __init__(self, kT, nH, Abundanc, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('mekal', [kT, nH, Abundanc, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class mkcflow(SpectralModel):
    pars_name = ('lowT', 'highT', 'Abundanc', 'Redshift', 'switch')
    pars_range = ((0.0808, 79.9), (0.0808, 79.9), (0.0, 5.0), (-0.999, 10.0), (0.0, 1.0))
    pars_default = (0.1, 4.0, 1.0, 0.0, 1)
    pars_frozen = (False, False, False, True, True)
    def __init__(self, lowT, highT, Abundanc, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('mkcflow', [lowT, highT, Abundanc, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class nei(SpectralModel):
    pars_name = ('kT', 'Abundanc', 'Tau', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 10000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 100000000000.0, 0.0)
    pars_frozen = (False, True, False, True)
    def __init__(self, kT, Abundanc, Tau, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('nei', [kT, Abundanc, Tau, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class nlapec(SpectralModel):
    pars_name = ('kT', 'Abundanc', 'Redshift')
    pars_range = ((0.008, 64.0), (0.0, 5.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 0.0)
    pars_frozen = (False, True, True)
    def __init__(self, kT, Abundanc, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('nlapec', [kT, Abundanc, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class notch(SpectralModel):
    pars_name = ('LineE', 'Width', 'CvrFract')
    pars_range = ((0.0, 20.0), (0.0, 20.0), (0.0, 1.0))
    pars_default = (3.5, 1.0, 1.0)
    pars_frozen = (False, False, True)
    def __init__(self, LineE, Width, CvrFract, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('notch', [LineE, Width, CvrFract], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class npshock(SpectralModel):
    pars_name = ('kT_a', 'kT_b', 'Abundanc', 'Tau_l', 'Tau_u', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.01, 79.9), (0.0, 10000.0), (0.0, 50000000000000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 0.5, 1.0, 0.0, 100000000000.0, 0.0)
    pars_frozen = (False, False, True, True, False, True)
    def __init__(self, kT_a, kT_b, Abundanc, Tau_l, Tau_u, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('npshock', [kT_a, kT_b, Abundanc, Tau_l, Tau_u, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class nsa(SpectralModel):
    pars_name = ('LogT_eff', 'M_ns', 'R_ns', 'MagField')
    pars_range = ((5.0, 7.0), (0.5, 2.5), (5.0, 20.0), (0.0, 50000000000000.0))
    pars_default = (6.0, 1.4, 10.0, 0.0)
    pars_frozen = (False, False, False, True)
    def __init__(self, LogT_eff, M_ns, R_ns, MagField, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('nsa', [LogT_eff, M_ns, R_ns, MagField], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class nsagrav(SpectralModel):
    pars_name = ('LogT_eff', 'NSmass', 'NSrad')
    pars_range = ((5.5, 6.5), (0.3, 2.5), (6.0, 20.0))
    pars_default = (6.0, 1.4, 10.0)
    pars_frozen = (False, False, False)
    def __init__(self, LogT_eff, NSmass, NSrad, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('nsagrav', [LogT_eff, NSmass, NSrad], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class nsatmos(SpectralModel):
    pars_name = ('LogT_eff', 'M_ns', 'R_ns', 'dist')
    pars_range = ((5.0, 6.5), (0.5, 3.0), (5.0, 30.0), (0.1, 100.0))
    pars_default = (6.0, 1.4, 10.0, 10.0)
    pars_frozen = (False, False, False, False)
    def __init__(self, LogT_eff, M_ns, R_ns, dist, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('nsatmos', [LogT_eff, M_ns, R_ns, dist], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class nsmax(SpectralModel):
    pars_name = ('logTeff', 'redshift', 'specfile')
    pars_range = ((5.5, 6.8), (1e-05, 2.0), (None, None))
    pars_default = (6.0, 0.1, 1200)
    pars_frozen = (False, False, True)
    def __init__(self, logTeff, redshift, specfile, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('nsmax', [logTeff, redshift, specfile], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class nsmaxg(SpectralModel):
    pars_name = ('logTeff', 'M_ns', 'R_ns', 'dist', 'specfile')
    pars_range = ((5.5, 6.9), (0.5, 3.0), (5.0, 30.0), (0.01, 100.0), (None, None))
    pars_default = (6.0, 1.4, 10.0, 1.0, 1200)
    pars_frozen = (False, False, False, False, True)
    def __init__(self, logTeff, M_ns, R_ns, dist, specfile, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('nsmaxg', [logTeff, M_ns, R_ns, dist, specfile], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class nsx(SpectralModel):
    pars_name = ('logTeff', 'M_ns', 'R_ns', 'dist', 'specfile')
    pars_range = ((5.5, 6.7), (0.5, 3.0), (5.0, 30.0), (0.01, 100.0), (None, None))
    pars_default = (6.0, 1.4, 10.0, 1.0, 6)
    pars_frozen = (False, False, False, False, True)
    def __init__(self, logTeff, M_ns, R_ns, dist, specfile, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('nsx', [logTeff, M_ns, R_ns, dist, specfile], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class nteea(SpectralModel):
    pars_name = ('l_nth', 'l_bb', 'f_refl', 'kT_bb', 'g_max', 'l_th', 'tau_p', 'G_inj', 'g_min', 'g_0', 'radius', 'pair_esc', 'cosIncl', 'Fe_abund', 'Redshift')
    pars_range = ((0.0, 10000.0), (0.0, 10000.0), (0.0, 4.0), (1.0, 100.0), (5.0, 10000.0), (0.0, 10000.0), (0.0, 10.0), (0.0, 5.0), (1.0, 1000.0), (1.0, 5.0), (100000.0, 1e+16), (0.0, 1.0), (0.05, 0.95), (0.1, 10.0), (-0.999, 10.0))
    pars_default = (100.0, 100.0, 0.0, 10.0, 1000.0, 0.0, 0.0, 0.0, 1.3, 1.3, 10000000000000.0, 0.0, 0.45, 1.0, 0.0)
    pars_frozen = (False, False, False, True, True, True, True, True, True, True, True, True, False, True, True)
    def __init__(self, l_nth, l_bb, f_refl, kT_bb, g_max, l_th, tau_p, G_inj, g_min, g_0, radius, pair_esc, cosIncl, Fe_abund, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('nteea', [l_nth, l_bb, f_refl, kT_bb, g_max, l_th, tau_p, G_inj, g_min, g_0, radius, pair_esc, cosIncl, Fe_abund, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class nthComp(SpectralModel):
    pars_name = ('Gamma', 'kT_e', 'kT_bb', 'inp_type', 'Redshift')
    pars_range = ((1.001, 10.0), (1.0, 1000.0), (0.001, 10.0), (0.0, 1.0), (-0.999, 10.0))
    pars_default = (1.7, 100.0, 0.1, 0.0, 0.0)
    pars_frozen = (False, False, True, True, True)
    def __init__(self, Gamma, kT_e, kT_bb, inp_type, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('nthComp', [Gamma, kT_e, kT_bb, inp_type, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class olivineabs(SpectralModel):
    pars_name = ('moliv', 'redshift')
    pars_range = ((0.0, 100000.0), (-1.0, 10.0))
    pars_default = (1.0, 0.0)
    pars_frozen = (False, True)
    def __init__(self, moliv, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('olivineabs', [moliv, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class optxagn(SpectralModel):
    pars_name = ('mass', 'dist', 'logLoLEdd', 'astar', 'rcor', 'logrout', 'kT_e', 'tau', 'Gamma', 'fpl', 'fcol', 'tscat', 'Redshift')
    pars_range = ((1.0, 1000000000.0), (0.01, 1000000000.0), (-10.0, 2.0), (0.0, 0.998), (1.0, 100.0), (3.0, 7.0), (0.01, 10.0), (0.1, 100.0), (0.5, 10.0), (0.0, 0.1), (1.0, 5.0), (10000.0, 100000.0), (0.0, 10.0))
    pars_default = (10000000.0, 100.0, -1.0, 0.0, 10.0, 5.0, 0.2, 10.0, 2.1, 0.0001, 2.4, 100000.0, 0.0)
    pars_frozen = (True, True, False, True, False, True, False, False, False, False, True, True, True)
    def __init__(self, mass, dist, logLoLEdd, astar, rcor, logrout, kT_e, tau, Gamma, fpl, fcol, tscat, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('optxagn', [mass, dist, logLoLEdd, astar, rcor, logrout, kT_e, tau, Gamma, fpl, fcol, tscat, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class optxagnf(SpectralModel):
    pars_name = ('mass', 'dist', 'logLoLEdd', 'astar', 'rcor', 'logrout', 'kT_e', 'tau', 'Gamma', 'fpl', 'Redshift')
    pars_range = ((1.0, 1000000000.0), (0.01, 1000000000.0), (-10.0, 2.0), (0.0, 0.998), (1.0, 100.0), (3.0, 7.0), (0.01, 10.0), (0.1, 100.0), (1.05, 10.0), (0.0, 1.0), (0.0, 10.0))
    pars_default = (10000000.0, 100.0, -1.0, 0.0, 10.0, 5.0, 0.2, 10.0, 2.1, 0.0001, 0.0)
    pars_frozen = (True, True, False, True, False, True, False, False, False, False, True)
    def __init__(self, mass, dist, logLoLEdd, astar, rcor, logrout, kT_e, tau, Gamma, fpl, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('optxagnf', [mass, dist, logLoLEdd, astar, rcor, logrout, kT_e, tau, Gamma, fpl, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class partcov(SpectralModel):
    pars_name = ('CvrFract',)
    pars_range = ((0.0, 1.0),)
    pars_default = (0.5,)
    pars_frozen = (False,)
    def __init__(self, CvrFract, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('partcov', [CvrFract], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class pcfabs(SpectralModel):
    pars_name = ('nH', 'CvrFract')
    pars_range = ((0.0, 1000000.0), (0.0, 1.0))
    pars_default = (1.0, 0.5)
    pars_frozen = (False, False)
    def __init__(self, nH, CvrFract, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('pcfabs', [nH, CvrFract], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class pegpwrlw(SpectralModel):
    pars_name = ('PhoIndex', 'eMin', 'eMax')
    pars_range = ((-3.0, 10.0), (-100.0, 10000000000.0), (-100.0, 10000000000.0))
    pars_default = (1.0, 2.0, 10.0)
    pars_frozen = (False, True, True)
    def __init__(self, PhoIndex, eMin, eMax, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('pegpwrlw', [PhoIndex, eMin, eMax], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class pexmon(SpectralModel):
    pars_name = ('PhoIndex', 'foldE', 'rel_refl', 'redshift', 'abund', 'Fe_abund', 'Incl')
    pars_range = ((1.1, 2.5), (1.0, 1000000.0), (-1000000.0, 1000000.0), (0.0, 4.0), (0.0, 1000000.0), (0.0, 100.0), (0.0, 85.0))
    pars_default = (2.0, 1000.0, -1.0, 0.0, 1.0, 1.0, 60.0)
    pars_frozen = (False, True, True, True, True, True, False)
    def __init__(self, PhoIndex, foldE, rel_refl, redshift, abund, Fe_abund, Incl, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('pexmon', [PhoIndex, foldE, rel_refl, redshift, abund, Fe_abund, Incl], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class pexrav(SpectralModel):
    pars_name = ('PhoIndex', 'foldE', 'rel_refl', 'Redshift', 'abund', 'Fe_abund', 'cosIncl')
    pars_range = ((-10.0, 10.0), (1.0, 1000000.0), (0.0, 1000000.0), (-0.999, 10.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.05, 0.95))
    pars_default = (2.0, 100.0, 0.0, 0.0, 1.0, 1.0, 0.45)
    pars_frozen = (False, False, False, True, True, True, True)
    def __init__(self, PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('pexrav', [PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class pexriv(SpectralModel):
    pars_name = ('PhoIndex', 'foldE', 'rel_refl', 'Redshift', 'abund', 'Fe_abund', 'cosIncl', 'T_disk', 'xi')
    pars_range = ((-10.0, 10.0), (1.0, 1000000.0), (0.0, 1000000.0), (-0.999, 10.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.05, 0.95), (10000.0, 1000000.0), (0.0, 5000.0))
    pars_default = (2.0, 100.0, 0.0, 0.0, 1.0, 1.0, 0.45, 30000.0, 1.0)
    pars_frozen = (False, False, False, True, True, True, True, True, False)
    def __init__(self, PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('pexriv', [PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, cosIncl, T_disk, xi], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class phabs(SpectralModel):
    pars_name = ('nH',)
    pars_range = ((0.0, 1000000.0),)
    pars_default = (1.0,)
    pars_frozen = (False,)
    def __init__(self, nH, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('phabs', [nH], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class plabs(SpectralModel):
    pars_name = ('index', 'coef')
    pars_range = ((0.0, 5.0), (0.0, 100.0))
    pars_default = (2.0, 1.0)
    pars_frozen = (False, False)
    def __init__(self, index, coef, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('plabs', [index, coef], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class plcabs(SpectralModel):
    pars_name = ('nH', 'nmax', 'FeAbun', 'FeKedge', 'PhoIndex', 'HighECut', 'foldE', 'acrit', 'FAST', 'Redshift')
    pars_range = ((0.0, 1000000.0), (None, None), (0.0, 10.0), (7.0, 10.0), (-3.0, 10.0), (0.01, 200.0), (1.0, 1000000.0), (0.0, 1.0), (None, None), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 7.11, 2.0, 95.0, 100.0, 1.0, 0.0, 0.0)
    pars_frozen = (False, True, True, True, False, True, True, True, True, True)
    def __init__(self, nH, nmax, FeAbun, FeKedge, PhoIndex, HighECut, foldE, acrit, FAST, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('plcabs', [nH, nmax, FeAbun, FeKedge, PhoIndex, HighECut, foldE, acrit, FAST, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class polconst(SpectralModel):
    pars_name = ('A', 'psi')
    pars_range = ((0.0, 1.0), (-90.0, 90.0))
    pars_default = (1.0, 45.0)
    pars_frozen = (False, False)
    def __init__(self, A, psi, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('polconst', [A, psi], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class pollin(SpectralModel):
    pars_name = ('A1', 'Aslope', 'psi1', 'psislope')
    pars_range = ((0.0, 1.0), (-5.0, 5.0), (-90.0, 90.0), (-5.0, 5.0))
    pars_default = (1.0, 0.0, 45.0, 0.0)
    pars_frozen = (False, False, False, False)
    def __init__(self, A1, Aslope, psi1, psislope, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('pollin', [A1, Aslope, psi1, psislope], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class polpow(SpectralModel):
    pars_name = ('Anorm', 'Aindex', 'psinorm', 'psiindex')
    pars_range = ((0.0, 1.0), (-5.0, 5.0), (-90.0, 90.0), (-5.0, 5.0))
    pars_default = (1.0, 0.0, 45.0, 0.0)
    pars_frozen = (False, False, False, False)
    def __init__(self, Anorm, Aindex, psinorm, psiindex, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('polpow', [Anorm, Aindex, psinorm, psiindex], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class posm(SpectralModel):
    pars_name = ()
    pars_range = ()
    pars_default = ()
    pars_frozen = ()
    def __init__(self, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('posm', [], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class powerlaw(SpectralModel):
    pars_name = ('PhoIndex',)
    pars_range = ((-3.0, 10.0),)
    pars_default = (1.0,)
    pars_frozen = (False,)
    def __init__(self, PhoIndex, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('powerlaw', [PhoIndex], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class pshock(SpectralModel):
    pars_name = ('kT', 'Abundanc', 'Tau_l', 'Tau_u', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 10000.0), (0.0, 50000000000000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 0.0, 100000000000.0, 0.0)
    pars_frozen = (False, True, True, False, True)
    def __init__(self, kT, Abundanc, Tau_l, Tau_u, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('pshock', [kT, Abundanc, Tau_l, Tau_u, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class pwab(SpectralModel):
    pars_name = ('nHmin', 'nHmax', 'beta')
    pars_range = ((1e-07, 1000000.0), (1e-07, 1000000.0), (-10.0, 20.0))
    pars_default = (1.0, 2.0, 1.0)
    pars_frozen = (False, False, True)
    def __init__(self, nHmin, nHmax, beta, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('pwab', [nHmin, nHmax, beta], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class qsosed(SpectralModel):
    pars_name = ('mass', 'dist', 'logmdot', 'astar', 'cosi', 'redshift')
    pars_range = ((100000.0, 10000000000.0), (0.01, 1000000000.0), (-1.65, 0.39), (-1.0, 0.998), (0.05, 1.0), (0.0, 5.0))
    pars_default = (10000000.0, 100.0, -1.0, 0.0, 0.5, 0.0)
    pars_frozen = (True, True, False, True, True, True)
    def __init__(self, mass, dist, logmdot, astar, cosi, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('qsosed', [mass, dist, logmdot, astar, cosi, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class raymond(SpectralModel):
    pars_name = ('kT', 'Abundanc', 'Redshift')
    pars_range = ((0.008, 64.0), (0.0, 5.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 0.0)
    pars_frozen = (False, True, True)
    def __init__(self, kT, Abundanc, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('raymond', [kT, Abundanc, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class rdblur(SpectralModel):
    pars_name = ('Betor10', 'Rin_M', 'Rout_M', 'Incl')
    pars_range = ((-10.0, 20.0), (6.0, 10000.0), (0.0, 10000000.0), (0.0, 90.0))
    pars_default = (-2.0, 10.0, 1000.0, 30.0)
    pars_frozen = (True, True, True, False)
    def __init__(self, Betor10, Rin_M, Rout_M, Incl, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('rdblur', [Betor10, Rin_M, Rout_M, Incl], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class redden(SpectralModel):
    pars_name = ('E_BmV',)
    pars_range = ((0.0, 10.0),)
    pars_default = (0.05,)
    pars_frozen = (False,)
    def __init__(self, E_BmV, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('redden', [E_BmV], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class redge(SpectralModel):
    pars_name = ('edge', 'kT')
    pars_range = ((0.001, 100.0), (0.001, 100.0))
    pars_default = (1.4, 1.0)
    pars_frozen = (False, False)
    def __init__(self, edge, kT, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('redge', [edge, kT], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class reflect(SpectralModel):
    pars_name = ('rel_refl', 'Redshift', 'abund', 'Fe_abund', 'cosIncl')
    pars_range = ((-1.0, 1000000.0), (-0.999, 10.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.05, 0.95))
    pars_default = (0.0, 0.0, 1.0, 1.0, 0.45)
    pars_frozen = (False, True, True, True, True)
    def __init__(self, rel_refl, Redshift, abund, Fe_abund, cosIncl, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('reflect', [rel_refl, Redshift, abund, Fe_abund, cosIncl], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class refsch(SpectralModel):
    pars_name = ('PhoIndex', 'foldE', 'rel_refl', 'Redshift', 'abund', 'Fe_abund', 'Incl', 'T_disk', 'xi', 'Betor10', 'Rin', 'Rout', 'accuracy')
    pars_range = ((-10.0, 10.0), (1.0, 1000000.0), (0.0, 2.0), (-0.999, 10.0), (0.5, 10.0), (0.1, 10.0), (19.0, 87.0), (10000.0, 1000000.0), (0.0, 5000.0), (-10.0, 20.0), (6.0, 10000.0), (0.0, 10000000.0), (30.0, 100000.0))
    pars_default = (2.0, 100.0, 0.0, 0.0, 1.0, 1.0, 30.0, 30000.0, 1.0, -2.0, 10.0, 1000.0, 30.0)
    pars_frozen = (False, False, False, True, True, True, True, True, False, True, True, True, True)
    def __init__(self, PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, Incl, T_disk, xi, Betor10, Rin, Rout, accuracy, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('refsch', [PhoIndex, foldE, rel_refl, Redshift, abund, Fe_abund, Incl, T_disk, xi, Betor10, Rin, Rout, accuracy], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class rfxconv(SpectralModel):
    pars_name = ('rel_refl', 'redshift', 'Fe_abund', 'cosIncl', 'log_xi')
    pars_range = ((-1.0, 1000000.0), (0.0, 4.0), (0.5, 3.0), (0.05, 0.95), (1.0, 6.0))
    pars_default = (-1.0, 0.0, 1.0, 0.5, 1.0)
    pars_frozen = (False, True, True, True, False)
    def __init__(self, rel_refl, redshift, Fe_abund, cosIncl, log_xi, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('rfxconv', [rel_refl, redshift, Fe_abund, cosIncl, log_xi], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class rgsxsrc(SpectralModel):
    pars_name = ('order',)
    pars_range = ((-3.0, -1.0),)
    pars_default = (-1.0,)
    pars_frozen = (True,)
    def __init__(self, order, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('rgsxsrc', [order], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class rnei(SpectralModel):
    pars_name = ('kT', 'kT_init', 'Abundanc', 'Tau', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0808, 79.9), (0.0, 10000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (0.5, 1.0, 1.0, 100000000000.0, 0.0)
    pars_frozen = (False, False, True, False, True)
    def __init__(self, kT, kT_init, Abundanc, Tau, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('rnei', [kT, kT_init, Abundanc, Tau, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class sedov(SpectralModel):
    pars_name = ('kT_a', 'kT_b', 'Abundanc', 'Tau', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.01, 79.9), (0.0, 10000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 0.5, 1.0, 100000000000.0, 0.0)
    pars_frozen = (False, False, True, False, True)
    def __init__(self, kT_a, kT_b, Abundanc, Tau, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('sedov', [kT_a, kT_b, Abundanc, Tau, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class simpl(SpectralModel):
    pars_name = ('Gamma', 'FracSctr', 'UpScOnly')
    pars_range = ((1.0, 5.0), (0.0, 1.0), (0.0, 100.0))
    pars_default = (2.3, 0.05, 1.0)
    pars_frozen = (False, False, True)
    def __init__(self, Gamma, FracSctr, UpScOnly, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('simpl', [Gamma, FracSctr, UpScOnly], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class sirf(SpectralModel):
    pars_name = ('tin', 'rin', 'rout', 'theta', 'incl', 'valpha', 'gamma', 'mdot', 'irrad')
    pars_range = ((0.01, 1000.0), (1e-06, 10.0), (0.1, 100000000.0), (0.0, 90.0), (-90.0, 90.0), (-1.5, 5.0), (0.5, 10.0), (0.5, 10000000.0), (0.0, 20.0))
    pars_default = (1.0, 0.01, 100.0, 22.9, 0.0, -0.5, 1.333, 1000.0, 2.0)
    pars_frozen = (False, False, False, False, True, True, True, True, True)
    def __init__(self, tin, rin, rout, theta, incl, valpha, gamma, mdot, irrad, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('sirf', [tin, rin, rout, theta, incl, valpha, gamma, mdot, irrad], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class slimbh(SpectralModel):
    pars_name = ('M', 'a', 'lumin', 'alpha', 'inc', 'D', 'f_hard', 'lflag', 'vflag')
    pars_range = ((0.0, 1000.0), (0.0, 0.999), (0.05, 1.0), (0.005, 0.1), (0.0, 85.0), (0.0, 10000.0), (-10.0, 10.0), (None, None), (None, None))
    pars_default = (10.0, 0.0, 0.5, 0.1, 60.0, 10.0, -1.0, 1, 1)
    pars_frozen = (True, False, False, True, True, True, True, True, True)
    def __init__(self, M, a, lumin, alpha, inc, D, f_hard, lflag, vflag, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('slimbh', [M, a, lumin, alpha, inc, D, f_hard, lflag, vflag], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class smaug(SpectralModel):
    pars_name = ('kT_cc', 'kT_dt', 'kT_ix', 'kT_ir', 'kT_cx', 'kT_cr', 'kT_tx', 'kT_tr', 'nH_cc', 'nH_ff', 'nH_cx', 'nH_cr', 'nH_gx', 'nH_gr', 'Ab_cc', 'Ab_xx', 'Ab_rr', 'redshift', 'meshpts', 'rcutoff', 'mode', 'itype')
    pars_range = ((0.08, 100.0), (0.0, 100.0), (0.0, 10.0), (0.0001, 1.0), (0.0, 10.0), (0.0001, 20.0), (0.0, 10.0), (0.0001, 3.0), (1e-06, 3.0), (0.0, 1.0), (0.0, 10.0), (0.0001, 2.0), (0.0, 10.0), (0.0001, 20.0), (0.0, 5.0), (0.0, 10.0), (0.0001, 1.0), (0.0001, 10.0), (1.0, 10000.0), (1.0, 3.0), (0.0, 2.0), (1.0, 4.0))
    pars_default = (1.0, 1.0, 0.0, 0.1, 0.5, 0.1, 0.0, 0.5, 1.0, 1.0, 0.5, 0.1, 0.0, 0.002, 1.0, 0.0, 0.1, 0.01, 10.0, 2.0, 1.0, 2.0)
    pars_frozen = (False, False, True, True, False, False, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT_cc, kT_dt, kT_ix, kT_ir, kT_cx, kT_cr, kT_tx, kT_tr, nH_cc, nH_ff, nH_cx, nH_cr, nH_gx, nH_gr, Ab_cc, Ab_xx, Ab_rr, redshift, meshpts, rcutoff, mode, itype, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('smaug', [kT_cc, kT_dt, kT_ix, kT_ir, kT_cx, kT_cr, kT_tx, kT_tr, nH_cc, nH_ff, nH_cx, nH_cr, nH_gx, nH_gr, Ab_cc, Ab_xx, Ab_rr, redshift, meshpts, rcutoff, mode, itype], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class smedge(SpectralModel):
    pars_name = ('edgeE', 'MaxTau', 'index', 'width')
    pars_range = ((0.1, 100.0), (0.0, 10.0), (-10.0, 10.0), (0.01, 100.0))
    pars_default = (7.0, 1.0, -2.67, 10.0)
    pars_frozen = (False, False, True, False)
    def __init__(self, edgeE, MaxTau, index, width, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('smedge', [edgeE, MaxTau, index, width], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class snapec(SpectralModel):
    pars_name = ('kT', 'N_SNe', 'R', 'SNIModelIndex', 'SNIIModelIndex', 'redshift')
    pars_range = ((0.008, 64.0), (0.0, 1e+20), (0.0, 1e+20), (0.0, 125.0), (0.0, 125.0), (0.0, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, False, False, True, True, True)
    def __init__(self, kT, N_SNe, R, SNIModelIndex, SNIIModelIndex, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('snapec', [kT, N_SNe, R, SNIModelIndex, SNIIModelIndex, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class spexpcut(SpectralModel):
    pars_name = ('Ecut', 'alpha')
    pars_range = ((0.0, 1000000.0), (-5.0, 5.0))
    pars_default = (10.0, 1.0)
    pars_frozen = (False, False)
    def __init__(self, Ecut, alpha, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('spexpcut', [Ecut, alpha], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class spline(SpectralModel):
    pars_name = ('Estart', 'Ystart', 'Yend', 'YPstart', 'YPend', 'Eend')
    pars_range = ((0.0, 100.0), (-1000000.0, 1000000.0), (-1000000.0, 1000000.0), (-1000000.0, 1000000.0), (-1000000.0, 1000000.0), (0.0, 100.0))
    pars_default = (0.1, 1.0, 1.0, 0.0, 0.0, 15.0)
    pars_frozen = (False, False, False, False, False, False)
    def __init__(self, Estart, Ystart, Yend, YPstart, YPend, Eend, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('spline', [Estart, Ystart, Yend, YPstart, YPend, Eend], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class srcut(SpectralModel):
    pars_name = ('alpha', 'break_')
    pars_range = ((1e-05, 1.0), (10000000000.0, 1e+25))
    pars_default = (0.5, 2.42e+17)
    pars_frozen = (False, False)
    def __init__(self, alpha, break_, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('srcut', [alpha, break_], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class sresc(SpectralModel):
    pars_name = ('alpha', 'rolloff')
    pars_range = ((1e-05, 1.0), (10000000000.0, 1e+25))
    pars_default = (0.5, 2.42e+17)
    pars_frozen = (False, False)
    def __init__(self, alpha, rolloff, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('sresc', [alpha, rolloff], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class ssa(SpectralModel):
    pars_name = ('te', 'y')
    pars_range = ((0.01, 0.5), (0.0001, 1000.0))
    pars_default = (0.1, 0.7)
    pars_frozen = (False, False)
    def __init__(self, te, y, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('ssa', [te, y], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class step(SpectralModel):
    pars_name = ('Energy', 'Sigma')
    pars_range = ((0.0, 100.0), (0.0, 20.0))
    pars_default = (6.5, 0.1)
    pars_frozen = (False, False)
    def __init__(self, Energy, Sigma, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('step', [Energy, Sigma], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class swind1(SpectralModel):
    pars_name = ('column', 'log_xi', 'sigma', 'Redshift')
    pars_range = ((3.0, 50.0), (2.1, 4.1), (0.0, 0.5), (-0.999, 10.0))
    pars_default = (6.0, 2.5, 0.1, 0.0)
    pars_frozen = (False, False, False, True)
    def __init__(self, column, log_xi, sigma, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('swind1', [column, log_xi, sigma, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class tapec(SpectralModel):
    pars_name = ('kT', 'kTi', 'Abundanc', 'Redshift')
    pars_range = ((0.008, 64.0), (0.008, 64.0), (0.0, 5.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, False, True, True)
    def __init__(self, kT, kTi, Abundanc, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('tapec', [kT, kTi, Abundanc, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class thcomp(SpectralModel):
    pars_name = ('Gamma_tau', 'kT_e', 'cov_frac', 'z')
    pars_range = ((1.001, 10.0), (0.5, 150.0), (0.0, 1.0), (0.0, 5.0))
    pars_default = (1.7, 50.0, 1.0, 0.0)
    pars_frozen = (False, False, False, True)
    def __init__(self, Gamma_tau, kT_e, cov_frac, z, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('thcomp', [Gamma_tau, kT_e, cov_frac, z], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class uvred(SpectralModel):
    pars_name = ('E_BmV',)
    pars_range = ((0.0, 10.0),)
    pars_default = (0.05,)
    pars_frozen = (False,)
    def __init__(self, E_BmV, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('uvred', [E_BmV], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vapec(SpectralModel):
    pars_name = ('kT', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift')
    pars_range = ((0.0808, 68.447), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0))
    pars_default = (6.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vapec', [kT, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class varabs(SpectralModel):
    pars_name = ('H', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Cl', 'Ar', 'Ca', 'Cr', 'Fe', 'Co', 'Ni')
    pars_range = ((0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    pars_frozen = (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, H, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('varabs', [H, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vashift(SpectralModel):
    pars_name = ('Velocity',)
    pars_range = ((-10000.0, 10000.0),)
    pars_default = (0.0,)
    pars_frozen = (True,)
    def __init__(self, Velocity, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vashift', [Velocity], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vbremss(SpectralModel):
    pars_name = ('kT', 'HeovrH')
    pars_range = ((0.0001, 200.0), (0.0, 100.0))
    pars_default = (3.0, 1.0)
    pars_frozen = (False, False)
    def __init__(self, kT, HeovrH, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vbremss', [kT, HeovrH], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vcph(SpectralModel):
    pars_name = ('peakT', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift', 'switch')
    pars_range = ((0.1, 100.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 50.0), (None, None))
    pars_default = (2.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, peakT, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vcph', [peakT, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vequil(SpectralModel):
    pars_name = ('kT', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vequil', [kT, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vgadem(SpectralModel):
    pars_name = ('Tmean', 'Tsigma', 'nH', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift', 'switch')
    pars_range = ((0.01, 20.0), (0.01, 100.0), (1e-06, 1e+20), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-0.999, 10.0), (None, None))
    pars_default = (4.0, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2)
    pars_frozen = (True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, Tmean, Tsigma, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vgadem', [Tmean, Tsigma, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vgnei(SpectralModel):
    pars_name = ('kT', 'H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Tau', 'meankT', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 1.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (100000000.0, 50000000000000.0), (0.0808, 79.9), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100000000000.0, 1.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True)
    def __init__(self, kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, meankT, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vgnei', [kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, meankT, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vmcflow(SpectralModel):
    pars_name = ('lowT', 'highT', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift', 'switch')
    pars_range = ((0.0808, 79.9), (0.0808, 79.9), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 10.0), (None, None))
    pars_default = (0.1, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, lowT, highT, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vmcflow', [lowT, highT, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vmeka(SpectralModel):
    pars_name = ('kT', 'nH', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift')
    pars_range = ((0.001, 100.0), (1e-06, 1e+20), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vmeka', [kT, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vmekal(SpectralModel):
    pars_name = ('kT', 'nH', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift', 'switch')
    pars_range = ((0.0808, 79.9), (1e-06, 1e+20), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0), (None, None))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vmekal', [kT, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vmshift(SpectralModel):
    pars_name = ('Velocity',)
    pars_range = ((-10000.0, 10000.0),)
    pars_default = (0.0,)
    pars_frozen = (True,)
    def __init__(self, Velocity, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vmshift', [Velocity], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vnei(SpectralModel):
    pars_name = ('kT', 'H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Tau', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 1.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100000000000.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True)
    def __init__(self, kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vnei', [kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vnpshock(SpectralModel):
    pars_name = ('kT_a', 'kT_b', 'H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Tau_l', 'Tau_u', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.01, 79.9), (0.0, 1.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 50000000000000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 100000000000.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True)
    def __init__(self, kT_a, kT_b, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau_l, Tau_u, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vnpshock', [kT_a, kT_b, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau_l, Tau_u, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class voigt(SpectralModel):
    pars_name = ('LineE', 'Sigma', 'Gamma')
    pars_range = ((0.0, 1000000.0), (0.0, 20.0), (0.0, 20.0))
    pars_default = (6.5, 0.01, 0.01)
    pars_frozen = (False, False, False)
    def __init__(self, LineE, Sigma, Gamma, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('voigt', [LineE, Sigma, Gamma], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vphabs(SpectralModel):
    pars_name = ('nH', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Cl', 'Ar', 'Ca', 'Cr', 'Fe', 'Co', 'Ni')
    pars_range = ((0.0, 1000000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vphabs', [nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vpshock(SpectralModel):
    pars_name = ('kT', 'H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Tau_l', 'Tau_u', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 1.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 50000000000000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 100000000000.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True)
    def __init__(self, kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau_l, Tau_u, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vpshock', [kT, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau_l, Tau_u, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vraymond(SpectralModel):
    pars_name = ('kT', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (-0.999, 10.0))
    pars_default = (6.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vraymond', [kT, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vrnei(SpectralModel):
    pars_name = ('kT', 'kT_init', 'H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Tau', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0808, 79.9), (0.0, 1.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100000000000.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True)
    def __init__(self, kT, kT_init, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vrnei', [kT, kT_init, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vsedov(SpectralModel):
    pars_name = ('kT_a', 'kT_b', 'H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Tau', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.01, 79.9), (0.0, 1.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100000000000.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True)
    def __init__(self, kT_a, kT_b, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vsedov', [kT_a, kT_b, H, He, C, N, O, Ne, Mg, Si, S, Ar, Ca, Fe, Ni, Tau, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vtapec(SpectralModel):
    pars_name = ('kT', 'kTi', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift')
    pars_range = ((0.0808, 68.447), (0.0808, 68.447), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0))
    pars_default = (6.5, 6.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, kTi, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vtapec', [kT, kTi, He, C, N, O, Ne, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vvapec(SpectralModel):
    pars_name = ('kT', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Redshift')
    pars_range = ((0.0808, 68.447), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0))
    pars_default = (6.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vvapec', [kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vvgnei(SpectralModel):
    pars_name = ('kT', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Tau', 'meankT', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (100000000.0, 50000000000000.0), (0.0808, 79.9), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100000000000.0, 1.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True)
    def __init__(self, kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, meankT, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vvgnei', [kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, meankT, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vvnei(SpectralModel):
    pars_name = ('kT', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Tau', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100000000000.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True)
    def __init__(self, kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vvnei', [kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vvnpshock(SpectralModel):
    pars_name = ('kT_a', 'kT_b', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Tau_l', 'Tau_u', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.01, 79.9), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 50000000000000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 100000000000.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True)
    def __init__(self, kT_a, kT_b, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau_l, Tau_u, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vvnpshock', [kT_a, kT_b, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau_l, Tau_u, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vvpshock(SpectralModel):
    pars_name = ('kT', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Tau_l', 'Tau_u', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 50000000000000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 100000000000.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True)
    def __init__(self, kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau_l, Tau_u, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vvpshock', [kT, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau_l, Tau_u, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vvrnei(SpectralModel):
    pars_name = ('kT', 'kT_init', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Tau', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.0808, 79.9), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100000000000.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True)
    def __init__(self, kT, kT_init, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vvrnei', [kT, kT_init, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vvsedov(SpectralModel):
    pars_name = ('kT_a', 'kT_b', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Tau', 'Redshift')
    pars_range = ((0.0808, 79.9), (0.01, 79.9), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (100000000.0, 50000000000000.0), (-0.999, 10.0))
    pars_default = (1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100000000000.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True)
    def __init__(self, kT_a, kT_b, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vvsedov', [kT_a, kT_b, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Tau, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vvtapec(SpectralModel):
    pars_name = ('kT', 'kTi', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Redshift')
    pars_range = ((0.0808, 68.447), (0.0808, 68.447), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0))
    pars_default = (6.5, 6.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, kT, kTi, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vvtapec', [kT, kTi, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vvwdem(SpectralModel):
    pars_name = ('Tmax', 'beta', 'inv_slope', 'nH', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Redshift', 'switch')
    pars_range = ((0.01, 20.0), (0.01, 1.0), (-1.0, 10.0), (1e-06, 1e+20), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0), (None, None))
    pars_default = (1.0, 0.1, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2)
    pars_frozen = (False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, Tmax, beta, inv_slope, nH, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vvwdem', [Tmax, beta, inv_slope, nH, H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P, S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class vwdem(SpectralModel):
    pars_name = ('Tmax', 'beta', 'inv_slope', 'nH', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Ar', 'Ca', 'Fe', 'Ni', 'Redshift', 'switch')
    pars_range = ((0.01, 20.0), (0.01, 1.0), (-1.0, 10.0), (1e-06, 1e+20), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (-0.999, 10.0), (None, None))
    pars_default = (1.0, 0.1, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2)
    pars_frozen = (False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, Tmax, beta, inv_slope, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('vwdem', [Tmax, beta, inv_slope, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class wabs(SpectralModel):
    pars_name = ('nH',)
    pars_range = ((0.0, 1000000.0),)
    pars_default = (1.0,)
    pars_frozen = (False,)
    def __init__(self, nH, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('wabs', [nH], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class wdem(SpectralModel):
    pars_name = ('Tmax', 'beta', 'inv_slope', 'nH', 'abundanc', 'Redshift', 'switch')
    pars_range = ((0.01, 20.0), (0.01, 1.0), (-1.0, 10.0), (1e-06, 1e+20), (0.0, 10.0), (-0.999, 10.0), (None, None))
    pars_default = (1.0, 0.1, 0.25, 1.0, 1.0, 0.0, 2)
    pars_frozen = (False, False, False, True, True, True, True)
    def __init__(self, Tmax, beta, inv_slope, nH, abundanc, Redshift, switch, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('wdem', [Tmax, beta, inv_slope, nH, abundanc, Redshift, switch], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class wndabs(SpectralModel):
    pars_name = ('nH', 'WindowE')
    pars_range = ((0.0, 20.0), (0.03, 20.0))
    pars_default = (1.0, 1.0)
    pars_frozen = (False, False)
    def __init__(self, nH, WindowE, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('wndabs', [nH, WindowE], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class xilconv(SpectralModel):
    pars_name = ('rel_refl', 'redshift', 'Fe_abund', 'cosIncl', 'log_xi', 'cutoff')
    pars_range = ((-1.0, 1000000.0), (0.0, 4.0), (0.5, 3.0), (0.05, 0.95), (1.0, 6.0), (20.0, 300.0))
    pars_default = (-1.0, 0.0, 1.0, 0.5, 1.0, 300.0)
    pars_frozen = (False, True, True, True, False, True)
    def __init__(self, rel_refl, redshift, Fe_abund, cosIncl, log_xi, cutoff, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('xilconv', [rel_refl, redshift, Fe_abund, cosIncl, log_xi, cutoff], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class xion(SpectralModel):
    pars_name = ('height', 'lxovrld', 'rate', 'cosAng', 'inner', 'outer', 'index', 'Redshift', 'Feabun', 'E_cut', 'Ref_type', 'Rel_smear', 'Geometry')
    pars_range = ((0.0, 100.0), (0.02, 100.0), (0.001, 1.0), (0.0, 1.0), (2.0, 1000.0), (2.1, 100000.0), (1.6, 2.2), (-0.999, 10.0), (0.0, 5.0), (20.0, 300.0), (1.0, 3.0), (1.0, 4.0), (1.0, 4.0))
    pars_default = (5.0, 0.3, 0.05, 0.9, 3.0, 100.0, 2.0, 0.0, 1.0, 150.0, 1.0, 4.0, 1.0)
    pars_frozen = (False, False, False, False, False, False, False, True, True, False, True, True, True)
    def __init__(self, height, lxovrld, rate, cosAng, inner, outer, index, Redshift, Feabun, E_cut, Ref_type, Rel_smear, Geometry, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('xion', [height, lxovrld, rate, cosAng, inner, outer, index, Redshift, Feabun, E_cut, Ref_type, Rel_smear, Geometry], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class xscat(SpectralModel):
    pars_name = ('NH', 'Xpos', 'Rext', 'DustModel')
    pars_range = ((0.0, 1000.0), (0.0, 0.999), (0.0, 240.0), (None, None))
    pars_default = (1.0, 0.5, 10.0, 1)
    pars_frozen = (False, False, True, True)
    def __init__(self, NH, Xpos, Rext, DustModel, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('xscat', [NH, Xpos, Rext, DustModel], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zTBabs(SpectralModel):
    pars_name = ('nH', 'Redshift')
    pars_range = ((0.0, 1000000.0), (-0.999, 10.0))
    pars_default = (1.0, 0.0)
    pars_frozen = (False, True)
    def __init__(self, nH, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zTBabs', [nH, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zagauss(SpectralModel):
    pars_name = ('LineE', 'Sigma', 'Redshift')
    pars_range = ((0.0, 1000000.0), (0.0, 1000000.0), (-0.999, 10.0))
    pars_default = (10.0, 1.0, 0.0)
    pars_frozen = (False, False, True)
    def __init__(self, LineE, Sigma, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zagauss', [LineE, Sigma, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zashift(SpectralModel):
    pars_name = ('Redshift',)
    pars_range = ((-0.999, 10.0),)
    pars_default = (0.0,)
    pars_frozen = (True,)
    def __init__(self, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zashift', [Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zbabs(SpectralModel):
    pars_name = ('nH', 'nHeI', 'nHeII', 'z')
    pars_range = ((0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0), (0.0, 1000000.0))
    pars_default = (0.0001, 1e-05, 1e-06, 0.0)
    pars_frozen = (False, False, False, False)
    def __init__(self, nH, nHeI, nHeII, z, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zbabs', [nH, nHeI, nHeII, z], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zbbody(SpectralModel):
    pars_name = ('kT', 'Redshift')
    pars_range = ((0.0001, 200.0), (-0.999, 10.0))
    pars_default = (3.0, 0.0)
    pars_frozen = (False, True)
    def __init__(self, kT, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zbbody', [kT, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zbknpower(SpectralModel):
    pars_name = ('PhoIndx1', 'BreakE', 'PhoIndx2', 'Redshift')
    pars_range = ((-3.0, 10.0), (0.0, 1000000.0), (-3.0, 10.0), (-0.999, 10.0))
    pars_default = (1.0, 5.0, 2.0, 0.0)
    pars_frozen = (False, False, False, True)
    def __init__(self, PhoIndx1, BreakE, PhoIndx2, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zbknpower', [PhoIndx1, BreakE, PhoIndx2, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zbremss(SpectralModel):
    pars_name = ('kT', 'Redshift')
    pars_range = ((0.0001, 200.0), (-0.999, 10.0))
    pars_default = (7.0, 0.0)
    pars_frozen = (False, True)
    def __init__(self, kT, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zbremss', [kT, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zcutoffpl(SpectralModel):
    pars_name = ('PhoIndex', 'HighECut', 'Redshift')
    pars_range = ((-3.0, 10.0), (0.01, 500.0), (-0.999, 10.0))
    pars_default = (1.0, 15.0, 0.0)
    pars_frozen = (False, False, True)
    def __init__(self, PhoIndex, HighECut, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zcutoffpl', [PhoIndex, HighECut, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zdust(SpectralModel):
    pars_name = ('method', 'E_BmV', 'Rv', 'Redshift')
    pars_range = ((None, None), (0.0, 100.0), (0.0, 10.0), (0.0, 20.0))
    pars_default = (1, 0.1, 3.1, 0.0)
    pars_frozen = (True, False, True, True)
    def __init__(self, method, E_BmV, Rv, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zdust', [method, E_BmV, Rv, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zedge(SpectralModel):
    pars_name = ('edgeE', 'MaxTau', 'Redshift')
    pars_range = ((0.0, 100.0), (0.0, 10.0), (-0.999, 10.0))
    pars_default = (7.0, 1.0, 0.0)
    pars_frozen = (False, False, True)
    def __init__(self, edgeE, MaxTau, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zedge', [edgeE, MaxTau, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zgauss(SpectralModel):
    pars_name = ('LineE', 'Sigma', 'Redshift')
    pars_range = ((0.0, 1000000.0), (0.0, 20.0), (-0.999, 10.0))
    pars_default = (6.5, 0.1, 0.0)
    pars_frozen = (False, False, True)
    def __init__(self, LineE, Sigma, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zgauss', [LineE, Sigma, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zhighect(SpectralModel):
    pars_name = ('cutoffE', 'foldE', 'Redshift')
    pars_range = ((0.0001, 200.0), (0.0001, 200.0), (-0.999, 10.0))
    pars_default = (10.0, 15.0, 0.0)
    pars_frozen = (False, False, True)
    def __init__(self, cutoffE, foldE, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zhighect', [cutoffE, foldE, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zigm(SpectralModel):
    pars_name = ('redshift', 'model', 'lyman_limit')
    pars_range = ((None, None), (None, None), (None, None))
    pars_default = (0.0, 0, 1)
    pars_frozen = (True, True, True)
    def __init__(self, redshift, model, lyman_limit, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zigm', [redshift, model, lyman_limit], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zkerrbb(SpectralModel):
    pars_name = ('eta', 'a', 'i', 'Mbh', 'Mdd', 'z', 'fcol', 'rflag', 'lflag')
    pars_range = ((0.0, 1.0), (-0.99, 0.9999), (0.0, 85.0), (3.0, 10000000000.0), (1e-05, 100000.0), (0.0, 10.0), (-100.0, 100.0), (None, None), (None, None))
    pars_default = (0.0, 0.5, 30.0, 10000000.0, 1.0, 0.01, 2.0, 1, 1)
    pars_frozen = (True, False, True, False, False, True, True, True, True)
    def __init__(self, eta, a, i, Mbh, Mdd, z, fcol, rflag, lflag, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zkerrbb', [eta, a, i, Mbh, Mdd, z, fcol, rflag, lflag], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zlogpar(SpectralModel):
    pars_name = ('alpha', 'beta', 'pivotE', 'Redshift')
    pars_range = ((0.0, 4.0), (-4.0, 4.0), (None, None), (-0.999, 10.0))
    pars_default = (1.5, 0.2, 1.0, 0.0)
    pars_frozen = (False, False, True, True)
    def __init__(self, alpha, beta, pivotE, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zlogpar', [alpha, beta, pivotE, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zmshift(SpectralModel):
    pars_name = ('Redshift',)
    pars_range = ((-0.999, 10.0),)
    pars_default = (0.0,)
    pars_frozen = (True,)
    def __init__(self, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zmshift', [Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zpcfabs(SpectralModel):
    pars_name = ('nH', 'CvrFract', 'Redshift')
    pars_range = ((0.0, 1000000.0), (0.0, 1.0), (-0.999, 10.0))
    pars_default = (1.0, 0.5, 0.0)
    pars_frozen = (False, False, True)
    def __init__(self, nH, CvrFract, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zpcfabs', [nH, CvrFract, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zphabs(SpectralModel):
    pars_name = ('nH', 'Redshift')
    pars_range = ((0.0, 1000000.0), (-0.999, 10.0))
    pars_default = (1.0, 0.0)
    pars_frozen = (False, True)
    def __init__(self, nH, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zphabs', [nH, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zpowerlw(SpectralModel):
    pars_name = ('PhoIndex', 'Redshift')
    pars_range = ((-3.0, 10.0), (-0.999, 10.0))
    pars_default = (1.0, 0.0)
    pars_frozen = (False, True)
    def __init__(self, PhoIndex, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zpowerlw', [PhoIndex, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zredden(SpectralModel):
    pars_name = ('E_BmV', 'Redshift')
    pars_range = ((0.0, 10.0), (-0.999, 10.0))
    pars_default = (0.05, 0.0)
    pars_frozen = (False, True)
    def __init__(self, E_BmV, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zredden', [E_BmV, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zsmdust(SpectralModel):
    pars_name = ('E_BmV', 'ExtIndex', 'Rv', 'redshift')
    pars_range = ((0.0, 100.0), (-10.0, 10.0), (0.0, 10.0), (0.0, 20.0))
    pars_default = (0.1, 1.0, 3.1, 0.0)
    pars_frozen = (False, False, True, True)
    def __init__(self, E_BmV, ExtIndex, Rv, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zsmdust', [E_BmV, ExtIndex, Rv, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zvarabs(SpectralModel):
    pars_name = ('H', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Cl', 'Ar', 'Ca', 'Cr', 'Fe', 'Co', 'Ni', 'Redshift')
    pars_range = ((0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (0.0, 10000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    pars_frozen = (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, H, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zvarabs', [H, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zvfeabs(SpectralModel):
    pars_name = ('nH', 'metals', 'FEabun', 'FEKedge', 'Redshift')
    pars_range = ((0.0, 1000000.0), (0.0, 100.0), (0.0, 100.0), (7.0, 9.5), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 7.11, 0.0)
    pars_frozen = (False, False, False, False, True)
    def __init__(self, nH, metals, FEabun, FEKedge, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zvfeabs', [nH, metals, FEabun, FEKedge, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zvphabs(SpectralModel):
    pars_name = ('nH', 'He', 'C', 'N', 'O', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'S', 'Cl', 'Ar', 'Ca', 'Cr', 'Fe', 'Co', 'Ni', 'Redshift')
    pars_range = ((0.0, 1000000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (0.0, 1000.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    pars_frozen = (False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)
    def __init__(self, nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zvphabs', [nH, He, C, N, O, Ne, Na, Mg, Al, Si, S, Cl, Ar, Ca, Cr, Fe, Co, Ni, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zwabs(SpectralModel):
    pars_name = ('nH', 'Redshift')
    pars_range = ((0.0, 1000000.0), (-0.999, 10.0))
    pars_default = (1.0, 0.0)
    pars_frozen = (False, True)
    def __init__(self, nH, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zwabs', [nH, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zwndabs(SpectralModel):
    pars_name = ('nH', 'WindowE', 'Redshift')
    pars_range = ((0.0, 20.0), (0.03, 20.0), (-0.999, 10.0))
    pars_default = (1.0, 1.0, 0.0)
    pars_frozen = (False, False, True)
    def __init__(self, nH, WindowE, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zwndabs', [nH, WindowE, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zxipab(SpectralModel):
    pars_name = ('nHmin', 'nHmax', 'beta', 'log_xi', 'redshift')
    pars_range = ((1e-07, 1000000.0), (1e-07, 1000000.0), (-10.0, 10.0), (-3.0, 6.0), (0.0, 10.0))
    pars_default = (0.01, 10.0, 0.0, 3.0, 0.0)
    pars_frozen = (False, False, False, False, True)
    def __init__(self, nHmin, nHmax, beta, log_xi, redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zxipab', [nHmin, nHmax, beta, log_xi, redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    

class zxipcf(SpectralModel):
    pars_name = ('Nh', 'log_xi', 'CvrFract', 'Redshift')
    pars_range = ((0.05, 500.0), (-3.0, 6.0), (0.0, 1.0), (-0.999, 10.0))
    pars_default = (10.0, 3.0, 0.5, 0.0)
    pars_frozen = (False, False, False, True)
    def __init__(self, Nh, log_xi, CvrFract, Redshift, *, settings='', grad_method='f', eps=1e-7):
        op = XspecNumericGradOp('zxipcf', [Nh, log_xi, CvrFract, Redshift], settings, grad_method, eps)
        super().__init__(op, op.optype)
    