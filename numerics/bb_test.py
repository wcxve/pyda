"""
Created at 01:48:36 on 2023-05-14

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import numpy as np
import matplotlib.pyplot as plt
from mpmath import log1p, log, polylog, exp, mp, mpf
from mxspec._pymXspec import callModFunc
from pyda.numerics.specfun import *

mp.dps = 250

def bb(kT, ebins):
    flux = []
    kT = mpf(kT)
    norm = mpf(8.0525/(kT*kT*kT))
    E = ebins[0]
    exp_EkT = exp(-E/kT)
    integral_low = E*E*log1p(-exp_EkT) - 2.0*kT*(E*polylog(2,exp_EkT) + kT*polylog(3,exp_EkT))
    for i in range(1, len(ebins)):
        E = ebins[i]
        exp_EkT = exp(-E/kT)
        integral_high = E*E*log1p(-exp_EkT) - 2.0*kT*(E*polylog(2,exp_EkT) + kT*polylog(3,exp_EkT))
        flux.append(norm*(integral_high - integral_low))
        integral_low = integral_high
    return flux

def bbrad(kT, ebins):
    flux = []
    kT = mpf(kT)
    norm = mpf(0.0010344 * kT)
    E = ebins[0]
    exp_EkT = exp(-E/kT)
    integral_low = E*E*log1p(-exp_EkT) - 2.0*kT*(E*polylog(2,exp_EkT) + kT*polylog(3,exp_EkT))
    for i in range(1, len(ebins)):
        E = ebins[i]
        exp_EkT = exp(-E/kT)
        integral_high = E*E*log1p(-exp_EkT) - 2.0*kT*(E*polylog(2,exp_EkT) + kT*polylog(3,exp_EkT))
        flux.append(norm*(integral_high - integral_low))
        integral_low = integral_high
    return flux

def xsbbrad(kT,ebins): xsflux=[];callModFunc('bbodyrad',ebins,[kT],xsflux,[],1,'');return np.asarray(xsflux)
def xsbb(kT,ebins): xsflux=[];callModFunc('bbody',ebins,[kT],xsflux,[],1,'');return np.asarray(xsflux)

if __name__ == '__main__':
    ebins = np.geomspace(1e-3, 1e3, 100)
    for kT in np.geomspace(1, 1e5, 20):
        stdflux = bb(kT, ebins)
        flux = bbody(kT, ebins)
        # xsflux = xsbbrad(kT,ebins)
        fig, axes = plt.subplots(2, 1, sharex=1)
        fig.subplots_adjust(hspace=0.0)
        axes[0].step(ebins, np.append(stdflux, stdflux[-1]), color='g')
        resd = [float(i) for i in np.abs(stdflux-flux)/stdflux]
        # xsresd = [float(i) for i in np.abs(stdflux-xsflux)/stdflux]
        axes[1].step(ebins, np.append(resd, resd[-1]), where='post')
        # axes[1].step(ebins, np.append(xsresd, xsresd[-1]), where='post')
        axes[1].axhline(1e-8, ls=':', color='g')
        axes[0].loglog()
        axes[1].loglog()
        axes[0].set_title(kT)