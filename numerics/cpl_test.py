"""
Created at 01:59:11 on 2023-05-13

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""

import numpy as np
import matplotlib.pyplot as plt
from pyda.numerics.specfun import *
from mxspec._pymXspec import callModFunc

from mpmath import hyp1f1, mpf, mp, log10, exp, power

mp.dps = 500
def cpl(a,b,ebins):
    a = mpf(a)

    b_ = mpf(b)
    integral = []
    low = power(ebins[0], 1-a) /(1-a) * exp(-ebins[0]/b_) * hyp1f1(1.0,(2-a),ebins[0]/b_)
    for i in range(1, len(ebins)):
        high = power(ebins[i], 1-a) /(1-a) * exp(-ebins[i]/b_) * hyp1f1(1.0,(2-a),ebins[i]/b_)
        integral.append(high-low)
        low = high
    return integral
def xscpl(a,b,ebins): xflux=[];callModFunc('cutoffpl',ebins,[a, b],xflux,[],1,'');return xflux
if __name__ == '__main__':
    ebins=np.geomspace(1e-3,1000,1001)

    import time
    t0 = time.time()
    np.random.seed(42)
    # plt.figure()
    for a in np.hstack((np.geomspace(1e-3, 80, 20)[:0],
                       -np.geomspace(1e-3, 80, 20)[:0],
                       [-80])):

        a*=np.random.uniform(0.99,1.01)
        b=1
        stdflux = cpl(a,b,ebins)
        fig, axes = plt.subplots(3, 1, sharex=1)
        fig.subplots_adjust(hspace=0.0)
        for f in [cutoffpl][:1]:
            flux = np.asarray(f(a, b, ebins))
            resd = np.abs(flux - stdflux)
            resd = [float(i) for i in resd]
            # plt.step(ebins / b, np.log(np.append(flux, flux[-1])), where='post',label=f'a={a:.2f}')
            axes[0].step(ebins / b, [log10(i) for i in np.append(flux, flux[-1])],
                         where='post')
            with np.errstate(all='ignore'):
                axes[1].step(ebins/b, np.log10(np.append(resd, resd[-1])),where='post')
            ratio = [float(i) for i in np.abs(stdflux-flux)/stdflux]
            axes[2].step(ebins/b, np.log10(np.append(ratio,ratio[-1])),where='post')
        # axes[2].axvline(abs(a))
        # print('%.2e'%stdflux[ratio.argmax()], '%.2e'%flux[ratio.argmax()], '%.2e'%ratio.max())
    # print(time.time()-t0)
    # axes[2].loglog()
        axes[0].set_title(a)
        axes[2].axhline(-8, ls=':', color='g')
        # axes[2].set_ylim(-20,20)
        plt.semilogx()
    # plt.legend()