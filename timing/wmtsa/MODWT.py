"""
This module contains functions to compute the MODWT of a time series
"""

import numpy as np
import numba as nb

from math import sqrt

import DWT

def get_scaling(name):
    """
    Return the coefficients of the MODWT scaling filter

    Input:
        type name = string
        name = Name of the wavelet filter
    Output:
        type g = 1D numpy array
        g = Vector of coefficients of the MODWT scaling filter
    """
    g = DWT.get_scaling(name)
    g = g / sqrt(2.0)
    return g

def get_wavelet(g):
    """
    Return the coefficients of the MODWT wavelet filter

    Input:
        type g = 1D numpy array
        g = Vector of coefficients of the MODWT scaling filter
    Output:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWT wavelet filter
    """
    h = DWT.get_wavelet(g)
    return h

@nb.njit
def get_WV(h, g, j, X):
    """
    Level j of pyramid algorithm.
    Take V_(j-1) and return W_j and V_j

    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the MODWT scaling filter
        type j = integer
        j = Current level of the pyramid algorithm
        type X = 1D numpy array
        X = V_(j-1)
    Output:
        type W = 1D numpy array
        W = W_j
        type V = 1D numpy array
        V = V_j
    """
    # assert (np.shape(h)[0] == np.shape(g)[0]), \
    #     'Wavelet and scaling filters have different lengths'
    N = np.shape(X)[0]
    W = np.zeros(N)
    V = np.zeros(N)
    L = np.shape(h)[0]
    for t in range(N):
        for l in range(L):
            index = (t - (2 ** (j - 1)) * l) % N
            W[t] = W[t] + h[l] * X[index]
            V[t] = V[t] + g[l] * X[index]
    return (W, V)

def get_X(h, g, j, W, V):
    """
    Level j of inverse pyramid algorithm.
    Take W_j and V_j and return V_(j-1)

    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the MODWT scaling filter
        type j = integer
        j = Current level of the pyramid algorithm
        type W = 1D numpy array
        W = W_j
        type V = 1D numpy array
        V = V_j
    Output:
        type X = 1D numpy array
        X = V_(j-1)
    """
    assert (np.shape(W)[0] == np.shape(V)[0]), \
        'Wj and Vj have different lengths'
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
    N = np.shape(W)[0]
    X = np.zeros(N)
    L = np.shape(h)[0]
    for t in range(0, N):
        for l in range(0, L):
            index = (t + (2 ** (j - 1)) * l) % N
            X[t] = X[t] + h[l] * W[index] + g[l] * V[index]
    return X

def pyramid(X, name, J):
    """
    Compute the MODDWT of X up to level J

    Input:
        type X = 1D numpy array
        X = Time series
        type name = string
        name = Name of the MODWT wavelet filter
        type J = integer
        J = Level of partial MODWT
    Output:
        type W = list of 1D numpy arrays (length J)
        W = List of vectors of MODWT wavelet coefficients
        type Vj = 1D numpy array
        Vj = Vector of MODWT scaling coefficients at level J
    """
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    g = get_scaling(name)
    h = get_wavelet(g)
    Vj = X
    W = []
    for j in range(1, (J + 1)):
        (Wj, Vj) = get_WV(h, g, j, Vj)
        W.append(Wj)
    return (W, Vj)

def inv_pyramid(W, Vj, name, J):
    """
    Compute the inverse MODWT of W up to level J

    Input:
        type W = list of 1D numpy arrays (length J)
        W = List of vectors of MODWT wavelet coefficients
        type Vj = 1D numpy array
        Vj = Vector of MODWT scaling coefficients at level J
        type name = string
        name = Name of the MODWT wavelet filter
        type J = integer
        J = Level of partial MODWT
    Output:
        type X = 1D numpy array
        X = Original time series
    """
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    g = get_scaling(name)
    h = get_wavelet(g)
    for j in range(J, 0, -1):
        Vj = get_X(h, g, j, W[j - 1], Vj)
    X = Vj
    return X

def get_DS(X, W, name, J):
    """
    Compute the details and the smooths of the time series
    using the MODWT coefficients

    Input:
        type X = 1D numpy array
        X =  Time series
        type W = list of 1D numpy arrays (length J)
        W = List of vectors of MODWT wavelet coefficients
        type name = string
        name = Name of the MODWT wavelet filter
        type J = integer
        J = Level of partial MODWT
    Output:
        type D = list of 1D numpy arrays (length J)
        D = List of details [D1, D2, ... , DJ]
        type S = list of 1D numpy arrays (length J+1)
        S = List of smooths [S0, S1, S2, ... , SJ]
    """
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    N = np.shape(X)[0]
    # Compute details
    D = []
    for j in range(1, J + 1):
        Wj = []
        if (j > 1):
            for k in range(1, j):
                Wj.append(np.zeros(N))
        Wj.append(W[j - 1])
        Vj = np.zeros(N)
        Dj = inv_pyramid(Wj, Vj, name, j)
        D.append(Dj)
    # Compute smooths
    S = [X]
    for j in range(0, J):
        Sj = S[-1] - D[j]
        S.append(Sj)
    return (D, S)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    Fs = 2**-15
    T = 1
    dt = T*Fs
    t = np.arange(0, T, dt)
    comp1 = np.cos(2*np.pi*200*t)*(t>0.6)
    comp2 = np.cos(2*np.pi*10*t)*(t<0.4)
    trend = np.sin(2*np.pi*1/2*t)
    # wgnNoise = 0.4*np.random.randn(t.size);
    # X = comp1+comp2+trend+wgnNoise;
    X = comp1+comp2+trend
    X = X-X.min()
    X_det = np.random.poisson(X)
    N = t.size
    # %%
    from stingray import Lightcurve, Powerspectrum
    # X = X_det
    X = lc
    ps = Powerspectrum(Lightcurve(t, X), norm="leahy").rebin_log(f=0.5)
    # X = (X-X.mean()) / X.std(ddof=1)
    import pycwt #import cwt1
    wave, scales, freqs, coi, _, _ = pycwt.cwt(X, t[1] - t[0], 1/4, wavelet='morlet')
    # import sys
    # sys.path.append('/Users/xuewc/Library/CloudStorage/OneDrive-个人/Documents/MyWork/DataAnalyzer/')
    # from waveletFunctions import wavelet
    # wave2, period, scale, coi2 = wavelet(X, t[1] - t[0], 1)
    Xvar = X.var(ddof=1)
    Xsum = X.sum()
    X = np.append(X, np.flip(X))
    L = 2
    J0 = round(np.log2(X.size / (L - 1) - 1)) - 1
    J = np.arange(1, J0+1)
    N = X.size
    (W, V) = pyramid(X, 'Haar', J0)


    #%%
    power = np.abs(W)**2
    power_global = power.mean(-1)

    # TODO: Boundary effect should be considered, or use reflection boundary
    # TODO: error calculation, using chi-sqaure distribution of EDOF degree to fit
    power_adj = power * 2**(J+1)[:,None]
    power_adj_global = power_adj.sum(-1)

    # alpha = np.corrcoef(X[:-1], X[1:])[0, 1]
    power_leahy_global = power_adj_global / X.sum()


    # plt.figure()
    # p = power_leahy_global
    dt = ((2 ** (J+1) - 1) * (L - 1) + 1) * (t[1] - t[0])
    # plt.plot(dt, p)
    # fbins = 1 / 2**np.arange(1, J0+2) / (t[1] - t[0])
    # plt.plot(freq,p)
    # plt.plot(freq, power_global)
    # plt.plot(freq, power_adj_global)
    # plt.step(fbins, np.append(p, p[-1]), where='post', c='r')
    plt.plot(dt, power_leahy_global, 'c.:')

    # plt.xlabel('Frequency [Hz]')
    # plt.xlabel('$\delta t$ [s]')
    # plt.loglog()
    # plt.axhline(2, ls=':', c='#00FF00')
    # plt.xlim(1e2, 5e4)
    # plt.ylim(1.8, 2.2)
    # plt.plot(1/ps.freq, ps.power, 'b.:')
    # plt.axhline(2, c='#00FF00',ls=':')
    # plt.title('PSD')
    # plt.xlabel('Frequency [Hz]')

    # plt.axhline(2, ls=':', c='#00FF00')
    # plt.xlim(1e2, 5e4)
    # plt.ylim(1.8, 2.2)
    #%%
    from scipy.stats import chi2
    # alpha=0
    cwt_power = np.abs(wave)**2# / scales[:,None]
    # plt.plot(1/freqs, cwt_power.sum(-1) / Xsum * 2, '.-')
    # cwt_power2 = (alpha + np.abs(wave2)**2)*2
    # plt.plot(period, cwt_power2.mean(-1), '.-')
    # plt.plot(1/ps.freq,ps.power, 'r.:')
    # plt.axhline(2, c='#00FF00', ls=':')
    plt.loglog()
    ylim = plt.gca().get_ylim()

    na = t.size - np.sum(freqs[:,None]<=1/coi, axis=1) / 2
    dofmin = [2, 1][0]
    gamma = [2.32, 1.43][0]
    edof = dofmin*np.sqrt(1 + (na * (t[1] - t[0]) / (gamma * scales))**2)

    signif = 0.99
    alpha = 1-(signif)#**(1/freqs.size)
    upper_bound = 2*(edof/chi2.ppf(alpha, edof))# / scales
    lower_bound = 2*(edof/chi2.ppf(1-alpha, edof)) #/ scales
    median = 2*(edof/chi2.ppf(0.5, edof))#/ scales
    # plt.plot(1/freqs, median, 'k--')
    # plt.fill_between(1/freqs, lower_bound, upper_bound,
                       # alpha=0.5)
    plt.xlabel('$\delta t$ [s]')
    plt.ylabel('Power')
    # plt.xlim(left=6e-5)
    # plt.ylim(cwt_power.mean(-1).min()/2, cwt_power.mean(-1)[-1]*2)
    mvt=min(1/freqs[cwt_power.sum(-1) / Xsum * 2>upper_bound])
    v = f'{mvt:.2e}'.split('e')[0]
    s = int(f'{mvt:.2e}'.split('e')[1])
    plt.axvline(mvt, c='r', ls=':', label=r'MVT$=%s\times10^{%s}$ s'%(v,s))
    # plt.gca().set_aspect('equal')
    plt.legend()
    # plt.axhline(2, ls=':', c='#00FF00')
    plt.show()
    #%%
    # import matplotlib.ticker as ticker
    # plt.figure()
    # p=power_adj
    # CS = plt.contourf(t-131903046.67, 1/2**np.arange(1, L+1)/np.diff(t)[0],
    #                   p,cmap='jet',
    #                   levels=np.logspace(-1,np.log10(p.max()),20),
    #                   locator=ticker.LogLocator())
    # plt.yscale('log')
    # #%%
    # plt.figure()
    # from scipy.ndimage import gaussian_filter
    # # plt.imshow(power, extent=[0, 1, 1, L], aspect='auto',
    # #            vmin=-power.max(), vmax=power.max())
    # fbins = 1/2**np.arange(0.5,L+0.5+1)/np.diff(t)[0]
    # tbins = np.linspace(0, T, t.size+1)
    # tt, ff = np.meshgrid(tbins, fbins)
    # p=power_adj
    # img = gaussian_filter(p, 0.)
    # pcm = plt.pcolormesh(tt, ff, img, norm='log',
    #                      vmin=10**-(np.log10(img.max()))*2)
    # plt.colorbar()
    # plt.semilogy()
    # plt.show()

    #%%
    # X_rep = inv_pyramid(W, V, 'Haar', L)
    # (D, S) = get_DS(X, W, 'Haar', L)
    # MRA = np.row_stack((D, S[-1]))
    #
    # fig, axes = plt.subplots(len(MRA)+1, 1, sharex=True)
    # fig.subplots_adjust(hspace=0, wspace=0)
    # fig.align_ylabels(axes)
    # dt = 1.0 / Fs
    # t = dt * np.arange(N)
    # axes[0].plot(t, X, 'k', label='X')
    # axes[0].set_xlim(np.min(t), np.max(t))
    # axes[0].set_ylabel('X', rotation=0)
    # labels = [f'D{i}' for i in range(1, L+1)] + [f'S{L}']
    # Lj = [(2 ** (j + 1) - 1) * (L - 1) + 1 for j in range(L)]
    # Lj = Lj + Lj[-1:]
    # Lj = np.array(Lj)
    # coi = np.column_stack((dt * (Lj - 2), dt * (N - Lj + 1)))
    # for j in range(L+1):
    #     axes[j+1].plot(t, MRA[j], 'k')
    #     axes[j+1].axvspan(t.min(), coi[j,0], color='red')
    #     axes[j+1].axvspan(coi[j,1], t.max(), color='red')
    #     axes[j+1].set_ylabel(labels[j], rotation=0)
