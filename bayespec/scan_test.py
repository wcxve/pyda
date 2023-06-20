import pytensor.tensor as pt
from pytensor.ifelse import ifelse
from pytensor import function
def wstat_background(s, n_on, n_off, a):
    c = a*(n_on + n_off) - (a + 1)*s
    d = pt.sqrt(c*c + 4*a*(a + 1)*n_off*s)
    b = pt.switch(
            pt.eq(n_on, 0),
            n_off/(1 + a),
            pt.switch(
                pt.eq(n_off, 0),
                pt.switch(
                    pt.le(s, a/(a + 1)*n_on),
                    n_on/(1 + a) - s/a,
                    0.0
                ),
                (c + d) / (2*a*(a + 1))
            )
        )
    return b

def wstat_background_scalar(s, n_on, n_off, a):
    c = a*(n_on + n_off) - (a + 1)*s
    d = pt.sqrt(c*c + 4*a*(a + 1)*n_off*s)
    b = ifelse(
            pt.eq(n_on, 0),
            n_off/(1 + a),
            ifelse(
                pt.eq(n_off, 0),
                ifelse(
                    pt.le(s, a/(a + 1)*n_on),
                    n_on/(1 + a) - s/a,
                    pt.constant(0, dtype=float)
                ),
                (c + d) / (2*a*(a + 1))
            )
        )
    # if n_on == 0.0:
    #     b = n_off/(1+a)
    # else:
    #     if n_off == 0.0:
    #         b = ifelse(
    #             pt.le(s, a/(a + 1)*n_on),
    #             n_on/(1 + a) - s/a,
    #             pt.constant(0, dtype=float)
    #         )
    #     else:
    #         c = a * (n_on + n_off) - (a + 1) * s
    #         d = pt.sqrt(c * c + 4 * a * (a + 1) * n_off * s)
    #         b = (c + d) / (2*a*(a + 1))
    return b

from numba import njit
@njit('float64[::1](float64[::1], float64[::1], float64[::1], float64)')
def b0(s, n_on, n_off, a):
    N = len(s)
    b = np.empty(N)
    v1 = a+1
    v2 = a/v1
    v3 = 4*a
    v4 = 2*a*v1
    for i in range(N):
        s_i = s[i]
        n_on_i = n_on[i]
        n_off_i = n_off[i]
        if n_on_i == 0.0:
            b[i] = n_off_i/(1+a)
        else:
            if n_off_i == 0.0:
                if s_i <= v2*n_on_i:
                    b[i] = n_on_i/v1 - s_i/a
                else:
                    b[i] = 0.0
            else:
                c = a * (n_on_i + n_off_i) - v1 * s_i
                d = np.sqrt(c * c + v3 * v1 * n_off_i * s_i)
                b[i] = (c + d) / v4
    return b

def b_profile(s, n_on, n_off, a):
    c = a * (n_on + n_off) - (a + 1) * s
    d = pt.sqrt(c * c + 4 * a * (a + 1) * n_off * s)
    b = (c + d) / (2*a*(a + 1))
    return b


from pytensor import scan
def wstat_background_scan(s, n_on, n_off, a):
    n_on = pt.constant(n_on)
    n_off = pt.constant(n_off)
    a = pt.constant(a, dtype=float)
    b, _ = scan(fn=wstat_background_scalar,
                sequences=[s, n_on, n_off],
                non_sequences=[a],
                strict=True)
    return b



if __name__ == '__main__':
    import numpy as np
    import pytensor
    # pytensor.config.mode = 'JAX'
    # pytensor.config.floatX = 'float64'

    rng = np.random.default_rng(42)
    n_on = rng.poisson(30+50, 300).astype(float)
    n_off = rng.poisson(50, 300).astype(float)
    # n_off[:100] = 0
    a = 1.0
    s = pt.tensor(dtype='float64', shape=(300,))
    s_value = rng.normal(30, 5, 300)
    b = function([s], b_profile(s, n_on, n_off, a))
    b1 = function([s], wstat_background(s, n_on, n_off, a))
    b2 = function([s], wstat_background_scan(s, n_on, n_off, a))

    E = np.geomspace(0.1, 10, 1000)
    PhoIndex = pt.dscalar()
    alpha = 1.0-PhoIndex
    integral = pt.pow(E, alpha)/alpha
    pl = function([PhoIndex], integral[1:]-integral[:-1])

    @njit('float64[::1](float64, float64[::1])')
    def pl2(PhoIndex, E):
        n = len(E)
        NE = np.empty(n)
        a = 1.0 - PhoIndex
        for i in range(n):
            NE[i] = E[i]**a/a
        return NE[1:] - NE[:-1]

    from pyda.bayespec import Powerlaw
    pl3 = function([PhoIndex], Powerlaw(PhoIndex)(E))
