"""
Created at 04:00:59 on 2023-05-06

@author: Wang-Chen Xue <https://orcid.org/0000-0001-8664-5085>
"""
import mpmath as mp
def dalpha(a, b, ebins):
    a = mp.mpf(a)
    b = mp.mpf(b)
    da = []
    for x in ebins:
        x = mp.mpf(x)
        x_b = x/b
        da.append(x ** (-a) * (x * mp.expint(a, x_b) * mp.log(x) + b * mp.meijerg(
            [[], [1 + a, 1 + a]], [[1, a, a], []], x_b)))
    da = [da[i+1] - da[i] for i in range(len(ebins)-1)]
    return da

if __name__ == '__main__':
    from pyda.numerics.specfun import cutoffpl_dalpha
    import numpy as np
    ebins = np.geomspace(1,10,101)
    da = cutoffpl_dalpha(2.01, 12.0, ebins)
    da_mp = dalpha(2.01, 12.0, ebins)