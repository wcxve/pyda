
import numpy as np
from numba import njit

@njit
def xsbbrd(kT, ebins):
    N = len(ebins)
    flux = np.empty(N-1)

    el = ebins[0]
    x = el/kT
    if x <= 1.0e-4:
        nl = el*kT # limit_{el/kT->1} el*el/(exp(el/kT)-1) = el*kT
    elif x > 60.0:
        flux[:] = 0.0
        return flux
    else:
        nl = el*el/(np.exp(x) - 1)

    norm = 1.0344e-3 / 2.0 # norm of 2-point approximation to integral

    for i in range(N-1):
        eh = ebins[i+1]
        x = eh/kT
        if x <= 1.0e-4:
            nh = eh*kT
        elif x > 60.0:
            flux[i:] = 0.0
            break
        else:
            nh = eh*eh/(np.exp(x)-1.0)
        flux[i] = norm * (nl + nh) * (eh - el)
        el = eh
        nl = nh

    return flux
