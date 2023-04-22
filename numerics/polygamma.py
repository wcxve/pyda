# =============================================================================
# Codes below are copied from:
# https://gist.github.com/tomblaze/c660f63ea194e2291d637335c9b58b39
# =============================================================================
#Quick write-up of implementation of polygamma function in numba.
#Doesn't work with complex numbers, only works for orders < 5 (trigamma, tetragamma). Higher orders could be added by adding a few more terms in the cot-deriv.
#This whole file is 100% completely based off of the SpecialFunctions.jl code in julia, credit to authors there (esp. @stevengj)
#Requires numba (obviously), scipy for testing.

from numba import vectorize, njit, jit
import ctypes
from numba.extending import get_cython_function_address
import numpy as np

gamma_addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1gamma") #note: not sure what 0gamma is but "__pyx_fuse_0gamma" doesn't give right answer.
gamma_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
gamma_fn = gamma_functype(gamma_addr)

@vectorize('float64(float64)')
def vec_gamma(x):
	return gamma_fn(x)

@njit
def nb_gamma(x):
	return vec_gamma(x)

@jit(nopython=True)
def zeta_fn(s, z):
	zt = 0.0
	x = z
	m = s - 1

	cutoff = 7 + m #assume no imaginary component

	if x < cutoff:
		xf = np.floor(x)
		nx = xf
		n = np.ceil(cutoff - nx)
		minus_s = -s

		if nx < 0:
			minus_z = -z
			zt += np.power(minus_z, minus_s)
			if xf != z:
				zt += np.power(z - nx, minus_s)
			if s > 0:
				for v in range(-nx -1, 0, -1):
					z_0 = zt
					zt += np.power(minus_z - v, minus_s)
					if zt == z_0:
						break
			else:
				for v in range(1, -nx):
					z_0 = zt
					zt += np.power(minus_z - v, minus_s)
					if zt == z_0:
						break
		else:
			zt += np.power(z, minus_s)

	if s > 0:
		for v in range(max(1, int(1 - nx)), int(n)):
			z_0 = zt
			zt += np.power(z + v, minus_s)
			if zt == z_0:
				break
	else:
		for v in range(n - 1, max(0, -nx), -1):
			z_0 = zt
			zt += np.power(z + v, minus_s)
			if zt == z_0:
				break
	z += n

	t = 1.0 / z
	w = t ** m
	zt += w * ((1.0 / m) + 0.5 * t)

	t = t * t

	#obtained by expanding the macro
	horner_term = ((m + 1) * (0.08333333333333333 + t * ((0.16666666666666666 * (m + 3) * (m + 2)) * (-0.008333333333333333 + t * ((0.05 * (m + 5) * (m + 4)) * (0.003968253968253968 + t * ((0.023809523809523808 * (m + 7) * (m + 6)) * (-0.004166666666666667 + t * ((0.013888888888888888 * (m + 9) * (m + 8)) * (0.007575757575757576 + t * ((0.00909090909090909 * (m + 11) * (m + 10)) * (-0.021092796092796094 + t * ((0.00641025641025641 * (m + 13) * (m + 12)) * (0.08333333333333333 + t * ((0.004761904761904762 * (m + 15) * (m + 14)) * (-0.4432598039215686 + t * ((m + 17) * (m + 16) * 0.01122777327305191)))))))))))))))))

	zt += w * t * horner_term
	return zt

@vectorize('float64(int64, float64)')
def vec_zeta(s, z):
	return zeta_fn(s, z)

@njit
def nb_zeta(s, z):
	return vec_zeta(s, z)

@jit(nopython=True)
def signflip(m, z):
	if m % 2 == 0:
		return z
	else:
		return -z

@jit(nopython=True)
def cot(x):
	return 1.0 / np.tan(x)

@jit(nopython=True)
def cotderiv(m, z):

	if m == 1:
		q = np.array([1., 1.])
	elif m == 2:
		q = np.array([1., 1.])
	elif m == 3:
		q = np.array([0.33333333, 1.33333333, 1.        ])
	elif m == 4:
		q = np.array([0.66666667, 1.66666667, 1.        ]) #hard code a few of these.

	if m == 0.0:
		return np.pi * (1.0 / cot(np.pi * z))

	if m < 5:
		x = cot(np.pi * z)
		y = x * x
		s = q[0] + q[1] * y
		t = y
		for i in range(2, len(q)):
			t *= y
			s += q[i] * t
		if m % 2 == 1:
			return (np.pi ** (m + 1)) * s
		else:
			return (np.pi ** (m + 1)) * (x * s)

@jit(nopython=True)
def polygamma_fn(order, z): #don't call this for digamma
	s = order + 1
	if z <= 0:
		return (nb_zeta(s, 1 - z) + signflip(order, cotderiv(order, z))) * -nb_gamma(s)
	else:
		return signflip(order, nb_zeta(s, z) * (-nb_gamma(s)))

@vectorize('float64(int64, float64)')
def vec_polygamma(order, z):
	return polygamma_fn(order, z)

@njit
def polygamma(order, z):
	return vec_polygamma(order, z)

if __name__ == "__main__":
	#Very basic test code.
	x = np.random.randn(5, 2)

	print(polygamma(3, x))

	import scipy
	import scipy.special

	print(scipy.special.polygamma(3, x))
