from numba import njit, vectorize
from numba.extending import get_cython_function_address
import ctypes
import numpy as np
from scipy.special import roots_legendre
from scipy.integrate import quad

addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1erfc")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
erfc_fn = functype(addr)


@vectorize("float64(float64)")
def _vec_erfc(x):
    return erfc_fn(x)


@njit
def _erfc_nb(x):
    return _vec_erfc(x)


def _integrate(integrand, t, *args, order=100, method="legendre"):
    if method == "legendre":
        roots, weights = roots_legendre(order)

        def integrate(t, *args):
            roots_adj = roots * (t - 0) / 2 + (0 + t) / 2
            F = integrand(roots_adj, *args).dot(weights) * (t - 0) / 2
            return F

    elif method == "quadrature":

        def integrate(t, *args):
            F = quad(integrand, 0, t, args=args.items)
            return F

    else:
        raise ValueError('Integration method should be "legendre" or "quadrature"')

    integrate_vec = np.vectorize(integrate)
    term = integrate_vec(t, *args)
    return term
