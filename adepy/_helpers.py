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


def _integrate(integrand, t, *args, t0=0.0, order=100, method="legendre"):
    if method == "legendre":
        roots, weights = roots_legendre(order)

        def integrate(t, *args):
            roots_adj = roots * (t - t0) / 2 + (t0 + t) / 2
            F = integrand(roots_adj, *args).dot(weights) * (t - t0) / 2
            return F

    elif method == "quadrature":

        def integrate(t, *args):
            F = quad(integrand, t0, t, args=args)
            return F[0]

    else:
        raise ValueError('Integration method should be "legendre" or "quadrature"')

    integrate_vec = np.vectorize(integrate)
    term = integrate_vec(t, *args)
    return term


def _dehoog(t, fbar, M=7, relerr=1e-4, **kwargs):
    """De Hoog Laplace inversion. Based on the MPNE1D source code.

    Source: [sspapa_2004]_, [neville_2000]_


    Parameters
    ----------
    t : float
        time
    fbar : callable
        function with the Laplace domain solution. The first argument is `p`, the Laplace transform parameter.
        Additional parameters are supplied through kwargs.
    M : int, optional
        Number of terms used in the inversion, by default 7
    relerr : float, optional
        Maximum relative error for convergence, by default 1e-4

    Returns
    -------
    float
        Inverted time-domain solution as a float

    References
    ----------
    .. [sspapa_2004] MPNE1D, 2004. MPNE1D Analytical Solution: User's Guide, version 4.1. S.S. Papadopulos & Associates, Inc.
    .. [neville_2000] Neville, C.J., Ibaraki, M., Sudicky, E.A., 2000. Solute transport with multiprocess nonequilibrium: a semi-analytical solution approach, Journal of Contaminant Hydrology 44-2, p. 141-159

    """

    alpha = 0.0
    bigt = 0.8 * t
    aterm = alpha - (np.log(relerr) / (2.0 * bigt))
    pi = np.pi
    factor = pi / bigt
    M2 = 2 * M
    z = np.exp(1j * t * factor)
    D = np.zeros(M2 + 1, dtype=complex)
    WORK = np.zeros(M2 + 1, dtype=complex)
    SMALL = 1e-15

    # Initial Pade table
    AOLD = fbar(aterm + 0j, **kwargs) / 2.0
    A = fbar(aterm + 1j * factor, **kwargs)
    D[0], WORK[0] = AOLD[0], 0.0
    if abs(AOLD) < SMALL:
        AOLD = np.atleast_1d(SMALL + 1j * SMALL)
    WORK[1] = A[0] / AOLD[0]
    D[1] = -WORK[1]
    AOLD = A

    for J in range(2, M2 + 1):
        OLD2, OLD1 = WORK[0], WORK[1]
        A = fbar(aterm + 1j * J * factor, **kwargs)
        WORK[0] = 0.0
        if abs(AOLD) < SMALL:
            AOLD = np.atleast_1d(SMALL + 1j * SMALL)
        WORK[1] = A[0] / AOLD[0]
        AOLD = A
        for II in range(2, J + 1):
            OLD3, OLD2, OLD1 = OLD2, OLD1, WORK[II] if II < len(WORK) else 0.0
            if II % 2 == 0:
                WORK[II] = OLD3 + (WORK[II - 1] - OLD2)
            else:
                if abs(OLD2) < SMALL:
                    OLD2 = SMALL + 1j * SMALL
                WORK[II] = OLD3 * (WORK[II - 1] / OLD2)
        D[J] = -WORK[J]

    AOLD2, AOLD1 = D[0], D[0]
    BOLD2, BOLD1 = 1.0 + 0j, 1.0 + D[1] * z
    for J in range(2, M2 + 1):
        A = AOLD1 + D[J] * z * AOLD2
        AOLD2, AOLD1 = AOLD1, A
        B = BOLD1 + D[J] * z * BOLD2
        BOLD2, BOLD1 = BOLD1, B

    result = (A / B).real if abs(B) >= SMALL else 0.0
    return np.exp(aterm * t) * result / bigt
