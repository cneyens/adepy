import numpy as np
from numba import njit
from adepy._helpers import _erfc_nb as erfc
from adepy._helpers import _integrate as integrate

def point3(c0, x, y, z, t, v, n, al, ah, av, Q, xc, yc, zc, Dm=0, lamb=0, R=1.0):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    t = np.atleast_1d(t)
    
    Dx = al * v + Dm
    Dy = ah * v + Dm
    Dz = av * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    Dx = Dx / R
    Dy = Dy / R
    Dz = Dz / R
    Q = Q / R

    beta = np.sqrt(v**2 + 4 * Dx * lamb)
    gamma = np.sqrt((x - xc)**2 + Dx * (y - yc)**2 / Dy + Dx * (z - zc)**2 / Dz)
    
    a = np.exp(v * (x - xc) / (2 * Dx)) / (8 * n * np.pi * gamma * np.sqrt(Dy * Dz))
    b = np.exp(gamma * beta / (2 * Dx)) * erfc((gamma + beta * t) / (2 * np.sqrt(Dx * t))) + \
            np.exp(- gamma * beta / (2 * Dx)) * erfc((gamma - beta * t) / (2 * np.sqrt(Dx * t)))
    
    return c0 * Q * a * b

# @njit
def _series_patchf(x, y, z, t, v, Dx, Dy, Dz, w, h, y1, y2, z1, z2, lamb, nterm):

    if len(t) > 1:
        series = np.zeros_like(t, dtype=np.float64)
    else:
        series = np.zeros_like(x, dtype=np.float64)

    for m in range(nterm):
        zeta = m * np.pi / h
        
        if m == 0:
            Om = (z2 - z1) / h
        else:
            Om = (np.sin(zeta * z2) - np.sin(zeta * z1)) / (m * np.pi)
            
        for n in range(nterm):
            eta = n * np.pi / w
            beta = np.sqrt(v**2 + 4 * Dx * (Dy * eta**2 + Dz * zeta**2 + lamb))

            if m == 0 and n == 0:
                Lmn = 0.5
            elif m > 0 and n > 0:
                Lmn = 2.0
            else:
                Lmn = 1.0

            if n == 0:
                Pn = (y2 - y1) / w
            else:
                Pn = (np.sin(eta * y2) - np.sin(eta * y1)) / (n * np.pi)
            
            term = np.exp((x * (v - beta) / (2 * Dx))) * erfc((x - beta * t) / (2 * np.sqrt(Dx * t))) +\
                    np.exp((x * (v + beta)) / (2 * Dx)) * erfc((x + beta * t) / (2 * np.sqrt(Dx * t)))

            add = Lmn * Om * Pn * np.cos(zeta * z) * np.cos(eta * y) * term

            add = np.where(np.isneginf(add), 0.0, add)
            add = np.where(np.isnan(add), 0.0, add)
            series += add
    
    return series

def patchf(c0, x, y, z, t, v, al, ah, av, w, h, y1, y2, z1, z2, Dm=0, lamb=0, R=1.0, nterm=50):
    
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    t = np.atleast_1d(t)

    Dx = al * v + Dm
    Dy = ah * v + Dm
    Dz = av * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    Dx = Dx / R
    Dy = Dy / R
    Dz = Dz / R

    if len(t) > 1 and (len(x) > 1 or len(y) > 1 or len(z) > 1):
        raise ValueError('If multiple values for t are specified, only one x, y and z value are allowed')

    series = _series_patchf(x, y, z, t, v, Dx, Dy, Dz, w, h, y1, y2, z1, z2, lamb, nterm)

    return c0 * series

@njit
def _integrand_patchi(tau, x, y, z, v, Dx, Dy, Dz, y1, y2, z1, z2, lamb):
    ig = 1 / tau**3 * np.exp(-(v**2 / (4 * Dx) + lamb) * tau**4 - x**2 / (4 * Dx * tau**4)) *\
                (erfc((y1 - y) / (2 * tau**2 * np.sqrt(Dy))) - erfc((y2 - y) / (2 * tau**2 * np.sqrt(Dy)))) *\
                (erfc((z1 - z) / (2 * tau**2 * np.sqrt(Dz))) - erfc((z2 - z) / (2 * tau**2 * np.sqrt(Dz))))
    
    return ig

def patchi(c0, x, y, z, t, v, ah, al, av, y1, y2, z1, z2, Dm=0, lamb=0, R=1.0, order=100):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    t = np.atleast_1d(t)

    Dx = al * v + Dm
    Dy = ah * v + Dm
    Dz = av * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    Dx = Dx / R
    Dy = Dy / R
    Dz = Dz / R
    
    term = integrate(_integrand_patchi, t**(1/4), x, y, z, v, Dx, Dy, Dz, y1, y2, z1, z2, lamb, order=order, method='legendre')

    term0 = x * np.exp(v * x / (2 * Dx)) / (2 * np.sqrt(np.pi * Dx))

    return c0 * term0 * term
    