from scipy.special import erfc
from scipy.integrate import quad
import numpy as np

def point3(c0, x, y, z, t, v, n, Dx, Dy, Dz, Q, xc, yc, zc, lamb=0):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    t = np.atleast_1d(t)
    
    beta = np.sqrt(v**2 + 4 * Dx * lamb)
    gamma = np.sqrt((x - xc)**2 + Dx * (y - yc)**2 / Dy + Dx * (z - zc)**2 / Dz)
    
    a = np.exp(v * (x - xc) / (2 * Dx)) / (8 * n * np.pi * gamma * np.sqrt(Dy * Dz))
    b = np.exp(gamma * beta / (2 * Dx)) * erfc((gamma + beta * t) / (2 * np.sqrt(Dx * t))) + \
            np.exp(- gamma * beta / (2 * Dx)) * erfc((gamma - beta * t) / (2 * np.sqrt(Dx * t)))
    
    return c0 * Q * a * b

def patchf(c0, x, y, z, t, v, Dx, Dy, Dz, w, h, y1, y2, z1, z2, lamb=0, nterm=50):
    
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    t = np.atleast_1d(t)

    if len(t) > 1 and (len(x) > 1 or len(y) > 1 or len(z) > 1):
        raise ValueError('If multiple values for t are specified, only one x, y and z value are allowed')

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

    return c0 * series

def patchi(c0, x, y, z, t, v, Dx, Dy, Dz, y1, y2, z1, z2, lamb=0):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    t = np.atleast_1d(t)

    def integrate(t, x, y, z):
        def integrand(tau, x, y, z):
            zeta = tau
            ig = 1 / zeta**3 * np.exp(-(v**2 / (4 * Dx) + lamb) * zeta**4 - x**2 / (4 * Dx * zeta**4)) *\
                     (erfc((y1 - y) / (2 * zeta**2 * np.sqrt(Dy))) - erfc((y2 - y) / (2 * zeta**2 * np.sqrt(Dy)))) *\
                     (erfc((z1 - z) / (2 * zeta**2 * np.sqrt(Dz))) - erfc((z2 - z) / (2 * zeta**2 * np.sqrt(Dz))))
            
            # ig = (tau**(-3 / 2)) * np.exp(-(v**2 / (4 * Dx) + lamb) * tau - x**2 / (4 * Dx * tau)) *\
            #         (erfc((y1 - y) / (2 * np.sqrt(Dy * tau))) - erfc((y2 - y) / (2 * np.sqrt(Dy * tau)))) *\
            #         (erfc((z1 - z) / (2 * np.sqrt(Dz * tau))) - erfc((z2 - z) / (2 * np.sqrt(Dz * tau))))
            return ig

        F = quad(integrand, 0, t**(1/4), args=(x, y, z), full_output=1)[0]
        # F = quad(integrand, 0, t, args=(x, y, z), full_output=1)[0]
        return F

    integrate_vec = np.vectorize(integrate)
    
    term = integrate_vec(t, x, y, z)
    term0 = x * np.exp(v * x / (2 * Dx)) / (2 * np.sqrt(np.pi * Dx))
    # term0 = x * np.exp(v * x / (2 * Dx)) / (8 * np.sqrt(np.pi * Dx))

    return c0 * term0 * term
    