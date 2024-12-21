from scipy.special import erfc
from scipy.integrate import quad
import numpy as np

def point2(c0, x, y, t, v, Dx, Dy, n, Q, xc, yc, lamb=0):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)

    if len(t) > 1 and (len(x) > 1 or len(y) > 1):
        raise ValueError('If multiple values for t are specified, only one x and y value are allowed')

    def integrate(t, x, y):
        def integrand(tau, x, y):
            return 1 / tau * np.exp(-(v**2 / (4 * Dx) + lamb) * tau - (x - xc)**2 / (4 * Dx * tau) - (y - yc)**2 / (4 * Dy * tau))
    
        F = quad(integrand, 0, t, args=(x, y), full_output=1)[0]
        return F
    integrate_vec = np.vectorize(integrate)

    term = integrate_vec(t, x, y)
    term0 = Q / (4 * n * np.pi * np.sqrt(Dx * Dy)) * np.exp(v * (x - xc) / (2 * Dx))
    
    return c0 * term0 * term

def stripf(c0, x, y, t, v, Dx, Dy, y2, y1, w, lamb=0, nterm=100):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)

    if lamb != 0 and Dy != 0:
        raise ValueError('Either Dy or lamb should be zero')

    if len(t) > 1 and (len(x) > 1 or len(y) > 1):
        raise ValueError('If multiple values for t are specified, only one x and y value are allowed')

    if len(t) > 1:
        series = np.zeros_like(t, dtype=np.float64)
    else:
        series = np.zeros_like(x, dtype=np.float64)
    
    for n in range(nterm):
        eta = n * np.pi / w
        beta = np.sqrt(v**2 + 4 * Dx * (eta**2 * Dy + lamb))
        
        if n == 0:
            Ln = 0.5
            Pn = (y2 - y1) / w
        else:
            Ln = 1
            Pn = (np.sin(eta * y2) - np.sin(eta * y1)) / (n * np.pi)

        term = np.exp((x * (v - beta)) / (2 * Dx)) * erfc((x - beta * t) / (2 * np.sqrt(Dx * t))) +\
                np.exp((x * (v + beta)) / (2 * Dx)) * erfc((x + beta * t) / (2 * np.sqrt(Dx * t)))
        
        add = Ln * Pn * np.cos(eta * y) * term

        add = np.where(np.isneginf(add), 0.0, add)
        add = np.where(np.isnan(add), 0.0, add)
        series += add

    return c0 * series

def stripi(c0, x, y, t, v, Dx, Dy, y2, y1, lamb=0):

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)

    def integrate(t, x, y):
        # error in Wexler, 1992, eq. 91a: denominator of last erfc term should be multiplied by 2. Correct in code below.
        def integrand(tau, x, y):
            ig = (tau**(-3 / 2)) * np.exp(-(v**2 / (4 * Dx) + lamb) * tau - x**2 / (4 * Dx * tau)) *\
                (erfc((y1 - y) / (2 * np.sqrt(Dy * tau))) - erfc((y2 - y) / (2 * np.sqrt(Dy * tau))))
            return ig

        F = quad(integrand, 0, t, args=(x, y), full_output=1)[0]
        return F

    integrate_vec = np.vectorize(integrate)

    term = integrate_vec(t, x, y)
    term0 = x / (4 * np.sqrt(np.pi * Dx)) * np.exp(v * x / (2 * Dx))

    return c0 * term0 * term

def gauss(c0, x, y, t, v, Dx, Dy, yc, sigma, lamb=0):
    
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)

    def integrate(t, x, y):
        def integrand(tau, x, y):
            num = np.exp(-(v**2 / (4 * Dx) + lamb) * tau - x**2 / (4 * Dx * tau) - ((y - yc)**2) / (4 * (Dy * tau + 0.5 * sigma**2)))
            denom = (tau**(3 / 2)) * np.sqrt(Dy * tau + 0.5 * sigma**2)
            return num / denom
    
        F = quad(integrand, 0, t, args=(x, y), full_output=1)[0]
        return F

    integrate_vec = np.vectorize(integrate)

    term = integrate_vec(t, x, y)
    term0 = x * sigma / np.sqrt(8 * np.pi * Dx) * np.exp(v * x / (2 * Dx))

    return c0 * term0 * term