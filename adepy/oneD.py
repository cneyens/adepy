from scipy.optimize import brentq
import numpy as np
from numba import njit, vectorize
from adepy._helpers import _erfc_nb as erfc

# @njit
def _bserie_finite1(betas, x, t, Pe, L, D, lamb):
    bs = 0.0
    if lamb == 0:
        for b in betas:
            bs += b * np.sin(b * x / L) * (b**2 + (Pe / 2)**2) * np.exp(-b**2 * D * t / L**2) /\
                    ((b**2 + (Pe / 2)**2 + Pe / 2) * (b**2 + (Pe / 2)**2 + lamb / D * L**2))
    else:
        for b in betas:
            bs += b * np.sin(b * x / L) * np.exp(-b**2 * D * t / L**2) / (b**2 + (Pe / 2)**2 + Pe / 2)
    return bs

def finite1(c0, x, t, v, al, L, Dm=0, lamb=0, R=1.0, nterm=1000):
    
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)
    
    if len(x) > 1 and len(t) > 1:
        raise ValueError('Either x or t should have length 1')
    
    D = al * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    D = D / R

    Pe = v * L / D
    
    # find roots
    def betaf(b):
        return b * 1 / np.tan(b) + Pe / 2

    intervals = [np.pi * i for i  in range(nterm)]
    betas = []
    for i in range(len(intervals) - 1):
        mi = intervals[i] + 1e-10
        ma = intervals[i + 1] - 1e-10
        betas.append(brentq(betaf, mi, ma))

    # calculate infinite sum up to nterm terms
    # bseries_vec = np.vectorize(_bserie_finite1)
    series = _bserie_finite1(betas, x, t, Pe, L, D, lamb)

    if lamb == 0:
        term0 = 1.0
    else:
        u = np.sqrt(v**2 + 4 * lamb * D)
        term0 = (np.exp((v - u) * x / (2 * D)) + (u - v) / (u + v) * np.exp((v + u) * x / (2 * D) - u * L / D)) /\
                    (1 + (u - v) / (u + v) * np.exp(-u * L / D))  
        
    term1 = -2 * np.exp(v * x / (2 * D) - v**2 * t / (4 * D) - lamb * t)
           
    return c0 * (term0 + term1 * series)

# @njit
def _bserie_finite3(betas, x, t, Pe, L, D, lamb):
    bs = 0.0
    for b in betas:
        bs += b * (b * np.cos(b * x / L) + (Pe / 2) * np.sin(b * x / L)) / (b**2 + (Pe / 2)**2 + Pe) *\
                np.exp(-b**2 * D * t / L**2) / (b**2 + (Pe / 2)**2 + lamb * L**2 / D)
    return bs

def finite3(c0, x, t, v, al, L, Dm=0, lamb=0, R=1.0, nterm=1000):
    # https://github.com/BYL4746/columntracer/blob/main/columntracer.py

    x = np.atleast_1d(x)
    t = np.atleast_1d(t)
    
    if len(x) > 1 and len(t) > 1:
        raise ValueError('Either x or t should have length 1')
    
    D = al * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    D = D / R

    Pe = v * L / D
    
    # find roots
    def betaf(b):
        return b * 1 / np.tan(b) - b**2 / Pe + Pe / 4

    intervals = [np.pi * i for i in range(nterm)]
    betas = []
    for i in range(len(intervals) - 1):
        mi = intervals[i] + 1e-10
        ma = intervals[i + 1] - 1e-10
        betas.append(brentq(betaf, mi, ma))

    # calculate infinite sum up to nterm terms
    # bseries_vec = np.vectorize(_bserie_finite3)
    series = _bserie_finite3(betas, x, t, Pe, L, D, lamb)

    if lamb == 0:
        term0 = 1.0
    else:
        u = np.sqrt(v**2 + 4 * lamb * D)
        term0 = (np.exp((v - u) * x / (2 * D)) + (u - v) / (u + v) * np.exp((v + u) * x / (2 * D) - u * L / D)) /\
                ((u + v) / (2 * v) - (u - v)**2 / (2 * v * (u + v)) * np.exp(-u * L / D)) 
        
    term1 = -2 * Pe * np.exp(v * x / (2 * D) - v**2 * t / (4 * D) - lamb * t)
           
    return c0 * (term0 + term1 * series)

def seminf1(c0, x, t, v, al, Dm=0, lamb=0, R=1.0):
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)

    D = al * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    D = D / R

    u = np.sqrt(v**2 + 4 * lamb * D)
    term = np.exp(x * (v - u) / (2 * D)) * erfc((x - u * t) / (2 * np.sqrt(D * t))) + \
            np.exp(x * (v + u) / (2 * D)) * erfc((x + u * t) / (2 * np.sqrt(D * t)))

    return c0 * 0.5 * term

def seminf3(c0, x, t, v, al, Dm=0, lamb=0, R=1.0):
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)

    D = al * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    D = D / R
    
    u = np.sqrt(v**2 + 4 * lamb * D)
    if lamb == 0:
        term = 0.5 * erfc((x - v * t) / (2 * np.sqrt(D * t))) + np.sqrt(t * v**2 / (np.pi * D)) * np.exp(-(x - v * t)**2 / (4 * D * t)) - \
            0.5 * (1 + v * x / D + t * v**2 / D) * np.exp(v * x / D) * erfc((x + v * t) / (2 * np.sqrt(D * t)))
        term0 = 1.0
    else:
        term = 2 * np.exp(x * v / D - lamb * t) * erfc((x + v * t) / (2 * np.sqrt(D * t))) +\
            (u / v - 1) * np.exp(x * (v - u) / (2 * D)) * erfc((x - u * t) / (2 * np.sqrt(D * t))) -\
            (u / v - 1) * np.exp(x * (v + u) / (2 * D)) * erfc((x + u * t) / (2 * np.sqrt(D * t)))
        term0 = v**2 / (4 * lamb * D)

    return c0 * term0 * term
                                 