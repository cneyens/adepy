from scipy.optimize import brentq
import numpy as np
from adepy._helpers import _erfc_nb as erfc
from adepy._helpers import _integrate as integrate
from numba import njit


# @njit
def _bserie_finite1(betas, x, t, Pe, L, D, lamb):
    # TODO check for series convergence
    bs = 0.0

    if lamb == 0:
        for b in betas:
            bs += (
                b
                * np.sin(b * x / L)
                * (b**2 + (Pe / 2) ** 2)
                * np.exp(-(b**2) * D * t / L**2)
                / (
                    (b**2 + (Pe / 2) ** 2 + Pe / 2)
                    * (b**2 + (Pe / 2) ** 2 + lamb / D * L**2)
                )
            )
    else:
        for b in betas:
            bs += (
                b
                * np.sin(b * x / L)
                * np.exp(-(b**2) * D * t / L**2)
                / (b**2 + (Pe / 2) ** 2 + Pe / 2)
            )
    return bs


def finite1(c0, x, t, v, al, L, Dm=0.0, lamb=0.0, R=1.0, nterm=1000):
    """Compute the 1D concentration field of a dissolved solute from a constant-concentration inlet source in
    a finite system with uniform background flow.

    Source: [wexler_1992]_ - FINITE (1) algorithm (equations 44-47).

    The one-dimensional advection-dispersion equation is solved for concentration at specified `x` location(s) and
    output time(s) `t`. A finite system with uniform background flow in the x-direction has a constant-concentration source boundary
    at the inlet. The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed
    in time and space.

    If multiple `x` values are specified, only one `t` can be supplied, and vice versa.

    The solution contains an infinite series summation. A maximum number of terms `nterm` is used. At early times near the source,
    the algorithm may have trouble converging.

    Parameters
    ----------
    c0 : float
        Source concentration [M/L**3]
    x : float or 1D of floats
        x-location(s) to compute output at [L].
    t : float or 1D of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    al : float
        Longitudinal dispersivity [L].
    L : float
        System length along the x-direction [L].
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    lamb : float, optional
        First-order decay rate [1/T], defaults to 0 (no decay).
    R : float, optional
        Retardation coefficient [-]; defaults to 1 (no retardation).
    nterm : integer, optional
        Maximum number of terms used in the series summation. Defaults to 1000.

    Returns
    -------
    ndarray
        Numpy array with computed concentrations at location(s) `x` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
    x = np.atleast_1d(x).astype(np.float64)
    t = np.atleast_1d(t)

    if len(x) > 1 and len(t) > 1:
        raise ValueError("Either x or t should have length 1")

    x[x > L] = np.nan  # set values outside finite column to NA
    D = al * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    D = D / R

    Pe = v * L / D

    # find roots
    def betaf(b):
        return b * 1 / np.tan(b) + Pe / 2

    intervals = [np.pi * i for i in range(nterm)]
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
        term0 = (
            np.exp((v - u) * x / (2 * D))
            + (u - v) / (u + v) * np.exp((v + u) * x / (2 * D) - u * L / D)
        ) / (1 + (u - v) / (u + v) * np.exp(-u * L / D))

    term1 = -2 * np.exp(v * x / (2 * D) - v**2 * t / (4 * D) - lamb * t)

    return c0 * (term0 + term1 * series)


# @njit
def _bserie_finite3(betas, x, t, Pe, L, D, lamb):
    bs = 0.0
    for b in betas:
        bs += (
            b
            * (b * np.cos(b * x / L) + (Pe / 2) * np.sin(b * x / L))
            / (b**2 + (Pe / 2) ** 2 + Pe)
            * np.exp(-(b**2) * D * t / L**2)
            / (b**2 + (Pe / 2) ** 2 + lamb * L**2 / D)
        )
    return bs


def finite3(c0, x, t, v, al, L, Dm=0.0, lamb=0.0, R=1.0, nterm=1000):
    """Compute the 1D concentration field of a dissolved solute from a Cauchy-type inlet source in
    a finite system with uniform background flow.

    Source: [wexler_1992]_ - FINITE (3) algorithm (equations 52-54).

    The one-dimensional advection-dispersion equation is solved for concentration at specified `x` location(s) and
    output time(s) `t`. A finite system with uniform background flow in the x-direction has a Cauchy-type source boundary
    at the inlet where water with specified concentration `c0` is flowing into the system with the background flow.
    The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed
    in time and space.

    If multiple `x` values are specified, only one `t` can be supplied, and vice versa.

    The solution contains an infinite series summation. A maximum number of terms `nterm` is used.

    Parameters
    ----------
    c0 : float
        Source concentration [M/L**3]
    x : float or 1D of floats
        x-location(s) to compute output at [L].
    t : float or 1D of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    al : float
        Longitudinal dispersivity [L].
    L : float
        System length along the x-direction [L].
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    lamb : float, optional
        First-order decay rate [1/T], defaults to 0 (no decay).
    R : float, optional
        Retardation coefficient [-]; defaults to 1 (no retardation).
    nterm : integer, optional
        Maximum number of terms used in the series summation. Defaults to 1000.

    Returns
    -------
    ndarray
        Numpy array with computed concentrations at location(s) `x` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
    x = np.atleast_1d(x).astype(np.float64)
    t = np.atleast_1d(t)

    if len(x) > 1 and len(t) > 1:
        raise ValueError("Either x or t should have length 1")

    x[x > L] = np.nan  # set values outside finite column to NA
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
        term0 = (
            np.exp((v - u) * x / (2 * D))
            + (u - v) / (u + v) * np.exp((v + u) * x / (2 * D) - u * L / D)
        ) / ((u + v) / (2 * v) - (u - v) ** 2 / (2 * v * (u + v)) * np.exp(-u * L / D))

    term1 = -2 * Pe * np.exp(v * x / (2 * D) - v**2 * t / (4 * D) - lamb * t)

    return c0 * (term0 + term1 * series)


def seminf1(c0, x, t, v, al, Dm=0.0, lamb=0.0, R=1.0):
    """Compute the 1D concentration field of a dissolved solute from a constant-concentration inlet source in
    a semi-finite system with uniform background flow.

    Source: [wexler_1992]_ - SEMINF (1) algorithm (equation 60).

    The one-dimensional advection-dispersion equation is solved for concentration at specified `x` location(s) and
    output time(s) `t`. A semi-finite system with uniform background flow in the x-direction has a constant-concentration
    source boundary at the inlet. The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources
    can be superimposed in time and space.

    Parameters
    ----------
    c0 : float
        Source concentration [M/L**3]
    x : float or 1D of floats
        x-location(s) to compute output at [L].
    t : float or 1D of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    al : float
        Longitudinal dispersivity [L].
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    lamb : float, optional
        First-order decay rate [1/T], defaults to 0 (no decay).
    R : float, optional
        Retardation coefficient [-]; defaults to 1 (no retardation).

    Returns
    -------
    ndarray
        Numpy array with computed concentrations at location(s) `x` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)

    D = al * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    D = D / R

    u = np.sqrt(v**2 + 4 * lamb * D)
    term = np.exp(x * (v - u) / (2 * D)) * erfc(
        (x - u * t) / (2 * np.sqrt(D * t))
    ) + np.exp(x * (v + u) / (2 * D)) * erfc((x + u * t) / (2 * np.sqrt(D * t)))

    return c0 * 0.5 * term


def seminf3(c0, x, t, v, al, Dm=0.0, lamb=0.0, R=1.0):
    """Compute the 1D concentration field of a dissolved solute from a Cauchy-type inlet source in
    a semi-finite system with uniform background flow.

    Source: [wexler_1992]_ - SEMINF (3) algorithm (equations 67 & 68).

    The one-dimensional advection-dispersion equation is solved for concentration at specified `x` location(s) and
    output time(s) `t`. A semi-finite system with uniform background flow in the x-direction has a Cauchy-type source boundary
    at the inlet where water with specified concentration `c0` is flowing into the system with the background flow.
    The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed
    in time and space.

    For very small non-zero values of `lamb`, the solution may suffer from round-off errors.

    Parameters
    ----------
    c0 : float
        Source concentration [M/L**3]
    x : float or 1D of floats
        x-location(s) to compute output at [L].
    t : float or 1D of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    al : float
        Longitudinal dispersivity [L].
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    lamb : float, optional
        First-order decay rate [1/T], defaults to 0 (no decay).
    R : float, optional
        Retardation coefficient [-]; defaults to 1 (no retardation).

    Returns
    -------
    ndarray
        Numpy array with computed concentrations at location(s) `x` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)

    D = al * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    D = D / R

    u = np.sqrt(v**2 + 4 * lamb * D)
    if lamb == 0:
        term = (
            0.5 * erfc((x - v * t) / (2 * np.sqrt(D * t)))
            + np.sqrt(t * v**2 / (np.pi * D))
            * np.exp(-((x - v * t) ** 2) / (4 * D * t))
            - 0.5
            * (1 + v * x / D + t * v**2 / D)
            * np.exp(v * x / D)
            * erfc((x + v * t) / (2 * np.sqrt(D * t)))
        )
        term0 = 1.0
    else:
        term = (
            2 * np.exp(x * v / D - lamb * t) * erfc((x + v * t) / (2 * np.sqrt(D * t)))
            + (u / v - 1)
            * np.exp(x * (v - u) / (2 * D))
            * erfc((x - u * t) / (2 * np.sqrt(D * t)))
            - (u / v - 1)
            * np.exp(x * (v + u) / (2 * D))
            * erfc((x + u * t) / (2 * np.sqrt(D * t)))
        )
        term0 = v**2 / (4 * lamb * D)

    return c0 * term0 * term


def pulse1(m0, x, t, v, n, al, xc=0.0, Dm=0.0, lamb=0.0, R=1.0):
    """Compute the 1D concentration field of a dissolved solute from an instantaneous pulse point source in an infinite aquifer
    with uniform background flow.

    Source: [bear_1979]_

    The one-dimensional advection-dispersion equation is solved for concentration at specified `x` location(s) and
    output time(s) `t`. An infinite system with uniform background flow in the x-direction is subjected to a pulse source
    with mass `m0` at `xc` at time `t=0`.
    The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed
    in time and space.
    Note that the equation has the same shape as the probability density function of a Gaussian distribution.

    The mass center of the plume at a given time `t` can be found at `x=xc + v*t/R`.

    Parameters
    ----------
    m0 : float
        Source mass [M].
    x : float or 1D of floats
        x-location(s) to compute output at [L].
    t : float or 1D of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    n : float
        Aquifer porosity. Should be between 0 and 1 [-].
    al : float
        Longitudinal dispersivity [L].
    xc : float
        x-coordinate of the point source [L], defaults to 0.0.
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    lamb : float, optional
        First-order decay rate [1/T], defaults to 0 (no decay).
    R : float, optional
        Retardation coefficient [-]; defaults to 1 (no retardation).

    Returns
    -------
    ndarray
        Numpy array with computed concentrations at location(s) `x` and time(s) `t`.

    References
    ----------
    .. [bear_1979] Bear, J., 1979. Hydraulics of Groundwater. New York, McGraw Hill, 596 p.

    """
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)

    D = al * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    D = D / R

    term0 = (
        1
        / (n * np.sqrt(4 * np.pi * D * t))
        * np.exp(-((x - xc - v * t) ** 2) / (4 * D * t) - lamb * t)
    )

    return m0 * term0


@njit
def _integrand_point1(tau, x, v, D, xc, lamb):
    return (
        1
        / np.sqrt(tau)
        * np.exp(-(v**2 / (4 * D) + lamb) * tau - (x - xc) ** 2 / (4 * D * tau))
    )


def point1(c0, x, t, v, n, al, qi, xc, Dm=0.0, lamb=0.0, R=1.0, order=100):
    """Compute the 1D concentration field of a dissolved solute from a continuous point source in an infinite aquifer or column
    with uniform background flow.

    Source: [bear_1979]_

    The one-dimensional advection-dispersion equation is solved for concentration at specified `x` location(s) and
    output time(s) `t`. A point source is continuously injecting a known concentration `c0` at known injection flux `qi` in the infinite aquifer
    with specified uniform background flow in the x-direction. It is assumed that the injection rate does not significantly alter the flow
    field. The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed in time and space.

    If multiple `x` values are specified, only one `t` can be supplied, and vice versa.

    A Gauss-Legendre quadrature of order `order` is used to solve the integral. For `x` values very close to the source location
    (`xc`), the algorithm might have trouble finding a solution.

    Parameters
    ----------
    c0 : float
        Point source concentration [M/L**3]
    x : float or 1D array of floats
        x-location(s) to compute output at [L].
    t : float or 1D or 2D array of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    n : float
        Aquifer porosity. Should be between 0 and 1 [-].
    al : float
        Longitudinal dispersivity [L].
    qi : float
        Injection flux rate (positive) of the point source [L/T].
    xc : float
        x-coordinate of the point source [L].
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    lamb : float, optional
        First-order decay rate [1/T], defaults to 0 (no decay).
    R : float, optional
        Retardation coefficient [-]; defaults to 1 (no retardation).
    order : integer, optional
        Order of the Gauss-Legendre polynomial used in the integration. Defaults to 100.

    Returns
    -------
    ndarray
        Numpy array with computed concentrations at location(s) `x` and time(s) `t`.

    References
    ----------
    .. [bear_1979] Bear, J., 1979. Hydraulics of Groundwater. New York, McGraw Hill, 596 p.

    """
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)

    D = al * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    D = D / R
    qi = qi / R

    if len(t) > 1 and len(x) > 1:
        raise ValueError(
            "If multiple values for t are specified, only one x value is allowed"
        )

    term = integrate(
        _integrand_point1,
        t,
        x,
        v,
        D,
        xc,
        lamb,
        order=order,
        method="legendre",
    )
    term0 = qi / (n * np.sqrt(4 * np.pi * D)) * np.exp(v * (x - xc) / (2 * D))

    return c0 * term0 * term
