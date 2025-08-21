from scipy.optimize import brentq
import numpy as np
from adepy._helpers import _erfc_nb as erfc
from adepy._helpers import _integrate as integrate
from adepy._helpers import _dehoog as dehoog
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
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and time(s) `t`.

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
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and time(s) `t`.

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
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and time(s) `t`.

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
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and time(s) `t`.

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
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and time(s) `t`.

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
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and time(s) `t`.

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


# @njit
def _mpne1d_laplace(
    p,
    c0,
    x,
    q,
    D,
    f,
    rhob,
    thm,
    thim,
    cout,
    delta,
    pchk,
    fchk,
    L,
    alfa,
    fm,
    fim,
    km,
    kim,
    km2,
    kim2,
    lamb,
    lsm1,
    lsm2,
    lim,
    lsim1,
    lsim2,
    icm,
    icim,
    icms,
    icims,
    domain=1,
    output="mobile",
):
    # Inflow Laplace transform
    fs = c0 / p

    # Initial condition terms
    G10 = (
        ((1 - f) * rhob * kim2 / (p + kim2 + lsim2)) * icims
        + (thim + (1 - f) * rhob * fim * kim) * icim
    ) / (
        p * (thim + (1 - f) * rhob * fim * kim)
        + thim * lim
        + (1 - f) * rhob * lsim1 * fim * kim
        + (1 - f) * rhob * (1 - fim) * kim * kim2 * (p + lsim2) / (p + kim2 + lsim2)
        + alfa
    )
    G20 = (thm + f * rhob * fm * km) * icm + (f * rhob * km2 / (p + km2 + lsm2)) * icms

    # Gamma terms
    GAM1 = (1 + (f * rhob * fm * km) / thm) * p
    GAM2 = (f * rhob * (1 - fm) * km * km2 * ((p + lsm2) / (p + km2 + lsm2))) / thm
    GAM3 = (
        alfa
        - (alfa**2)
        / (
            (thim + (1 - f) * rhob * fim * kim) * p
            + thim * lim
            + (1 - f) * rhob * lsim1 * fim * kim
            + (1 - f)
            * rhob
            * (1 - fim)
            * kim
            * kim2
            * ((p + lsim2) / (p + kim2 + lsim2))
            + alfa
        )
    ) / thm
    GAM4 = lamb + (f * rhob * lsm1 * fm * km) / thm

    B = thm * (GAM1 + GAM2 + GAM3 + GAM4)
    H1 = (q - np.sqrt(q**2 + 4 * B * thm * D)) / (2 * thm * D)
    H2 = (q + np.sqrt(q**2 + 4 * B * thm * D)) / (2 * thm * D)

    # Solution for different domains
    if domain == 1:  # Semi-infinite
        E1 = (
            fs - (alfa * G10 + G20) / B
            if (abs(q) <= 0.0 and delta <= 0.0)
            else (q / (q - thm * delta * D * H1)) * (fs - (alfa * G10 + G20) / B)
        )
        CMB = E1 * np.exp(H1 * x) + (alfa * G10 + G20) / B
    elif domain == 2:  # Finite, type II
        if abs(q) <= 0.0 and delta <= 0.0:
            D1 = H2 * np.exp(H2 * L) - H1 * np.exp(H1 * L)
            D2 = (fs - (alfa * G10 + G20) / B) * H2 * np.exp(H2 * L)
            D3 = -(fs - (alfa * G10 + G20) / B) * H1 * np.exp(H1 * L)
        else:
            D1 = H2 * (q - thm * delta * D * H1) * np.exp(H2 * L) - H1 * (
                q - thm * delta * D * H2
            ) * np.exp(H1 * L)
            D2 = q * (fs - (alfa * G10 + G20) / B) * H2 * np.exp(H2 * L)
            D3 = -q * (fs - (alfa * G10 + G20) / B) * H1 * np.exp(H1 * L)
        E1, E2 = D2 / D1, D3 / D1
        CMB = E1 * np.exp(H1 * x) + E2 * np.exp(H2 * x) + (alfa * G10 + G20) / B
    elif domain == 3:  # Finite, type I
        if abs(q) <= 0.0 and delta <= 0.0:
            D4 = np.exp(H2 * L) - np.exp(H1 * L)
            D5 = (fs - (alfa * G10 + G20) / B) * np.exp(H2 * L) - (
                cout / p - (alfa * G10 + G20) / B
            )
            D6 = -(fs - (alfa * G10 + G20) / B) * np.exp(H1 * L) + (
                cout / p - (alfa * G10 + G20) / B
            )
        else:
            D4 = (q - thm * delta * D * H1) * np.exp(H2 * L) - (
                q - thm * delta * D * H2
            ) * np.exp(H1 * L)
            D5 = q * (fs - (alfa * G10 + G20) / B) * np.exp(H2 * L) - (
                cout / p - (alfa * G10 + G20) / B
            ) * (q - thm * delta * D * H2)
            D6 = -q * (fs - (alfa * G10 + G20) / B) * np.exp(H1 * L) + (
                cout / p - (alfa * G10 + G20) / B
            ) * (q - thm * delta * D * H1)
        E1, E2 = D5 / D4, D6 / D4
        CMB = E1 * np.exp(H1 * x) + E2 * np.exp(H2 * x) + (alfa * G10 + G20) / B
    else:
        raise ValueError('"domain" should be 1, 2 or 3.')

    if output == "mobile":
        return np.atleast_1d(CMB)
    elif output == "immobile":
        if (pchk < 1e-10) and (fchk < 1e-10):
            return np.atleast_1d(0.0)
        denom = (
            p * (thim + (1 - f) * rhob * fim * kim)
            + thim * lim
            + (1 - f) * rhob * lsim1 * fim * kim
            + (1 - f)
            * rhob
            * (1 - fim)
            * kim
            * kim2
            * ((p + lsim2) / (p + kim2 + lsim2))
            + alfa
        )
        return np.atleast_1d(CMB * (alfa / denom) + G10)
    else:
        raise ValueError('output should be "mobile" or "immobile"')


def mpne(
    c0,
    x,
    t,
    v,
    al,
    n,
    rhob,
    L=None,
    Dm=0.0,
    phi=1.0,
    f=None,
    alfa=1.0,
    fm=1.0,
    fim=1.0,
    km=0.0,
    kim=0.0,
    km2=0.0,
    kim2=0.0,
    lamb=0.0,
    lsm1=None,
    lsm2=None,
    lim=None,
    lsim1=None,
    lsim2=None,
    icm=0.0,
    icim=0.0,
    icms=0.0,
    icims=0.0,
    cout=0.0,
    domain=1,
    inflowbc="cauchy",
    output="mobile",
):
    """Simulate 1D non-equilibrium transport in uniform background flow using the Multi-Process Non-Equilibrium (MPNE) model.

    Source: [sspapa_2004]_, [neville_2000]_

    1D solute transport in uniform background flow can be modelled using a 1st- or 3th-type boundary condition at the inlet, for a semi-infinite system,
    a finite system with constant outlet concentration, or a finite system with zero-gradient outlet. Chemical and physical non-equilibrium can be simulated seperately or simultaneously through a two-site and two-region approach, respectively.
    The two-site approach simulates chemical non-equilibrium, allowing for both linear equilibrium and first-order sorption to occur. The two-region capability allows for mobile-immobile transport, with a kinetic first-order mass transfer rate.
    Separate sorption coefficients can be specified for the immobile and mobile domains. Separate first-order decay rates can be specified for the aqueous (both mobile and immobile) and sorbed (mobile and immobile, for both the equilibrium and rate-limited sorbed) phases.
    Aqueous concentration in either the mobile or the immobile domain can be returned. Either `x` or `t` should be of length 1.

    The two-site and two-region models are interchangeable, so few codes include both non-equilibrium processes, unlike the MPNE model.

    The source code is taken from the MNPE1D code ([sspapa_2004]_, [neville_2000]_). The system is solved in the Laplace domain and back-transformed using the De Hoog algorithm.

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
    n : float
        Aquifer total porosity [-].
    rhob : float
        Dry bulk density of the aquifer [M/L**3].
    L : float, optional
        System length for a finite system, by default None. Only used when `domain` is 2 or 3.
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    phi : float, optional
        Fraction of the total porosity which is mobile [-], by default 1.0 (no immobile domain).
    f : float, optional
        Mass fraction of the sorbent in contact with the mobile region dissolved phase [-], by default equal to phi.
    alfa : float, optional
        First-order mass transfer rate between the mobile and immobile domain [1/T], by default 1.0 (unity)
    fm : float, optional
        Fraction of equilibrium sorption sites in the mobile domain [-], by default 1.0 (only equilibrium sorption).
    fim : float, optional
        Fraction of equilibrium sorption sites in the immobile domain [-], by default 1.0 (only equilibrium sorption).
    km : float, optional
        Linear equilibrium sorption coefficient in the mobile domain [L**3/M], by default 0.0 (no equilibrium sorption).
    kim : float, optional
        Linear equilibrium sorption coefficient in the immobile domain [L**3/M], by default 0.0 (no equilibrium sorption).
    km2 : float, optional
        First-order kinetic sorption rate in the mobile domain [1/T], by default 0.0 (no kinetic sorption).
    kim2 : float, optional
        First-order kinetic sorption rate in the immobile domain [1/T], by default 0.0 (no kinetic sorption).
    lamb : float, optional
        First-order decay rate of the aqueous phase in the mobile domain [1/T], by default 0.0 (no decay).
    lsm1 : float, optional
        First-order decay rate of the equilibrium sorbed phase in the mobile domain  [1/T], by default equal to `lamb`.
    lsm2 : float, optional
        First-order decay rate of the rate-limited sorbed phase in the mobile domain  [1/T], by default equal to `lsm1`.
    lim : float, optional
        First-order decay rate of the aqueous phase in the immobile domain [1/T], by default equal to `lamb`.
    lsim1 : float, optional
        First-order decay rate of the equilibrium sorbed phase in the immobile domain  [1/T], by default equal to `lim`.
    lsim2 : float, optional
        First-order decay rate of the rate-limited sorbed phase in the immobile domain  [1/T], by default equal to `lsim1`.
    icm : float, optional
        Initial condition of the aqueous phase in the mobile domain, by default 0.0
    icim : float, optional
        Initial condition of the aqueous phase in the immobile domain, by default 0.0
    icms : float, optional
        Initial condition of the sorbed phase in the mobile domain, by default 0.0
    icims : float, optional
        Initial condition of the sorbed phase in the immobile domain, by default 0.0
    cout : float, optional
        Outlet concentration for a finite system with a constant-concentration outlet (`idomain=3`), by default 0.0
    domain : int, optional
        Type of system. 1 = semi-infinite (default), 2 = finite with a zero-gradient outlet, 3 = finite with a constant-concentration outlet.
    inflowbc : str, optional
        Inlet boundary condition, either "dirichlet" (i.e. a constant-concentration) or "cauchy" (default, i.e. a 3th-type).
    output : str, optional
        Output concentration domain, either "mobile" (default) or "immobile".

    Returns
    -------
    ndarray
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and time(s) `t`.

    References
    -------
    .. [sspapa_2004] MPNE1D, 2004. MPNE1D Analytical Solution: User's Guide, version 4.1. S.S. Papadopulos & Associates, Inc.
    .. [neville_2000] Neville, C.J., Ibaraki, M., Sudicky, E.A., 2000. Solute transport with multiprocess nonequilibrium: a semi-analytical solution approach, Journal of Contaminant Hydrology 44-2, p. 141-159

    """
    t = np.atleast_1d(t)
    x = np.atleast_1d(x)

    if len(t) > 1 and len(x) > 1:
        raise ValueError('Either "t" or "x" should be of length 1')

    if inflowbc == "dirichlet":
        delta = 0.0
    elif inflowbc == "cauchy":
        delta = 1.0
    else:
        raise ValueError('inflowbc should be "dirichlet" or "cauchy"')

    # set porosities
    thm = phi * n
    thim = (1.0 - phi) * n
    pchk, fchk = abs(phi - 1.0), abs(f - 1.0)
    if (pchk < 1e-10) and (fchk < 1e-10):
        alfa = 1.0

    # set darcy and dispersion flux
    q = v * thm
    D = al * v + Dm

    # set defaults
    if f is None:
        f = phi
    if lsm1 is None:
        lsm1 = lamb
    if lsm2 is None:
        lsm2 = lsm1
    if lim is None:
        lim = lamb
    if lsim1 is None:
        lsim1 = lim
    if lsim2 is None:
        lsim2 = lsim1

    if len(t) > 1:
        c = np.zeros_like(t)
    else:
        c = np.zeros_like(x)
    if np.any(t == 0.0):
        raise ValueError("All t should be > 0.0")

    for i, ti in enumerate(t):
        c[i] = dehoog(
            t=ti,
            fbar=_mpne1d_laplace,
            c0=c0,
            x=x,
            q=q,
            D=D,
            f=f,
            rhob=rhob,
            thm=thm,
            thim=thim,
            cout=cout,
            delta=delta,
            pchk=pchk,
            fchk=fchk,
            L=L,
            alfa=alfa,
            fm=fm,
            fim=fim,
            km=km,
            kim=kim,
            km2=km2,
            kim2=kim2,
            lamb=lamb,
            lsm1=lsm1,
            lsm2=lsm2,
            lim=lim,
            lsim1=lsim1,
            lsim2=lsim2,
            icm=icm,
            icim=icim,
            icms=icms,
            icims=icims,
            domain=domain,
            output=output,
        )
    return c
