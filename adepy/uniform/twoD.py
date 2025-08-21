import numpy as np
from numba import njit
from adepy._helpers import _erfc_nb as erfc
from adepy._helpers import _integrate as integrate


@njit
def _integrand_point2(tau, x, y, v, Dx, Dy, xc, yc, lamb):
    return (
        1
        / tau
        * np.exp(
            -(v**2 / (4 * Dx) + lamb) * tau
            - (x - xc) ** 2 / (4 * Dx * tau)
            - (y - yc) ** 2 / (4 * Dy * tau)
        )
    )


def point2(c0, x, y, t, v, n, al, ah, Qa, xc, yc, Dm=0.0, lamb=0.0, R=1.0, order=100):
    """Compute the 2D concentration field of a dissolved solute from a continuous point source in an infinite aquifer
    with uniform background flow.

    Source: [wexler_1992]_ - POINT2 algorithm (equation 76).

    The two-dimensional advection-dispersion equation is solved for concentration at specified `x` and `y` location(s) and
    output time(s) `t`. A point source is continuously injecting a known concentration `c0` at known injection rate `Qa` in the infinite aquifer
    with specified uniform background flow in the x-direction. It is assumed that the injection rate does not significantly alter the flow
    field. The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed in time and space.

    If multiple `x` or `y` values are specified, only one `t` can be supplied, and vice versa.

    A Gauss-Legendre quadrature of order `order` is used to solve the integral. For `x` and `y` values very close to the source location
    (`xc-yc`), the algorithm might have trouble finding a solution since the integral becomes a form of an exponential integral. See [wexler_1992]_.

    Parameters
    ----------
    c0 : float
        Point source concentration [M/L**3]
    x : float or 1D or 2D array of floats
        x-location(s) to compute output at [L].
    y : float or 1D or 2D array of floats
        y-location(s) to compute output at [L].
    t : float or 1D or 2D array of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    n : float
        Aquifer porosity. Should be between 0 and 1 [-].
    al : float
        Longitudinal dispersivity [L].
    ah : float
        Horizontal transverse dispersivity [L].
    Qa : float
        Volumetric injection rate (positive) of the point source per unit aquifer thickness [L**2/T].
    xc : float
        x-coordinate of the point source [L].
    yc : float
        y-coordinate of the point source [L].
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
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and `y` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)

    Dx = al * v + Dm
    Dy = ah * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    Dx = Dx / R
    Dy = Dy / R
    Qa = Qa / R

    if len(t) > 1 and (len(x) > 1 or len(y) > 1):
        raise ValueError(
            "If multiple values for t are specified, only one x and y value are allowed"
        )

    term = integrate(
        _integrand_point2,
        t,
        x,
        y,
        v,
        Dx,
        Dy,
        xc,
        yc,
        lamb,
        order=order,
        method="legendre",
    )
    term0 = Qa / (4 * n * np.pi * np.sqrt(Dx * Dy)) * np.exp(v * (x - xc) / (2 * Dx))

    return c0 * term0 * term


# @njit
def _series_stripf(x, y, t, v, Dx, Dy, y2, y1, w, lamb, nterm):
    if len(t) > 1:
        series = np.zeros_like(t, dtype=np.float64)
    else:
        series = np.zeros_like(x, dtype=np.float64)

    subterm = 0.0
    for n in range(nterm):
        eta = n * np.pi / w
        beta = np.sqrt(v**2 + 4 * Dx * (eta**2 * Dy + lamb))

        if n == 0:
            Ln = 0.5
            Pn = (y2 - y1) / w
        else:
            Ln = 1
            Pn = (np.sin(eta * y2) - np.sin(eta * y1)) / (n * np.pi)

        term = np.exp((x * (v - beta)) / (2 * Dx)) * erfc(
            (x - beta * t) / (2 * np.sqrt(Dx * t))
        ) + np.exp((x * (v + beta)) / (2 * Dx)) * erfc(
            (x + beta * t) / (2 * np.sqrt(Dx * t))
        )

        add = Ln * Pn * np.cos(eta * y) * term

        add = np.where(np.isneginf(add), 0.0, add)
        add = np.where(np.isnan(add), 0.0, add)
        series += add

        # if last 10 terms sum to < 1e-12, exit loop
        # checked every 10 terms
        # TODO check this
        subterm += add
        if (n + 1) % 10 == 0:
            if np.all(abs(subterm) < 1e-12):
                break
            else:
                subterm = 0.0

    # if n == (nterm - 1):
    #     warnings.warn(f"Series did not converge in {nterm} summations")

    return series


def stripf(c0, x, y, t, v, al, ah, y1, y2, w, Dm=0.0, lamb=0.0, R=1.0, nterm=100):
    """Compute the 2D concentration field of a dissolved solute from a finite-width source in an finite-width aquifer
    with uniform background flow.

    Source: [wexler_1992]_ - STRIPF algorithm (equation 85).

    The two-dimensional advection-dispersion equation is solved for concentration at specified `x` and `y` location(s) and
    output time(s) `t`. The source is located at `x=0` and has a finite width (along the y-axis), i.e. a "strip" source.
    The concentration at the source location remains constant. The aquifer has a finite width (y-extent) and a specified uniform background
    flow in the x-direction. The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed
    in time and space.

    If multiple `x` or `y` values are specified, only one `t` can be supplied, and vice versa.
    If both `lambda` and the horizontal transverse dispersion coefficient are zero, no solution can be found.

    The solution consists of an infinite summation. A maximum of `nterm` terms are used in the algorithm.
    If the solution oscillates, try increasing `nterm`. For small values of `x`, the solution may have trouble converging.

    Parameters
    ----------
    c0 : float
        Source concentration [M/L**3]
    x : float or 1D or 2D array of floats
        x-location(s) to compute output at [L].
    y : float or 1D or 2D array of floats
        y-location(s) to compute output at [L].
    t : float or 1D or 2D array of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    al : float
        Longitudinal dispersivity [L].
    ah : float
        Horizontal transverse dispersivity [L].
    y1 : float
        Lower y-coordinate of the solute source at `x=0` [L].
    y2 : float
        Upper y-coordinate of the solute source at `x=0` [L].
    w : float
        Aquifer width [L].
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    lamb : float, optional
        First-order decay rate [1/T], defaults to 0 (no decay).
    R : float, optional
        Retardation coefficient [-]; defaults to 1 (no retardation).
    nterm : integer, optional
        Maximum number of terms used in the series summation. Defaults to 100.

    Returns
    -------
    ndarray
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and `y` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)

    Dx = al * v + Dm
    Dy = ah * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    Dx = Dx / R
    Dy = Dy / R

    if lamb == 0 and Dy == 0:
        raise ValueError("Either Dy or lamb should be non-zero")

    if len(t) > 1 and (len(x) > 1 or len(y) > 1):
        raise ValueError(
            "If multiple values for t are specified, only one x and y value are allowed"
        )

    series = _series_stripf(x, y, t, v, Dx, Dy, y2, y1, w, lamb, nterm)

    return c0 * series


@njit
def _integrand_stripi(tau, x, y, v, Dx, Dy, y2, y1, lamb):
    # error in Wexler, 1992, eq. 91a: denominator of last erfc term should be multiplied by 2.
    # here, 91b is used
    ig = (
        (1 / (tau**3))
        * np.exp(-(v**2 / (4 * Dx) + lamb) * tau**4 - x**2 / (4 * Dx * tau**4))
        * (
            erfc((y1 - y) / (2 * tau**2 * np.sqrt(Dy)))
            - erfc((y2 - y) / (2 * tau**2 * np.sqrt(Dy)))
        )
    )
    return ig


def stripi(c0, x, y, t, v, al, ah, y1, y2, Dm=0.0, lamb=0.0, R=1.0, order=100):
    """Compute the 2D concentration field of a dissolved solute from a finite-width source in an semi-infinite aquifer
    with uniform background flow.

    Source: [wexler_1992]_ - STRIPI algorithm (equation 91b).

    The two-dimensional advection-dispersion equation is solved for concentration at specified `x` and `y` location(s) and
    output time(s) `t`. The source is located at `x=0` and has a finite width (along the y-axis), i.e. a "strip" source.
    The concentration at the source location remains constant. The aquifer has a infinite width (y-extent) and a specified uniform background
    flow in the x-direction. The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed
    in time and space.

    If multiple `x` or `y` values are specified, only one `t` can be supplied, and vice versa.

    A Gauss-Legendre quadrature of order `order` is used to solve the integral. For very small `x` values at large times `t`,
    round-off errors may occur.

    Parameters
    ----------
    c0 : float
        Source concentration [M/L**3]
    x : float or 1D or 2D array of floats
        x-location(s) to compute output at [L].
    y : float or 1D or 2D array of floats
        y-location(s) to compute output at [L].
    t : float or 1D or 2D array of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    al : float
        Longitudinal dispersivity [L].
    ah : float
        Horizontal transverse dispersivity [L].
    y1 : float
        Lower y-coordinate of the solute source at `x=0` [L].
    y2 : float
        Upper y-coordinate of the solute source at `x=0` [L].
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
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and `y` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)

    Dx = al * v + Dm
    Dy = ah * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    Dx = Dx / R
    Dy = Dy / R

    term = integrate(
        _integrand_stripi,
        t ** (1 / 4),
        x,
        y,
        v,
        Dx,
        Dy,
        y2,
        y1,
        lamb,
        order=order,
        method="legendre",
    )
    term0 = x / (np.sqrt(np.pi * Dx)) * np.exp(v * x / (2 * Dx))

    return c0 * term0 * term


@njit
def _integrand_gauss(tau, x, y, v, Dx, Dy, yc, sigma, lamb):
    beta = v**2 / (4 * Dx) + lamb
    gamma = Dy * tau**4 + sigma**2 / 2
    ig = np.exp(
        -beta * tau**4 - x**2 / (4 * Dx * tau**4) - (y - yc) ** 2 / (4 * gamma)
    ) / (tau**3 * np.sqrt(gamma))

    return ig


def gauss(c0, x, y, t, v, al, ah, yc, sigma, Dm=0.0, lamb=0.0, R=1.0, order=100):
    """Compute the 2D concentration field of a dissolved solute from a Gaussian source in an semi-infinite aquifer
    with uniform background flow.

    Source: [wexler_1992]_ - GAUSS algorithm (equation 98).

    The two-dimensional advection-dispersion equation is solved for concentration `x` and `y` location(s) and
    output time(s) `t`. The source is located at `x=0` and has a Gaussian concentration distribution along the y-axis.
    The concentration at the source location remains constant. The aquifer has a infinite width (y-extent) and a specified uniform background
    flow in the x-direction. The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed
    in time and space.

    If multiple `x` or `y` values are specified, only one `t` can be supplied, and vice versa.

    A Gauss-Legendre quadrature of order `order` is used to solve the integral.

    Parameters
    ----------
    c0 : float
        Maximum concentration at the center of the Gaussian solute source (`yc`) [M/L**3]
    x : float or 1D or 2D array of floats
        x-location(s) to compute output at [L].
    y : float or 1D or 2D array of floats
        y-location(s) to compute output at [L].
    t : float or 1D or 2D array of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    al : float
        Longitudinal dispersivity [L].
    ah : float
        Horizontal transverse dispersivity [L].
    yc : float
        Center y-coordinate of the Gaussian solute source at `x=0` [L].
    sigma : float
        Standard deviation of the geometry of the Gaussian solute source at `x=0` [L].
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
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and `y` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)

    Dx = al * v + Dm
    Dy = ah * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    Dx = Dx / R
    Dy = Dy / R

    term = integrate(
        _integrand_gauss,
        t ** (1 / 4),
        x,
        y,
        v,
        Dx,
        Dy,
        yc,
        sigma,
        lamb,
        order=order,
        method="legendre",
    )
    term0 = 2 * x * sigma / np.sqrt(2 * np.pi * Dx) * np.exp(v * x / (2 * Dx))

    return c0 * term0 * term


def pulse2(m0, x, y, t, v, n, al, ah, xc=0.0, yc=0.0, Dm=0.0, lamb=0.0, R=1.0):
    """Compute the 2D concentration field of a dissolved solute from an instantaneous pulse point source in an infinite aquifer
    with uniform background flow.

    Source: [bear_1979]_

    The two-dimensional advection-dispersion equation is solved for concentration at specified `x` and `y` location(s) and
    output time(s) `t`. An infinite system with uniform background flow in the x-direction is subjected to a pulse source
    with mass `m0` at `xc-yc` at time `t=0`.
    The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed
    in time and space.
    Note that the equation has the same shape as the probability density function of a bivariate Gaussian distribution.

    The mass center of the plume at a given time `t` can be found at `y=yc` and `x=xc + v*t/R`.

    Parameters
    ----------
    m0 : float
        Source mass [M].
    x : float or 1D or 2D array of floats
        x-location(s) to compute output at [L].
    y : float or 1D or 2D array of floats
        y-location(s) to compute output at [L].
    t : float or 1D of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    n : float
        Aquifer porosity. Should be between 0 and 1 [-].
    al : float
        Longitudinal dispersivity [L].
    ah : float
        Horizontal transverse dispersivity [L].
    xc : float
        x-coordinate of the point source [L], defaults to 0.0.
    yc : float
        y-coordinate of the point source [L], defaults to 0.0.
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    lamb : float, optional
        First-order decay rate [1/T], defaults to 0 (no decay).
    R : float, optional
        Retardation coefficient [-]; defaults to 1 (no retardation).

    Returns
    -------
    ndarray
        Numpy array with computed concentrations [M/L**3] at location(s) `x` and `y` and time(s) `t`.

    References
    ----------
    .. [bear_1979] Bear, J., 1979. Hydraulics of Groundwater. New York, McGraw Hill, 596 p.

    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    t = np.atleast_1d(t)

    Dx = al * v + Dm
    Dy = ah * v + Dm

    # apply retardation coefficient to right-hand side
    v = v / R
    Dx = Dx / R
    Dy = Dy / R

    term0 = (
        1
        / (4 * np.pi * n * t * np.sqrt(Dx * Dy))
        * np.exp(
            -((x - xc - v * t) ** 2) / (4 * Dx * t)
            - (y - yc) ** 2 / (4 * Dy * t)
            - lamb * t
        )
    )

    return m0 * term0
