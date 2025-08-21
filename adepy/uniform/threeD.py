import numpy as np
from numba import njit
from adepy._helpers import _erfc_nb as erfc
from adepy._helpers import _integrate as integrate


def point3(c0, x, y, z, t, v, n, al, ah, av, Q, xc, yc, zc, Dm=0.0, lamb=0.0, R=1.0):
    """Compute the 3D concentration field of a dissolved solute from a continuous point source in an infinite aquifer
    with uniform background flow.

    Source: [wexler_1992]_ - POINT3 algorithm (equation 105).

    The three-dimensional advection-dispersion equation is solved for concentration at specified `x`, `y` and `z` location(s) and
    output time(s) `t`. A point source is continuously injecting a known concentration `c0` at known injection rate `Q` in the infinite aquifer
    with specified uniform background flow in the x-direction. It is assumed that the injection rate does not significantly alter the flow
    field. The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed in time and space.

    If multiple `x`, `y` or `z` values are specified, only one `t` can be supplied, and vice versa.

    For `x`, `y` and `z` values very close to the source location (`xc-yc-zc`), the algorithm might have trouble finding a solution.
    At the source location, the solution is undefined. Simulated concentrations near the source may exceed `c0` for certain parameter combinations.
    See [wexler_1992]_.

    Parameters
    ----------
    c0 : float
        Point source concentration [M/L**3]
    x : float or 1D or 2D array of floats
        x-location(s) to compute output at [L].
    y : float or 1D or 2D array of floats
        y-location(s) to compute output at [L].
    z : float or 1D or 2D array of floats
        z-location(s) to compute output at [L].
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
    av : float
        Vertical transverse dispersivity [L].
    Q : float
        Volumetric injection rate (positive) of the point source [L**3/T].
    xc : float
        x-coordinate of the point source [L].
    yc : float
        y-coordinate of the point source [L].
    zc : float
        z-coordinate of the point source [L].
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    lamb : float, optional
        First-order decay rate [1/T], defaults to 0 (no decay).
    R : float, optional
        Retardation coefficient [-]; defaults to 1 (no retardation).

    Returns
    -------
    ndarray
        Numpy array with computed concentrations [M/L**3] at location(s) `x`, `y` and `z` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
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
    gamma = np.sqrt((x - xc) ** 2 + Dx * (y - yc) ** 2 / Dy + Dx * (z - zc) ** 2 / Dz)

    a = np.exp(v * (x - xc) / (2 * Dx)) / (8 * n * np.pi * gamma * np.sqrt(Dy * Dz))
    b = np.exp(gamma * beta / (2 * Dx)) * erfc(
        (gamma + beta * t) / (2 * np.sqrt(Dx * t))
    ) + np.exp(-gamma * beta / (2 * Dx)) * erfc(
        (gamma - beta * t) / (2 * np.sqrt(Dx * t))
    )

    return c0 * Q * a * b


@njit
def _isnan(x):
    return x != x


@njit
def _isneginf(x):
    return x == -np.inf


@njit
def _series_patchf(x, y, z, t, v, Dx, Dy, Dz, w, h, y1, y2, z1, z2, lamb, nterm):
    t = np.asarray(t.flatten(), dtype=np.float64)
    x = np.asarray(x.flatten(), dtype=np.float64)
    y = np.asarray(y.flatten(), dtype=np.float64)
    z = np.asarray(z.flatten(), dtype=np.float64)

    if len(t) > 1:
        series = np.zeros_like(t, dtype=np.float64)
        t_arr = t
        x_arr = np.full_like(t, x[0])
        y_arr = np.full_like(t, y[0])
        z_arr = np.full_like(t, z[0])
    else:
        series = np.zeros_like(x, dtype=np.float64)
        x_arr = x
        y_arr = y
        z_arr = z
        t_arr = np.full_like(x, t[0])

    subterm_outer = np.array([0.0], dtype=np.float64)
    outer_close_contour = 0
    for m in range(nterm):
        zeta = m * np.pi / h

        if m == 0:
            Om = (z2 - z1) / h
        else:
            Om = (np.sin(zeta * z2) - np.sin(zeta * z1)) / (m * np.pi)

        subterm_inner = 0.0
        inner_close = False
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

            for i in range(series.size):
                xi = x_arr[i]
                yi = y_arr[i]
                zi = z_arr[i]
                ti = t_arr[i]
                sqrt_Dx_t = 2 * np.sqrt(Dx * ti)
                arg1 = (xi - beta * ti) / sqrt_Dx_t
                arg2 = (xi + beta * ti) / sqrt_Dx_t
                term = np.exp((xi * (v - beta)) / (2 * Dx)) * erfc(arg1) + np.exp(
                    (xi * (v + beta)) / (2 * Dx)
                ) * erfc(arg2)
                add = Lmn * Om * Pn * np.cos(zeta * zi) * np.cos(eta * yi) * term

                # Numba-compatible nan/inf handling
                if _isneginf(add) or _isnan(add):
                    add = 0.0

                series[i] += add
                subterm_inner += add

            # if last 10 inner terms sum to < 1e-12, exit inner loop
            if (n + 1) % 10 == 0:
                if np.abs(subterm_inner) < 1e-12:
                    inner_close = True
                    break
                else:
                    subterm_inner = 0.0

        # if summation difference < 1e-12 for 10 subsequent outer iterations
        # and inner summation is closed every time,
        # then exit outer loop
        if m > 0 and inner_close:
            if np.all(np.abs(np.asarray(subterm_outer) - np.asarray(series)) < 1e-12):
                outer_close_contour += 1
            else:
                outer_close_contour = 0
        else:
            outer_close_contour = 0

        if outer_close_contour == 10:
            break

        subterm_outer = series.copy()

    return series


def patchf(
    c0,
    x,
    y,
    z,
    t,
    v,
    al,
    ah,
    av,
    w,
    h,
    y1,
    y2,
    z1,
    z2,
    Dm=0.0,
    lamb=0.0,
    R=1.0,
    nterm=100,
):
    """Compute the 3D concentration field of a dissolved solute from a finite-width and height source in an finite-width
    and height aquifer with uniform background flow.

    Source: [wexler_1992]_ - PATCHF algorithm (equation 114).

    The three-dimensional advection-dispersion equation is solved for concentration at specified `x`, `y` and `z` location(s) and
    output time(s) `t`. The source is located at `x=0` and has a finite width (along the y-axis) and finite height (along the z-axis),
    i.e. a "patch" source. The concentration at the source location remains constant. The aquifer has a finite width (y-extent),
    finite height (z-extent) and a specified uniform background flow in the x-direction. The solute can be subjected to 1st-order decay.
    Since the equation is linear, multiple sources can be superimposed in time and space.

    If multiple `x`, `y` or `z` values are specified, only one `t` can be supplied, and vice versa.
    If `lambda` and the transverse dispersion coefficients are all zero, no solution can be found.

    The solution consists of an infinite summation. A maximum of `nterm` terms are used in the algorithm.
    If the solution oscillates, try increasing `nterm`. For small values of `x and `t`, the solution may have trouble converging.

    Parameters
    ----------
    c0 : float
        Source concentration [M/L**3]
    x : float or 1D or 2D array of floats
        x-location(s) to compute output at [L].
    y : float or 1D or 2D array of floats
        y-location(s) to compute output at [L].
    z : float or 1D or 2D array of floats
        z-location(s) to compute output at [L].
    t : float or 1D or 2D array of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    al : float
        Longitudinal dispersivity [L].
    ah : float
        Horizontal transverse dispersivity [L].
    av : float
        Vertical transverse dispersivity [L].
    w : float
        Aquifer width [L].
    h : float
        Aquifer height [L].
    y1 : float
        Lower y-coordinate of the solute source at `x=0` [L].
    y2 : float
        Upper y-coordinate of the solute source at `x=0` [L].
    z1 : float
        Lower z-coordinate of the solute source at `x=0` [L].
    z2 : float
        Upper z-coordinate of the solute source at `x=0` [L].
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
        Numpy array with computed concentrations [M/L**3] at location(s) `x`, `y` and `z` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
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
        raise ValueError(
            "If multiple values for t are specified, only one x, y and z value are allowed"
        )

    if lamb == 0 and Dy == 0 and Dz == 0:
        raise ValueError("One of Dy, Dz or lamb should be non-zero")

    series = _series_patchf(
        x, y, z, t, v, Dx, Dy, Dz, w, h, y1, y2, z1, z2, lamb, nterm
    )

    if len(t) > 1:
        series = series.reshape(t.shape)
    else:
        series = series.reshape(x.shape)
    return c0 * series


@njit
def _integrand_patchi(tau, x, y, z, v, Dx, Dy, Dz, y1, y2, z1, z2, lamb):
    ig = (
        1
        / tau**3
        * np.exp(-(v**2 / (4 * Dx) + lamb) * tau**4 - x**2 / (4 * Dx * tau**4))
        * (
            erfc((y1 - y) / (2 * tau**2 * np.sqrt(Dy)))
            - erfc((y2 - y) / (2 * tau**2 * np.sqrt(Dy)))
        )
        * (
            erfc((z1 - z) / (2 * tau**2 * np.sqrt(Dz)))
            - erfc((z2 - z) / (2 * tau**2 * np.sqrt(Dz)))
        )
    )

    return ig


def patchi(
    c0, x, y, z, t, v, al, ah, av, y1, y2, z1, z2, Dm=0.0, lamb=0.0, R=1.0, order=100
):
    """Compute the 3D concentration field of a dissolved solute from a finite-width and height source in an semi-infinite
    aquifer with uniform background flow.

    Source: [wexler_1992]_ - PATCHI algorithm (equation 121b).

    The three-dimensional advection-dispersion equation is solved for concentration at specified `x`, `y` and `z` location(s) and
    output time(s) `t`. The source is located at `x=0` and has a finite width (along the y-axis) and finite height (along the z-axis),
    i.e. a "patch" source. The concentration at the source location remains constant. The aquifer has a infinite width (y-extent),
    infinite height (z-extent) and a specified uniform background flow in the x-direction. The solute can be subjected to 1st-order decay.
    Since the equation is linear, multiple sources can be superimposed in time and space.

    If multiple `x`, `y` or `z` values are specified, only one `t` can be supplied, and vice versa.

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
    z : float or 1D or 2D array of floats
        z-location(s) to compute output at [L].
    t : float or 1D or 2D array of floats
        Time(s) to compute output at [T].
    v : float
        Average linear groundwater flow velocity of the uniform background flow in the x-direction [L/T].
    al : float
        Longitudinal dispersivity [L].
    ah : float
        Horizontal transverse dispersivity [L].
    av : float
        Vertical transverse dispersivity [L].
    y1 : float
        Lower y-coordinate of the solute source at `x=0` [L].
    y2 : float
        Upper y-coordinate of the solute source at `x=0` [L].
    z1 : float
        Lower z-coordinate of the solute source at `x=0` [L].
    z2 : float
        Upper z-coordinate of the solute source at `x=0` [L].
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
        Numpy array with computed concentrations [M/L**3] at location(s) `x`, `y` and `z` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
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

    term = integrate(
        _integrand_patchi,
        t ** (1 / 4),
        x,
        y,
        z,
        v,
        Dx,
        Dy,
        Dz,
        y1,
        y2,
        z1,
        z2,
        lamb,
        order=order,
        method="legendre",
    )

    term0 = x * np.exp(v * x / (2 * Dx)) / (2 * np.sqrt(np.pi * Dx))

    return c0 * term0 * term


def pulse3(
    m0, x, y, z, t, v, n, al, ah, av, xc=0.0, yc=0.0, zc=0.0, Dm=0.0, lamb=0.0, R=1.0
):
    """Compute the 3D concentration field of a dissolved solute from an instantaneous pulse point source in an infinite aquifer
    with uniform background flow.

    Source: [wexler_1992]_ - POINT3 algorithm (equation 104).

    The three-dimensional advection-dispersion equation is solved for concentration at specified `x`, `y` and `z` location(s) and
    output time(s) `t`. An infinite system with uniform background flow in the x-direction is subjected to a pulse source
    with mass `m0` at `xc-yc-zc` at time `t=0`.
    The solute can be subjected to 1st-order decay. Since the equation is linear, multiple sources can be superimposed
    in time and space.
    Note that the equation has the same shape as the probability density function of a trivariate Gaussian distribution.

    The mass center of the plume at a given time `t` can be found at `y=yc`, `z=zc` and `x=xc + v*t/R`.

    Parameters
    ----------
    m0 : float
        Source mass [M]
    x : float or 1D or 2D array of floats
        x-location(s) to compute output at [L].
    y : float or 1D or 2D array of floats
        y-location(s) to compute output at [L].
    z : float or 1D or 2D array of floats
        z-location(s) to compute output at [L].
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
    av : float
        Vertical transverse dispersivity [L].
    xc : float
        x-coordinate of the point source [L], defaults to 0.0.
    yc : float
        y-coordinate of the point source [L], defaults to 0.0.
    zc : float
        z-coordinate of the point source [L], defaults to 0.0.
    Dm : float, optional
        Effective molecular diffusion coefficient [L**2/T]; defaults to 0 (no molecular diffusion).
    lamb : float, optional
        First-order decay rate [1/T], defaults to 0 (no decay).
    R : float, optional
        Retardation coefficient [-]; defaults to 1 (no retardation).

    Returns
    -------
    ndarray
        Numpy array with computed concentrations [M/L**3] at location(s) `x`, `y` and `z` and time(s) `t`.

    References
    ----------
    .. [wexler_1992] Wexler, E.J., 1992. Analytical solutions for one-, two-, and three-dimensional
        solute transport in ground-water systems with uniform flow, USGS Techniques of Water-Resources
        Investigations 03-B7, 190 pp., https://doi.org/10.3133/twri03B7

    """
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

    term0 = (
        1
        / (8 * n * np.sqrt(Dx * Dy * Dz * (np.pi * t) ** 3))
        * np.exp(
            -((x - xc - v * t) ** 2) / (4 * Dx * t)
            - (y - yc) ** 2 / (4 * Dy * t)
            - (z - zc) ** 2 / (4 * Dz * t)
            - lamb * t
        )
    )

    return m0 * term0
