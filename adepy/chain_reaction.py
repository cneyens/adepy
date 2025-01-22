import numpy as np
from scipy.linalg import solve


def _get_p_transform(ancestry, lamb, stoi):
    nspecies = len(ancestry)
    species = np.arange(nspecies)

    # create generation series and ancestors
    ancestors = {}
    for i in species:
        cont = True
        j = ancestry[i]
        ancestors[i] = []
        while cont:
            if j < 0:
                cont = False
            elif j in species:
                ancestors[i].append(j)
                j = ancestry[j]
            else:
                cont = False

    # get generation index gen
    gen = [len(ancestors[i]) + 1 for i in species]

    # get species-species matrix F
    F = np.zeros((nspecies, nspecies))
    for i in species:
        for j in species:
            if i == j:
                F[i, j] = gen[j]
            elif j in ancestors[i]:
                F[i, j] = gen[j]
            else:
                F[i, j] = 0

    # get species-generation matrix G
    maxg = np.max(gen)
    G = np.zeros((nspecies, maxg))
    for i in species:
        for j in range(maxg):
            for sp in species:
                if F[i, sp] == (j + 1):
                    G[i, j] = sp + 1

    F = F.astype(np.int32)
    G = G.astype(np.int32)

    # get transformation matrix P
    P = np.ones((nspecies, nspecies))
    for i in species:
        for j in species:
            fij = F[i, j]
            if fij == 0:
                P[i, j] = 0.0
            elif j == i:
                P[i, j] = 1.0
            else:
                for li in range(fij, F[i, i] - 1 + 1):
                    P[i, j] *= (
                        stoi[G[i, li + 1 - 1] - 1]
                        * lamb[G[i, li - 1] - 1]
                        / (lamb[G[i, li - 1] - 1] - lamb[i])
                    )

    return P


def chain_reaction(ancestry, lamb, stoi, c0, fun, **kwargs):
    """Compute sequential or parallel first-order parent-daughter chain reactions.

    First-order sequential or parallel parent-daughter chain reactions can be computed for an arbitrary number of reactive species. The
    concentration field for each species is computed using `fun` and the contribution of parent-daughter chain reactions is
    calculated using the method described by [sunea_1999]_. This method is restricted to the use of a single, constant retardation
    coefficient for all species.

    Parameters
    ----------
    ancestry : dict
        Dictionary with keys containing a species number and values the parent species number. A value of -1 means no parent.
    lamb : 1D array of floats
        1D array with the first-order decay rates [1/T] for each species.
    stoi : 1D array of floats
        1D array with the stoichiometric constants of the parent-daughter reaction. Species without a parent should be set to zero.
    c0 : 1D array of floats
        Source concentrations for each species [M/L**3].
    fun : function
        Function used to compute the concentration field. Kwargs are passed to this function.

    Returns
    -------
    list
        List with the concentration field for each species as computed by `fun` and subjected to first-order parent-daughter
        decay reactions.

    References
    ----------
    .. [sunea_1999] Sun, Y., Clement, T.P., 1999. A Decomposition Method for Solving Coupled Multi–Species Reactive Transport Problems,
        Transport in Porous Media 37, pp. 327–346, https://doi.org/10.1023/A:1006507514019

    """
    lamb = np.atleast_1d(lamb)
    stoi = np.atleast_1d(stoi)
    assert len(lamb) == len(stoi), "len(lamb) should equal len(stoi)"
    assert stoi[0] == 0, "First stoi value should equal 0.0"
    # assert len(np.unique(lamb)) == len(lamb), "No duplicate lamb coefficients allowed"

    P = _get_p_transform(ancestry, lamb, stoi)

    nspecies = len(lamb)

    # TODO initial concentrations (need transformation as well)
    # can they just be supplied as a matrix at t=0
    c0 = np.atleast_1d(c0)
    if len(c0.flatten()) == 1:
        c0 = np.repeat(c0, nspecies)
    # elif len(c0) != nspecies:
    #     raise ValueError('Specify c0 for all values or give 1 c0 value')

    # transform boundary conditions
    a0 = solve(a=P, b=c0)

    # solve in transformed domain
    a = []
    for i in range(nspecies):
        a.append(fun(c0=a0[i], lamb=lamb[i], **kwargs))
        if i == 0:
            shp = a[i].shape
        a[i] = a[i].flatten()

        # remove Inf and NA for inversion
        a[i][np.isposinf(a[i])] = 1e30
        a[i][np.isneginf(a[i])] = -1e30
        a[i][np.isnan(a[i])] = 0.0

    # back transform and reshape
    a = np.vstack(a)
    c = solve(a=np.linalg.inv(P), b=a)
    c = [c[i, :].reshape(shp) for i in range(nspecies)]

    return c
