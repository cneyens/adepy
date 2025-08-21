from adepy.uniform.oneD import (
    finite1,
    finite3,
    seminf1,
    seminf3,
    pulse1,
    point1,
    mpne,
)
import numpy as np

# TODO add tests with retardation and decay


def test_finite1():
    c = finite1(1, 2.5, 5, v=0.6, al=1, L=12)  # 0.73160
    np.testing.assert_approx_equal(c[0], 0.73160, significant=5)

    c = finite1(1, 2.5, 5, v=0.6, al=1, L=12, R=8.31)  # 0.0105
    np.testing.assert_approx_equal(c[0], 0.01054, significant=4)

    L = 12
    c = finite1(1, [L, L + 1], 5, 0.6, 1, L)
    assert np.isnan(c[1])
    assert ~np.isnan(c[0])


def test_finite1_shape():
    x = [5.0, 6.0, 7.0]
    c = finite1(1.0, x, 5, v=0.6, al=1, L=12)
    assert c.shape == (len(x),)

    t = [10, 15, 20, 25]
    c = finite1(1.0, 2.5, t, v=0.6, al=1, L=12)
    assert c.shape == (len(t),)


def test_finite3():
    c = finite3(1, [2.5, 5, 7.5], 5, v=0.6, al=1, L=12)
    np.testing.assert_array_almost_equal(
        c, np.array([0.55821, 0.17878, 0.02525]), decimal=5
    )

    L = 12
    c = finite3(1, [L, L + 1], 5, 0.6, 1, L)
    assert np.isnan(c[1])
    assert ~np.isnan(c[0])


def test_finite3_shape():
    x = [5.0, 6.0, 7.0]
    c = finite3(1.0, x, 5, v=0.6, al=1, L=12)

    assert c.shape == (len(x),)

    t = [10, 15, 20, 25]
    c = finite3(1.0, 2.5, t, v=0.6, al=1, L=12)

    assert c.shape == (len(t),)


def test_seminf1():
    c = seminf1(1, [0.5, 2.5], 5, 0.6, al=1)
    np.testing.assert_array_almost_equal(c, np.array([0.97244, 0.73160]), decimal=5)


def test_seminf1_shape():
    x = [5.0, 6.0, 7.0]
    c = seminf1(1.0, x, 5, v=0.6, al=1)
    assert c.shape == (len(x),)

    t = [10, 15, 20, 25]
    c = seminf1(1.0, 2.5, t, v=0.6, al=1)
    assert c.shape == (len(t),)


def test_seminf3():
    c = seminf3(1, [0.5, 2.5], 5, 0.6, al=1)
    np.testing.assert_array_almost_equal(c, np.array([0.85904, 0.55821]), decimal=5)


def test_seminf3_shape():
    x = [5.0, 6.0, 7.0]
    c = seminf3(1.0, x, 5, v=0.6, al=1)
    assert c.shape == (len(x),)

    t = [10, 15, 20, 25]
    c = seminf3(1.0, 2.5, t, v=0.6, al=1)
    assert c.shape == (len(t),)


def test_pulse1():
    m0 = 1.0
    x = np.array([0.5, 2.0])
    t = 10
    v = 0.05
    al = 1.0
    n = 0.25

    sig = np.sqrt(2 * al * v * t)
    mu = v * t

    c = pulse1(m0, x, t, v, n, al)
    f = (
        m0
        / n
        * 1.0
        / np.sqrt(2 * np.pi * sig**2)
        * np.exp(-((x - mu) ** 2) / (2 * sig**2))
    )  # probability density function of gaussian distribution
    np.testing.assert_array_equal(c, f)


def test_pulse1_shape():
    x = [5.0, 6.0, 7.0]
    c = pulse1(1.0, x, 5, v=0.6, n=0.25, al=1)
    assert c.shape == (len(x),)

    t = [10, 15, 20, 25]
    c = pulse1(1.0, 2.5, t, v=0.6, n=0.25, al=1)
    assert c.shape == (len(t),)


def test_point1():
    # TODO verify
    c0 = 1.0
    qi = 1.0
    v = 0.05
    n = 0.2
    al = 1.2
    x = 5.0
    t = [5.0, 10.0]

    c = point1(c0, x, t, v, n, al, qi, xc=1.5)
    np.testing.assert_array_almost_equal(
        c, np.array([0.00016767, 0.06758618]), decimal=8
    )


def test_point1_shape():
    x = [5.0, 6.0, 7.0]
    c = point1(1.0, x, 5, v=0.6, n=0.25, qi=1.2, al=1, xc=0.0)
    assert c.shape == (len(x),)

    t = [10, 15, 20, 25]
    c = point1(1.0, 2.5, t, v=0.6, n=0.25, qi=1.2, al=1, xc=0.0)
    assert c.shape == (len(t),)

    np.testing.assert_raises(
        ValueError, point1, 1.0, [5.0, 2.0], [5.0, 10.0], 0.6, 0.25, 1.2, 1, 0.0
    )


def test_mpne_ex1():
    rhob = 1.360
    q = 5.11
    D = 3.673

    n = 0.473
    phi = 0.929

    v = q / (phi * n)
    al = D / v

    f = 0.929
    fm = 0.5
    fim = 0.50
    alfa = 0.075
    km = 0.429
    kim = 0.416
    km2 = 0.663
    kim2 = 0.663
    t0 = 7.672
    L = 30

    x = 30
    c0 = 0.5
    inflowbc = "cauchy"
    domain = 2  # finite with zero-gradient outlet

    cobs = np.array(
        [
            0.000000e00,
            1.401382e-08,
            6.519159e-06,
            2.294814e-05,
            8.087235e-05,
            2.641875e-03,
            2.375844e-02,
            7.822927e-02,
            1.471788e-01,
            2.082363e-01,
            2.572890e-01,
            2.973704e-01,
            3.309203e-01,
            3.592396e-01,
            3.831439e-01,
            4.032173e-01,
            4.200962e-01,
            4.342043e-01,
            4.459541e-01,
            4.557026e-01,
            4.629334e-01,
            4.576077e-01,
            4.196572e-01,
            3.568394e-01,
            2.954673e-01,
            2.455863e-01,
            2.052159e-01,
            1.716006e-01,
            1.432169e-01,
            1.191917e-01,
            9.890030e-02,
            8.182431e-02,
            6.750978e-02,
            5.555493e-02,
            4.560596e-02,
            3.735244e-02,
            3.052760e-02,
            2.490097e-02,
            2.027341e-02,
            1.647702e-02,
            1.336962e-02,
            1.083157e-02,
            8.762630e-03,
            7.079199e-03,
            5.711803e-03,
            4.602909e-03,
            3.705016e-03,
            2.979017e-03,
            2.392800e-03,
            1.920054e-03,
        ]
    )
    tobs = np.array(
        [
            5.000000e-01,
            1.000000e00,
            1.500000e00,
            2.000000e00,
            2.500000e00,
            3.000000e00,
            3.500000e00,
            4.000000e00,
            4.500000e00,
            5.000000e00,
            5.500000e00,
            6.000000e00,
            6.500000e00,
            7.000000e00,
            7.500000e00,
            8.000000e00,
            8.500000e00,
            9.000000e00,
            9.500000e00,
            1.000000e01,
            1.050000e01,
            1.100000e01,
            1.150000e01,
            1.200000e01,
            1.250000e01,
            1.300000e01,
            1.350000e01,
            1.400000e01,
            1.450000e01,
            1.500000e01,
            1.550000e01,
            1.600000e01,
            1.650000e01,
            1.700000e01,
            1.750000e01,
            1.800000e01,
            1.850000e01,
            1.900000e01,
            1.950000e01,
            2.000000e01,
            2.050000e01,
            2.100000e01,
            2.150000e01,
            2.200000e01,
            2.250000e01,
            2.300000e01,
            2.350000e01,
            2.400000e01,
            2.450000e01,
            2.500000e01,
        ]
    )
    cimobs = np.array(
        [
            0.000000e00,
            1.934931e-09,
            2.168152e-06,
            1.404369e-05,
            3.071076e-05,
            4.691693e-04,
            5.976666e-03,
            2.776300e-02,
            6.905335e-02,
            1.198774e-01,
            1.702363e-01,
            2.159905e-01,
            2.564480e-01,
            2.919273e-01,
            3.228982e-01,
            3.497636e-01,
            3.730368e-01,
            3.930651e-01,
            4.102191e-01,
            4.248435e-01,
            4.371408e-01,
            4.449532e-01,
            4.387816e-01,
            4.109601e-01,
            3.682457e-01,
            3.221777e-01,
            2.789845e-01,
            2.403114e-01,
            2.061486e-01,
            1.761412e-01,
            1.499122e-01,
            1.271045e-01,
            1.073775e-01,
            9.040361e-02,
            7.587071e-02,
            6.348500e-02,
            5.297419e-02,
            4.408951e-02,
            3.660655e-02,
            3.032484e-02,
            2.506804e-02,
            2.068158e-02,
            1.703083e-02,
            1.399986e-02,
            1.148922e-02,
            9.414025e-03,
            7.702193e-03,
            6.292758e-03,
            5.134356e-03,
            4.183867e-03,
        ]
    )

    c = mpne(
        c0,
        x,
        tobs,
        v,
        al,
        n,
        rhob,
        L=L,
        phi=phi,
        f=f,
        alfa=alfa,
        fm=fm,
        fim=fim,
        km=km,
        kim=kim,
        km2=km2,
        kim2=kim2,
        inflowbc=inflowbc,
        domain=domain,
    )

    c[tobs > t0] += mpne(
        -c0,
        x,
        tobs[tobs > t0] - t0,
        v,
        al,
        n,
        rhob,
        L=L,
        phi=phi,
        f=f,
        alfa=alfa,
        fm=fm,
        fim=fim,
        km=km,
        kim=kim,
        km2=km2,
        kim2=kim2,
        inflowbc=inflowbc,
        domain=domain,
    )

    # immobile
    cim = mpne(
        c0,
        x,
        tobs,
        v,
        al,
        n,
        rhob,
        L=L,
        phi=phi,
        f=f,
        alfa=alfa,
        fm=fm,
        fim=fim,
        km=km,
        kim=kim,
        km2=km2,
        kim2=kim2,
        inflowbc=inflowbc,
        domain=domain,
        output="immobile",
    )

    cim[tobs > t0] += mpne(
        -c0,
        x,
        tobs[tobs > t0] - t0,
        v,
        al,
        n,
        rhob,
        L=L,
        phi=phi,
        f=f,
        alfa=alfa,
        fm=fm,
        fim=fim,
        km=km,
        kim=kim,
        km2=km2,
        kim2=kim2,
        inflowbc=inflowbc,
        domain=domain,
        output="immobile",
    )

    np.testing.assert_array_almost_equal(cobs, c, decimal=4)
    np.testing.assert_array_almost_equal(cimobs, cim, decimal=4)


def test_mpne_ex3_5():
    # exp 3-5
    tobs = np.array(
        [
            5.000000e-01,
            1.000000e00,
            1.500000e00,
            2.000000e00,
            2.500000e00,
            3.000000e00,
            3.500000e00,
            4.000000e00,
            4.500000e00,
            5.000000e00,
            5.500000e00,
            6.000000e00,
            6.500000e00,
            7.000000e00,
            7.500000e00,
            8.000000e00,
            8.500000e00,
            9.000000e00,
            9.500000e00,
            1.000000e01,
            1.050000e01,
            1.100000e01,
            1.150000e01,
            1.200000e01,
            1.250000e01,
            1.300000e01,
            1.350000e01,
            1.400000e01,
            1.450000e01,
            1.500000e01,
            1.550000e01,
            1.600000e01,
            1.650000e01,
            1.700000e01,
            1.750000e01,
            1.800000e01,
            1.850000e01,
            1.900000e01,
            1.950000e01,
            2.000000e01,
            2.050000e01,
            2.100000e01,
            2.150000e01,
            2.200000e01,
            2.250000e01,
            2.300000e01,
            2.350000e01,
            2.400000e01,
            2.450000e01,
            2.500000e01,
            2.550000e01,
            2.600000e01,
            2.650000e01,
            2.700000e01,
            2.750000e01,
            2.800000e01,
            2.850000e01,
            2.900000e01,
            2.950000e01,
            3.000000e01,
        ]
    )
    cobs = np.array(
        [
            0.000000e00,
            3.388377e-08,
            6.281422e-06,
            2.856976e-05,
            1.986874e-04,
            3.587401e-03,
            2.370469e-02,
            7.600978e-02,
            1.560633e-01,
            2.445066e-01,
            3.263427e-01,
            3.967238e-01,
            4.564205e-01,
            5.074383e-01,
            5.514083e-01,
            5.894555e-01,
            6.223977e-01,
            6.508955e-01,
            6.755202e-01,
            6.966977e-01,
            7.150403e-01,
            7.308672e-01,
            7.445238e-01,
            7.562925e-01,
            7.650260e-01,
            7.614188e-01,
            7.280991e-01,
            6.620336e-01,
            5.799741e-01,
            5.000126e-01,
            4.302048e-01,
            3.710616e-01,
            3.207662e-01,
            2.775880e-01,
            2.403105e-01,
            2.080788e-01,
            1.802293e-01,
            1.561996e-01,
            1.354927e-01,
            1.176657e-01,
            1.023248e-01,
            8.912325e-02,
            7.775795e-02,
            6.796581e-02,
            5.951976e-02,
            5.222469e-02,
            4.591356e-02,
            4.044375e-02,
            3.569377e-02,
            3.156018e-02,
            2.795510e-02,
            2.480402e-02,
            2.204364e-02,
            1.962012e-02,
            1.748770e-02,
            1.560744e-02,
            1.394612e-02,
            1.247540e-02,
            1.117101e-02,
            1.001213e-02,
        ]
    )
    cimobs = np.array(
        [
            0.000000e00,
            1.455238e-09,
            6.848795e-07,
            6.257481e-06,
            2.175876e-05,
            2.257783e-04,
            1.989175e-03,
            8.870710e-03,
            2.457722e-02,
            4.996697e-02,
            8.307971e-02,
            1.210779e-01,
            1.615963e-01,
            2.030308e-01,
            2.443541e-01,
            2.848944e-01,
            3.241952e-01,
            3.619416e-01,
            3.979214e-01,
            4.319304e-01,
            4.640355e-01,
            4.941462e-01,
            5.222760e-01,
            5.484661e-01,
            5.727093e-01,
            5.942768e-01,
            6.103790e-01,
            6.170461e-01,
            6.124074e-01,
            5.977752e-01,
            5.760020e-01,
            5.497769e-01,
            5.210341e-01,
            4.910501e-01,
            4.606765e-01,
            4.305072e-01,
            4.009706e-01,
            3.723769e-01,
            3.449465e-01,
            3.188281e-01,
            2.941140e-01,
            2.708514e-01,
            2.490527e-01,
            2.287037e-01,
            2.097704e-01,
            1.922041e-01,
            1.759460e-01,
            1.609310e-01,
            1.470897e-01,
            1.343512e-01,
            1.226440e-01,
            1.118988e-01,
            1.020471e-01,
            9.302331e-02,
            8.476512e-02,
            7.721333e-02,
            7.031221e-02,
            6.400954e-02,
            5.825654e-02,
            5.300783e-02,
        ]
    )

    rhob = 1.222
    q = 3.975
    D = 5.307

    n = 0.456
    phi = 0.880

    v = q / (phi * n)
    al = D / v

    f = 0.880
    fm = 0.50
    fim = 0.50
    alfa = 0.030
    km = 0.42587
    kim = 0.42587
    km2 = 0.660
    kim2 = 0.660
    lamb = 0.058
    t0 = 9.665
    L = 30

    x = 30
    c0 = 1.0
    inflowbc = "cauchy"
    domain = 2

    c = mpne(
        c0,
        x,
        tobs,
        v,
        al,
        n,
        rhob,
        L=L,
        phi=phi,
        f=f,
        alfa=alfa,
        fm=fm,
        fim=fim,
        km=km,
        kim=kim,
        km2=km2,
        kim2=kim2,
        lamb=lamb,
        lsm1=0.0,
        lim=0.0,
        inflowbc=inflowbc,
        domain=domain,
    )

    c[tobs > t0] += mpne(
        -c0,
        x,
        tobs[tobs > t0] - t0,
        v,
        al,
        n,
        rhob,
        L=L,
        phi=phi,
        f=f,
        alfa=alfa,
        fm=fm,
        fim=fim,
        km=km,
        kim=kim,
        km2=km2,
        kim2=kim2,
        lamb=lamb,
        lsm1=0.0,
        lim=0.0,
        inflowbc=inflowbc,
        domain=domain,
    )

    # immobile
    cim = mpne(
        c0,
        x,
        tobs,
        v,
        al,
        n,
        rhob,
        L=L,
        phi=phi,
        f=f,
        alfa=alfa,
        fm=fm,
        fim=fim,
        km=km,
        kim=kim,
        km2=km2,
        kim2=kim2,
        inflowbc=inflowbc,
        domain=domain,
        lamb=lamb,
        lsm1=0.0,
        lim=0.0,
        output="immobile",
    )

    cim[tobs > t0] += mpne(
        -c0,
        x,
        tobs[tobs > t0] - t0,
        v,
        al,
        n,
        rhob,
        L=L,
        phi=phi,
        f=f,
        alfa=alfa,
        fm=fm,
        fim=fim,
        km=km,
        kim=kim,
        km2=km2,
        kim2=kim2,
        lamb=lamb,
        lsm1=0.0,
        lim=0.0,
        inflowbc=inflowbc,
        domain=domain,
        output="immobile",
    )

    np.testing.assert_array_almost_equal(cobs, c, decimal=4)
    np.testing.assert_array_almost_equal(cimobs, cim, decimal=4)
