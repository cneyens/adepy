from adepy.uniform import finite1, finite3, seminf1, seminf3, pulse1, point1
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
