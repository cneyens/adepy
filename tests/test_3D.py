from adepy.uniform.threeD import point3, patchi, patchf, pulse3, plume3
import numpy as np
from scipy.stats import multivariate_normal


def test_point3_shape():
    x, y = np.meshgrid([5.0, 6.0, 7.0], [11.0, 12.0, 13.0])

    c = point3(
        c0=1.0,
        x=x,
        y=y,
        z=0.0,
        t=100.0,
        v=0.1,
        n=0.25,
        al=1,
        ah=1,
        av=1,
        Q=1.0,
        xc=0,
        yc=0,
        zc=0,
    )

    assert c.shape == (len(x), len(y))

    t = [100, 200, 300]
    c = point3(
        c0=1.0,
        x=10,
        y=0.0,
        z=0.0,
        t=t,
        v=0.1,
        n=0.25,
        al=1,
        ah=1,
        av=1,
        Q=1.0,
        xc=0,
        yc=0,
        zc=0,
    )

    assert c.shape == (len(t),)


def test_point3():
    n = 0.25
    v = 0.1
    al = 0.6
    ah = 0.03
    av = 0.006
    c0 = 1.0
    Q = 1.0
    wc = (0, 0, 0)
    t = 300
    x = 10
    y = 0
    z = 0

    c = point3(c0, x, y, z, t, v, n, al, ah, av, Q, wc[0], wc[1], wc[2])

    np.testing.assert_approx_equal(c[0], 23.72061)

    lamb = 0.05
    c = point3(c0, x, y, z, t, v, n, al, ah, av, Q, wc[0], wc[1], wc[2], lamb=lamb)

    np.testing.assert_approx_equal(c[0], 0.422971, significant=6)


def test_patchf():
    c = patchf(
        c0=1000,
        x=1050,
        y=1000,
        z=50,
        t=3000,
        v=1,
        al=200,
        ah=60,
        av=10,
        w=3000,
        h=100,
        y1=400,
        y2=2000,
        z1=50,
        z2=100,
    )  # 468.69839

    np.testing.assert_approx_equal(c[0], 468.69839)


def test_patchf_shape():
    x, y = np.meshgrid([1050, 1075, 1080], [1000, 1002, 1004])
    lamb = np.log(2) / (28 * 365)

    c = patchf(
        100,
        x,
        y,
        1750,
        t=3652.5,
        v=1,
        al=100,
        ah=20,
        av=20,
        y1=900,
        y2=2100,
        z1=1350,
        z2=1650,
        w=4000,
        h=2000,
    )

    assert c.shape == (len(x), len(y))

    t = [1000, 2000, 3000]
    c = patchf(
        100,
        1050,
        1000,
        1750,
        t=t,
        v=1,
        al=100,
        ah=20,
        av=20,
        y1=900,
        y2=2100,
        z1=1350,
        z2=1650,
        w=4000,
        h=2000,
        lamb=lamb,
    )

    assert c.shape == (len(t),)


def test_patchi():
    c = patchi(
        100,
        1050,
        1000,
        1750,
        t=3652.5,
        v=1,
        al=100,
        ah=20,
        av=20,
        y1=900,
        y2=2100,
        z1=1350,
        z2=1650,
        lamb=6.78e-5,
    )  # 17.828021

    np.testing.assert_approx_equal(c[0], 17.828021, significant=6)


def test_patchi_shape():
    x, y = np.meshgrid([1050, 1075, 1080], [1000, 1002, 1004])
    lamb = np.log(2) / (28 * 365)

    c = patchi(
        100,
        x,
        y,
        1750,
        t=3652.5,
        v=1,
        al=100,
        ah=20,
        av=20,
        y1=900,
        y2=2100,
        z1=1350,
        z2=1650,
        lamb=lamb,
    )

    assert c.shape == (len(x), len(y))

    t = [1000, 2000, 3000]
    c = patchi(
        100,
        1050,
        1000,
        1750,
        t=t,
        v=1,
        al=100,
        ah=20,
        av=20,
        y1=900,
        y2=2100,
        z1=1350,
        z2=1650,
        lamb=lamb,
    )

    assert c.shape == (len(t),)


def test_pulse3():
    m0 = 15.0
    v = 0.05
    n = 0.2
    al = 1.2
    ah = al / 3
    av = ah
    x = [5.0, 10.0]
    y = [2.0, 3.0]
    z = [0.0, 0.0]
    xc = 1.0
    yc = 0.0
    zc = 0.5
    t = 50.0

    x, y = np.meshgrid(np.linspace(-2.5, 10, 100), np.linspace(-5, 5, 100))
    z = 0.0
    c = pulse3(m0, x, y, z, t, v, n, al, ah, av, xc, yc, zc)

    # trivariate gaussian probability density function
    sigX = np.sqrt(2 * al * v * t)
    sigY = np.sqrt(2 * ah * v * t)
    sigZ = np.sqrt(2 * ah * v * t)

    mu = v * t
    cov = [[sigX**2, 0, 0], [0, sigY**2, 0], [0, 0, sigZ**2]]
    dist = multivariate_normal(mean=[mu, 0.0, 0.0], cov=cov)
    cpdf = m0 / n * dist.pdf(np.dstack([x - xc, y - yc, z * np.ones(y.shape) - zc]))
    np.testing.assert_array_almost_equal(c, cpdf, decimal=6)


def test_pulse3_shape():
    x, y = np.meshgrid([5.0, 6.0, 7.0], [11.0, 12.0, 13.0])

    c = pulse3(m0=1.0, x=x, y=y, z=0, t=100.0, v=0.1, n=0.2, al=1, ah=1, av=1)

    assert c.shape == (len(x), len(y))

    t = [100, 200, 300]
    c = pulse3(m0=1.0, x=10, y=11, z=0, t=t, v=0.1, n=0.2, al=1, ah=1, av=1, lamb=0.05)

    assert c.shape == (len(t),)


def test_plume3_shape():
    dx = 10
    dy = 10
    dz = 2
    x, y, z = np.meshgrid(
        np.arange(0, 101, dx), np.arange(0, 101, dy), np.arange(-1, -21, -dz)
    )
    v = 0.05
    n = 0.20
    al = 1
    ah = 0.1
    av = 0.01

    c = point3(
        1.0, x, y, z, 5 * 365, v, n, Q=1.0, al=al, ah=ah, av=av, xc=0.0, yc=0.0, zc=0.0
    )

    m0 = c * dx * dy * dz * n
    xn, yn, zn = np.meshgrid(
        np.arange(0, 101 * 2, dx / 2),
        np.arange(0, 101, dy / 2),
        np.arange(0, -21, -dz / 2),
    )
    cn = plume3(m0, xn, yn, zn, 10 * 365, v, n, al, ah, av, x, y, z)
    assert cn.shape == xn.shape

    tn = np.linspace(0.1, 10 * 365, 10)
    cn = plume3(m0, 25, 0, 0, tn, v, n, al, ah, av, x, y, z)
    assert cn.shape == tn.shape
