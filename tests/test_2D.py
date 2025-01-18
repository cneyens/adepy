from adepy.uniform import point2, stripf, stripi, gauss
import numpy as np


def test_point2_shape():
    x, y = np.meshgrid([5.0, 6.0, 7.0], [11.0, 12.0, 13.0])

    c = point2(
        c0=1.0,
        x=x,
        y=y,
        t=100.0,
        v=0.1,
        n=0.25,
        al=1,
        ah=1,
        Qa=1.0,
        xc=0,
        yc=0,
    )

    assert c.shape == (len(x), len(y))

    t = [100, 200, 300]
    c = point2(
        c0=1.0,
        x=10,
        y=0.0,
        t=t,
        v=0.1,
        n=0.25,
        al=1,
        ah=1,
        Qa=1.0,
        xc=0,
        yc=0,
        lamb=0.05,
    )

    assert c.shape == (len(t),)


def test_point2():
    c = point2(
        1000, 50, 475, [25, 50, 80], v=2, al=30, ah=6, n=0.25, Qa=12.5, xc=0, yc=500
    )

    np.testing.assert_approx_equal(c[0], 63.23244)


def test_stripf_shape():
    x, y = np.meshgrid([5.0, 6.0, 7.0], [11.0, 12.0, 13.0])

    c = stripf(
        c0=1.0,
        x=x,
        y=y,
        t=100.0,
        v=0.1,
        al=1,
        ah=1,
        y1=9,
        y2=15,
        w=20,
    )

    assert c.shape == (len(x), len(y))

    t = [100, 200, 300]
    c = stripf(c0=1.0, x=10, y=11, t=t, v=0.1, al=1, ah=1, y1=9, y2=15, w=20, lamb=0.05)

    assert c.shape == (len(t),)


def test_stripf():
    c = stripf(
        1000, [750, 1350], [100, 500], 1500, 1, 200, 60, 400, 2000, 3000
    )  # 144.48110, 423.82109
    np.testing.assert_approx_equal(c[0], 144.48110)
    np.testing.assert_approx_equal(c[1], 423.82109)


def test_stripi_shape():
    x, y = np.meshgrid([5.0, 6.0, 7.0], [11.0, 12.0, 13.0])

    c = stripi(
        c0=1.0,
        x=x,
        y=y,
        t=100.0,
        v=0.1,
        al=1,
        ah=1,
        y1=9,
        y2=15,
    )

    assert c.shape == (len(x), len(y))

    t = [100, 200, 300]
    c = stripi(c0=1.0, x=10, y=11, t=t, v=0.1, al=1, ah=1, y1=9, y2=15, lamb=0.05)

    assert c.shape == (len(t),)


def test_stripi():
    c = stripi(
        40,
        x=[2500, 500],
        y=[750, 600],
        t=1826,
        v=1.42,
        al=100 / 1.42,
        ah=20 / 1.42,
        y1=635,
        y2=865,
    )  # 8.85856

    np.testing.assert_approx_equal(c[0], 8.85856, significant=5)


def test_gauss_shape():
    x, y = np.meshgrid([5.0, 6.0, 7.0], [11.0, 12.0, 13.0])

    c = gauss(
        c0=1.0,
        x=x,
        y=y,
        t=100.0,
        v=0.1,
        al=1,
        ah=1,
        yc=12,
        sigma=10,
    )

    assert c.shape == (len(x), len(y))

    t = [100, 200, 300]
    c = gauss(c0=1.0, x=10, y=11, t=t, v=0.1, al=1, ah=1, yc=12, sigma=10, lamb=0.05)

    assert c.shape == (len(t),)


def test_gauss():
    c = gauss(1000, [1000, 1050], [250, 300], 300, 4, 37.5, 7.5, 450, 130)  # 304.096099

    np.testing.assert_approx_equal(c[0], 304.096099)
