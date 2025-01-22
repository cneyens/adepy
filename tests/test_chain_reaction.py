import numpy as np
from adepy.uniform import point2, seminf1
from adepy.chain_reaction import chain_reaction


def test_chain_1d():
    lamb = [0.03, 0.02, 0.015, 0.01, 0.011, 0.006, 0.002, 0.0015, 0.003, 0.002]
    stoi = [0, 0.79, 0.50, 0.45, 0.74, 0.64, 0.37, 0.51, 0.51, 0.45]
    ancestry = {
        0: -1,  # species: parent
        1: 0,
        2: 1,
        3: 2,
        4: 1,
        5: 4,
        6: 2,
        7: 6,
        8: 3,
        9: 5,
    }

    x = np.linspace(0, 500, 100)  # ft
    t = 360  # d
    c0 = np.zeros(len(lamb))  # all c0 are zero except for the first species
    c0[0] = 100
    al = 10
    v = 0.5  # ft/d

    c = chain_reaction(ancestry, lamb, stoi, c0, seminf1, x=x, t=t, al=al, v=v)

    assert len(c) == len(lamb)
    assert c[0].shape == (len(x),)
    np.testing.assert_approx_equal(c[0][10], 11.870861, significant=6)
    np.testing.assert_approx_equal(c[4][10], 13.652508, significant=6)
    np.testing.assert_approx_equal(c[9][10], 3.7739144, significant=6)


def test_chain_2d():
    lamb = [0.03, 0.02, 0.015, 0.01, 0.011, 0.006, 0.002, 0.0015, 0.003, 0.002]
    stoi = [0, 0.79, 0.50, 0.45, 0.74, 0.64, 0.37, 0.51, 0.51, 0.45]
    ancestry = {
        0: -1,  # species: parent
        1: 0,
        2: 1,
        3: 2,
        4: 1,
        5: 4,
        6: 2,
        7: 6,
        8: 3,
        9: 5,
    }

    x, y = np.meshgrid(np.linspace(0, 500, 100), np.linspace(0, 300, 100))
    v = 0.5
    al = 10
    ah = 3
    n = 0.2
    xc = 155
    yc = 155
    Qa = 1 / 10
    t = 360

    c0 = np.zeros(len(lamb))  # all c0 are zero except for the first species
    c0[0] = 100

    c = chain_reaction(
        ancestry,
        lamb,
        stoi,
        c0,
        point2,
        x=x,
        y=y,
        t=t,
        v=v,
        n=n,
        al=al,
        ah=ah,
        Qa=Qa,
        xc=xc,
        yc=yc,
    )

    assert len(c) == len(lamb)
    assert c[0].shape == (len(x), len(y))

    t = np.linspace(0, 360, 100)
    x = 200
    y = 150

    c = chain_reaction(
        ancestry,
        lamb,
        stoi,
        c0,
        point2,
        x=x,
        y=y,
        t=t,
        v=v,
        al=al,
        ah=ah,
        n=n,
        Qa=Qa,
        xc=xc,
        yc=yc,
    )
    assert len(c) == len(lamb)
    assert c[0].shape == (len(t),)
    np.testing.assert_approx_equal(c[0][-1], 0.237167, significant=6)
    np.testing.assert_approx_equal(c[4][-1], 0.301282, significant=6)
    np.testing.assert_approx_equal(c[9][-1], 0.079441, significant=5)
