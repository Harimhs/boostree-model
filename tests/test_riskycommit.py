"""From-scratch booster: learns, explains exactly, and the delay-aware math is right."""

import numpy as np

from riskycommit.engine import Booster, Params, sigmoid


def toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    z = 2 * X[:, 0] - 1.5 * (X[:, 1] > 0.5) + 0.5 * X[:, 2] * X[:, 0]
    y = (rng.random(n) < sigmoid(z)).astype(float)
    return X, y


def test_booster_learns_and_beats_constant():
    X, y = toy()
    m = Booster(Params(n_rounds=60, early_stopping=100)).fit(X[:2000], y[:2000], X[2000:], y[2000:], log=lambda *_: None)
    p = m.predict_proba(X[2000:])
    ll = -np.mean(y[2000:] * np.log(p) + (1 - y[2000:]) * np.log(1 - p))
    base = y[:2000].mean()
    ll0 = -np.mean(y[2000:] * np.log(base) + (1 - y[2000:]) * np.log(1 - base))
    assert ll < 0.8 * ll0


def test_contributions_add_up_to_score():
    X, y = toy()
    m = Booster(Params(n_rounds=30)).fit(X, y, log=lambda *_: None)
    for i in range(5):
        bias, c = m.contributions(X[i])
        assert abs(bias + c.sum() - m.raw(X[i:i + 1])[0]) < 1e-9
    avg = np.mean([np.abs(m.contributions(X[i])[1]) for i in range(200)], axis=0)
    assert avg[3] < avg[0]  # on average, the pure-noise feature matters less than the strongest one


def test_delay_aware_gradient_hessian_math():
    """loss = -log(1 - p q): analytic gradient/hessian match finite differences,
    the raw hessian can go negative, the Fisher version never does and equals p(1-p) at q=1."""
    rng = np.random.default_rng(1)
    saw_negative = False
    for _ in range(200):
        z, q = rng.normal() * 3, rng.uniform(0.05, 1.0)
        L = lambda t: -np.log(1 - q * sigmoid(t))
        p, e = sigmoid(z), 1e-4
        g = p * (1 - p) * q / (1 - p * q)
        h = p * (1 - p) * q * (1 - 2 * p + p * p * q) / (1 - p * q) ** 2
        fisher = q * p * (1 - p) ** 2 / (1 - p * q)
        assert abs(g - (L(z + e) - L(z - e)) / (2 * e)) < 1e-6
        assert abs(h - (L(z + e) - 2 * L(z) + L(z - e)) / e**2) < 1e-4
        assert fisher > 0
        saw_negative |= h < 0
    assert saw_negative
    p = 0.7
    assert abs(1.0 * p * (1 - p) ** 2 / (1 - p) - p * (1 - p)) < 1e-12


def test_tree_and_network_delay_objectives_agree():
    from riskycommit.delay import DelayObjective
    from riskycommit.frozen_boost import delay_aware_obj
    rng = np.random.default_rng(2)
    z, y, q = rng.normal(size=500) * 2, (rng.random(500) < 0.3).astype(float), rng.uniform(0.05, 1, 500)
    g1, h1 = DelayObjective(q, fisher=True)(z, y)
    _, g2, h2 = delay_aware_obj(z, y, q)
    assert np.allclose(g1, g2) and np.allclose(np.maximum(h2, 1e-6), h1)


def test_frozen_weights_never_change_under_boosting():
    from riskycommit.frozen_boost import boost, train_mlp
    rng = np.random.default_rng(3)
    X = rng.normal(size=(1500, 6)); R = X[:, :3].copy()
    y = (rng.random(1500) < sigmoid(2 * X[:, 0] - X[:, 1])).astype(float)
    net = train_mlp(X[:1000], y[:1000], X[1000:1200], y[1000:1200], epochs=5, log=lambda *_: None)
    before = [w.copy() for w in net.W]
    bn, hist = boost(net, X[1200:1400], R[1200:1400], y[1200:1400], X[1400:], R[1400:], y[1400:],
                     ["a", "b", "c"], rounds=3, min_leaf=20, log=lambda *_: None)
    assert all(np.array_equal(a, b) for a, b in zip(before, bn.W))
