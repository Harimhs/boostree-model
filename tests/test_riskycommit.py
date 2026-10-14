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
