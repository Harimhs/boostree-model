"""Gradient boosting from scratch (NumPy only).

Second-order ("Newton") boosting, the same idea XGBoost uses:
  * every round, the objective gives each example a gradient g and hessian h
  * a tree is grown by histogram splits that maximise
        gain = 1/2 [ GL^2/(HL+lam) + GR^2/(HR+lam) - G^2/(H+lam) ] - gamma
  * each leaf gets the Newton step  w = -G / (H + lam)

The objective is pluggable: anything that returns (g, h) works. That is where
the delay-aware loss goes next.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------- objectives
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def logistic_obj(z, y):
    """Standard log-loss: g = p - y, h = p(1-p)."""
    p = sigmoid(z)
    return p - y, np.maximum(p * (1 - p), 1e-12)


# ---------------------------------------------------------------- binning
class Binner:
    """Quantile bins per feature, so splits are searched over <= n_bins edges."""

    def __init__(self, n_bins: int = 64):
        self.n_bins = n_bins
        self.edges: list[np.ndarray] = []

    def fit(self, X: np.ndarray) -> "Binner":
        self.edges = []
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for j in range(X.shape[1]):
            e = np.unique(np.quantile(X[:, j], qs))
            self.edges.append(e)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape, dtype=np.uint8)
        for j, e in enumerate(self.edges):
            out[:, j] = np.searchsorted(e, X[:, j], side="right")
        return out


# ---------------------------------------------------------------- trees
@dataclass
class Tree:
    feature: list = field(default_factory=list)   # -1 for leaf
    bin: list = field(default_factory=list)       # go left if x_bin <= bin
    left: list = field(default_factory=list)
    right: list = field(default_factory=list)
    value: list = field(default_factory=list)     # Newton value at every node (used for explanations)

    def add(self, value):
        for lst, v in ((self.feature, -1), (self.bin, 0), (self.left, -1), (self.right, -1),
                       (self.value, value)):
            lst.append(v)
        return len(self.value) - 1

    def arrays(self):
        return (np.array(self.feature), np.array(self.bin), np.array(self.left),
                np.array(self.right), np.array(self.value, dtype=float))


@dataclass
class Params:
    n_rounds: int = 400
    learning_rate: float = 0.05
    max_depth: int = 5
    reg_lambda: float = 1.0
    gamma: float = 0.0
    min_child_hess: float = 1.0
    subsample: float = 0.8
    early_stopping: int = 40
    seed: int = 0


class Booster:
    def __init__(self, params: Params | None = None, objective=logistic_obj):
        self.p = params or Params()
        self.objective = objective
        self.trees: list[Tree] = []
        self.base = 0.0
        self.binner: Binner | None = None
        self.history: list[dict] = []

    # -- growing one tree on binned data
    def _grow(self, B, g, h, rows):
        p = self.p
        nb = self.binner.n_bins
        nf = B.shape[1]
        tree = Tree()
        G, H = g[rows].sum(), h[rows].sum()
        root = tree.add(-G / (H + p.reg_lambda))
        frontier = [(root, rows)]
        for _depth in range(p.max_depth):
            nxt = []
            for node, idx in frontier:
                if len(idx) < 2:
                    continue
                gi, hi, Bi = g[idx], h[idx], B[idx]
                G, H = gi.sum(), hi.sum()
                best = (0.0, -1, -1)
                parent = G * G / (H + p.reg_lambda)
                for j in range(nf):
                    gh = np.bincount(Bi[:, j], weights=gi, minlength=nb)
                    hh = np.bincount(Bi[:, j], weights=hi, minlength=nb)
                    GL, HL = np.cumsum(gh)[:-1], np.cumsum(hh)[:-1]
                    GR, HR = G - GL, H - HL
                    ok = (HL >= p.min_child_hess) & (HR >= p.min_child_hess)
                    if not ok.any():
                        continue
                    gain = 0.5 * (GL**2 / (HL + p.reg_lambda) + GR**2 / (HR + p.reg_lambda) - parent) - p.gamma
                    gain = np.where(ok, gain, -np.inf)
                    k = int(np.argmax(gain))
                    if gain[k] > best[0]:
                        best = (float(gain[k]), j, k)
                if best[1] < 0:
                    continue
                _, j, k = best
                go_left = Bi[:, j] <= k
                li, ri = idx[go_left], idx[~go_left]
                lnode = tree.add(-g[li].sum() / (h[li].sum() + p.reg_lambda))
                rnode = tree.add(-g[ri].sum() / (h[ri].sum() + p.reg_lambda))
                tree.feature[node], tree.bin[node] = j, k
                tree.left[node], tree.right[node] = lnode, rnode
                nxt += [(lnode, li), (rnode, ri)]
            frontier = nxt
            if not frontier:
                break
        return tree

    @staticmethod
    def _tree_predict(arrs, B):
        feat, bins, left, right, value = arrs
        node = np.zeros(len(B), dtype=int)
        while True:
            f = feat[node]
            active = f >= 0
            if not active.any():
                break
            a = np.where(active)[0]
            goes_left = B[a, f[a]] <= bins[node[a]]
            node[a] = np.where(goes_left, left[node[a]], right[node[a]])
        return value[node]

    def fit(self, X, y, X_val=None, y_val=None, log=print):
        p = self.p
        rng = np.random.default_rng(p.seed)
        self.binner = Binner().fit(X)
        B = self.binner.transform(X)
        Bv = self.binner.transform(X_val) if X_val is not None else None
        prior = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
        self.base = float(np.log(prior / (1 - prior)))
        z = np.full(len(y), self.base)
        zv = np.full(len(y_val), self.base) if Bv is not None else None
        best, best_round = np.inf, 0
        self._arrays = []
        for r in range(p.n_rounds):
            g, h = self.objective(z, y)
            rows = np.where(rng.random(len(y)) < p.subsample)[0]
            tree = self._grow(B, g, h, rows)
            arrs = tree.arrays()
            self.trees.append(tree)
            self._arrays.append(arrs)
            z += p.learning_rate * self._tree_predict(arrs, B)
            if zv is not None:
                zv += p.learning_rate * self._tree_predict(arrs, Bv)
                pv = sigmoid(zv)
                vloss = float(-np.mean(y_val * np.log(pv + 1e-12) + (1 - y_val) * np.log(1 - pv + 1e-12)))
                self.history.append({"round": r + 1, "val_logloss": vloss})
                if vloss < best - 1e-6:
                    best, best_round = vloss, r + 1
                elif r + 1 - best_round >= p.early_stopping:
                    break
        if zv is not None and best_round:
            self.trees, self._arrays = self.trees[:best_round], self._arrays[:best_round]
        log(f"trees kept: {len(self.trees)}")
        return self

    def raw(self, X):
        B = self.binner.transform(X)
        z = np.full(len(X), self.base)
        for arrs in self._arrays:
            z += self.p.learning_rate * self._tree_predict(arrs, B)
        return z

    def predict_proba(self, X):
        return sigmoid(self.raw(X))

    def contributions(self, x):
        """Per-feature contribution in log-odds for one example (path attribution:
        every split hands the change in node value to the feature it split on).
        bias + sum(contrib) == raw score exactly."""
        b = self.binner.transform(x.reshape(1, -1))[0]
        contrib = np.zeros(len(b))
        bias = self.base
        lr = self.p.learning_rate
        for feat, bins, left, right, value in self._arrays:
            node = 0
            bias += lr * value[0]
            while feat[node] >= 0:
                f = feat[node]
                child = left[node] if b[f] <= bins[node] else right[node]
                contrib[f] += lr * (value[child] - value[node])
                node = child
        return bias, contrib

    # -- save / load (plain JSON so anyone can read it)
    def to_json(self, feature_names):
        return {
            "format": "riskycommit-boost-v1",
            "features": feature_names,
            "base": self.base,
            "learning_rate": self.p.learning_rate,
            "bin_edges": [e.tolist() for e in self.binner.edges],
            "trees": [{"feature": t.feature, "bin": t.bin, "left": t.left, "right": t.right,
                       "value": t.value} for t in self.trees],
        }
