"""A small neural net trained from scratch, then FROZEN, then adapted only by
booster trees that apply gradient + hessian (Newton) logic to its weights.

Network:   x (dense features) -> W1 -> ReLU -> W2 -> ReLU -> W3 -> risk logit

One booster round (the boostree idea):
  1. For each weight matrix W_k, take the gradient of the loss on recent data
     and its best rank-1 direction D_k = u v^T (top singular pair).
  2. Moving W_k along D_k by t changes every commit's logit; per commit we get
         g_i = dL/dt = g_z * z'          h_i = d2L/dt2 ~= h_z * z'^2
     where (g_z, h_z) come from the OBJECTIVE. Plain log-loss today; the
     delay-aware objective (with its Fisher hessian) plugs in here.
  3. Pick the matrix with the largest Newton gain n*mean(g)^2 / (mean(h)+lam).
  4. Grow a small tree on READABLE commit features; each leaf gets its own
     Newton step w = -G/(H+lam). Commits in that leaf see W_k + w * u v^T.
  5. Keep the booster only if held-out loss improves (line search).
The original weights never change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from riskycommit.engine import sigmoid


# ------------------------------------------------------------- objectives
def logloss_obj(z, y, q=None):
    """Plain log-loss in logit space: loss, g_z, h_z."""
    p = sigmoid(z)
    loss = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    return loss, p - y, np.maximum(p * (1 - p), 1e-12)


def delay_aware_obj(z, y, q):
    """Incomplete labels: y=1 means a bug WAS found; y=0 means none found YET.
    q = chance a bug would have surfaced by now (from the delay curve).
    Returns the Fisher hessian, which is always positive (raw one can be < 0)."""
    p = sigmoid(z)
    pq = np.clip(p * q, 1e-12, 1 - 1e-12)
    loss = np.where(y == 1, -np.log(p + 1e-12), -np.log(1 - pq))
    g = np.where(y == 1, p - 1, p * (1 - p) * q / (1 - pq))
    h = np.where(y == 1, p * (1 - p), q * p * (1 - p) ** 2 / (1 - pq))
    return loss, g, np.maximum(h, 1e-12)


# ------------------------------------------------------------- small MLP
class MLP:
    def __init__(self, d, sizes=(64, 32), seed=0):
        rng = np.random.default_rng(seed)
        dims = [d, *sizes, 1]
        self.W = [rng.normal(0, np.sqrt(2 / dims[i]), (dims[i + 1], dims[i])) for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self.boosters: list[Booster] = []

    def copy(self):
        m = MLP.__new__(MLP)
        m.W = [w.copy() for w in self.W]
        m.b = [b.copy() for b in self.b]
        m.boosters = list(self.boosters)
        return m

    def forward(self, X, R=None, extra=None):
        """Logits. R = readable features (to route boosters). extra = (layer, u, v, w_per_example)
        adds one trial perturbation without storing it."""
        acts, h = [], X
        n_layers = len(self.W)
        for k in range(n_layers):
            a = h @ self.W[k].T + self.b[k]
            for bo in self.boosters:
                if bo.layer == k:
                    a = a + (bo.leaf_weights(R) * (h @ bo.v))[:, None] * bo.u[None, :]
            if extra is not None and extra[0] == k:
                _, u, v, w = extra
                a = a + (w * (h @ v))[:, None] * u[None, :]
            acts.append((h, a))
            h = np.maximum(a, 0) if k < n_layers - 1 else a
        return h[:, 0], acts

    def grads(self, X, y, R=None, obj=logloss_obj, q=None, sw=None):
        """Mean-loss gradient for every weight matrix (backprop, boosters included)."""
        z, acts = self.forward(X, R)
        _, gz, _ = obj(z, y, q)
        if sw is not None:
            gz = gz * sw
        delta = (gz / len(z))[:, None]
        out = [None] * len(self.W)
        for k in range(len(self.W) - 1, -1, -1):
            h_in, a = acts[k]
            out[k] = delta.T @ h_in
            if k:
                delta = (delta @ self.W[k]) * (acts[k - 1][1] > 0)
        return out


def train_mlp(X, y, Xv, yv, epochs=60, lr=1e-3, batch=256, seed=0, patience=6, log=print):
    """Ordinary training from scratch with Adam + early stopping (this is the base model)."""
    rng = np.random.default_rng(seed)
    m = MLP(X.shape[1], seed=seed)
    params = m.W + m.b
    mom = [np.zeros_like(p) for p in params]
    vel = [np.zeros_like(p) for p in params]
    t, best, best_state, bad = 0, np.inf, None, 0
    for ep in range(epochs):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            z, acts = m.forward(X[idx])
            p = sigmoid(z)
            delta = ((p - y[idx]) / len(idx))[:, None]
            gW, gb = [None] * len(m.W), [None] * len(m.W)
            for k in range(len(m.W) - 1, -1, -1):
                h_in, a = acts[k]
                gW[k], gb[k] = delta.T @ h_in, delta.sum(0)
                if k:
                    delta = (delta @ m.W[k]) * (acts[k - 1][1] > 0)
            t += 1
            for i, g in enumerate(gW + gb):
                mom[i] = 0.9 * mom[i] + 0.1 * g
                vel[i] = 0.999 * vel[i] + 0.001 * g * g
                step = lr * (mom[i] / (1 - 0.9**t)) / (np.sqrt(vel[i] / (1 - 0.999**t)) + 1e-8)
                params[i] -= step
        vl = float(logloss_obj(m.forward(Xv)[0], yv)[0].mean())
        if vl < best - 1e-5:
            best, bad, best_state = vl, 0, ([w.copy() for w in m.W], [b.copy() for b in m.b])
        else:
            bad += 1
            if bad >= patience:
                break
    m.W, m.b = best_state
    log(f"base net: {ep + 1} epochs, best val loss {best:.4f}")
    return m


# ------------------------------------------------------------- booster trees
@dataclass
class Booster:
    layer: int
    u: np.ndarray
    v: np.ndarray
    tree: dict
    gain: float = 0.0
    record: dict = field(default_factory=dict)

    def leaf_weights(self, R):
        return assign(self.tree, R)


def assign(tree, R):
    if "w" in tree:
        return np.full(len(R), tree["w"])
    go = R[:, tree["f"]] <= tree["thr"]
    out = np.empty(len(R))
    out[go] = assign(tree["left"], R[go])
    out[~go] = assign(tree["right"], R[~go])
    return out


def grow(R, g, h, idx, depth, max_depth, min_leaf, lam, lr, names, cands):
    G, H = g[idx].sum(), h[idx].sum()
    leaf = {"w": float(-lr * G / (H + lam)), "n": int(len(idx)), "G": float(G), "H": float(H)}
    if depth >= max_depth or len(idx) < 2 * min_leaf:
        return leaf
    parent = G * G / (H + lam)
    best = (0.0, None, None)
    for f in range(R.shape[1]):
        col = R[idx, f]
        for thr in cands[f]:
            m = col <= thr
            nl = int(m.sum())
            if nl < min_leaf or len(idx) - nl < min_leaf:
                continue
            GL, HL = g[idx][m].sum(), h[idx][m].sum()
            gain = 0.5 * (GL**2 / (HL + lam) + (G - GL) ** 2 / (H - HL + lam) - parent)
            if gain > best[0]:
                best = (gain, f, thr)
    if best[1] is None:
        return leaf
    _, f, thr = best
    m = R[idx, f] <= thr
    return {"f": int(f), "thr": float(thr), "name": names[f], "gain": float(best[0]),
            "left": grow(R, g, h, idx[m], depth + 1, max_depth, min_leaf, lam, lr, names, cands),
            "right": grow(R, g, h, idx[~m], depth + 1, max_depth, min_leaf, lam, lr, names, cands)}


def leaves(tree, path=()):
    if "w" in tree:
        return [{"rule": " and ".join(path) or "all commits", "weight": round(tree["w"], 4), "commits": tree["n"]}]
    nm = tree["name"]
    return (leaves(tree["left"], path + (f"{nm} <= {tree['thr']:g}",)) +
            leaves(tree["right"], path + (f"{nm} > {tree['thr']:g}",)))


def boost(net, X, R, y, Xv, Rv, yv, names, obj=logloss_obj, q=None, qv=None, rounds=30,
          max_depth=2, min_leaf=150, lam=1.0, lr=1.0, log=print, on_round=None):
    """Add booster trees to a FROZEN net. Returns the boosted copy and per-round records."""
    net = net.copy()
    cands = [np.unique(np.quantile(R[:, f], np.linspace(0.05, 0.95, 15))) for f in range(R.shape[1])]
    eps = 1e-3
    history = []
    val_loss = lambda m: float(obj(m.forward(Xv, Rv)[0], yv, qv)[0].mean())
    current = val_loss(net)
    for r in range(rounds):
        gW = net.grads(X, y, R, obj, q)
        z0, _ = net.forward(X, R)
        _, gz, hz = obj(z0, y, q)
        cand = []
        for k, G in enumerate(gW):
            U, S, Vt = np.linalg.svd(G, full_matrices=False)
            u, v = U[:, 0], Vt[0]
            ones = np.ones(len(X))
            zp, _ = net.forward(X, R, extra=(k, u, v, eps * ones))
            zm, _ = net.forward(X, R, extra=(k, u, v, -eps * ones))
            dz = (zp - zm) / (2 * eps)                     # how each commit's logit moves
            g, h = gz * dz, hz * dz * dz                   # chain rule (Gauss-Newton)
            gain = len(g) * g.mean() ** 2 / (h.mean() + lam / len(g))
            cand.append((gain, k, u, v, g, h))
        cand.sort(key=lambda c: -c[0])
        kept = False
        for gain, k, u, v, g, h in cand[:2]:
            tree = grow(R, g, h, np.arange(len(R)), 0, max_depth, min_leaf, lam, lr, names, cands)
            for shrink in (1.0, 0.5, 0.25, 0.125):
                t = scale(tree, shrink)
                trial = net.copy()
                trial.boosters = net.boosters + [Booster(k, u, v, t, float(gain))]
                vl = val_loss(trial)
                if vl < current - 1e-6:
                    net, current, kept = trial, vl, True
                    history.append({"round": r + 1, "layer": k, "gain": round(float(gain), 3),
                                    "val_loss": round(vl, 5), "leaves": leaves(t)})
                    if on_round is not None:
                        on_round(net, history[-1])
                    break
            if kept:
                break
        if not kept:
            log(f"stopped at round {r + 1}: no booster improved held-out loss")
            break
    return net, history


def scale(tree, s):
    if "w" in tree:
        return {**tree, "w": tree["w"] * s}
    return {**tree, "left": scale(tree["left"], s), "right": scale(tree["right"], s)}


def finetune(net, X, y, Xv, yv, epochs=20, lr=3e-4, batch=256, seed=1):
    """Baseline: ordinary SGD fine-tuning of ALL weights on the recent data."""
    rng = np.random.default_rng(seed)
    m = net.copy()
    best, state = float(logloss_obj(m.forward(Xv)[0], yv)[0].mean()), ([w.copy() for w in m.W], [b.copy() for b in m.b])
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            z, acts = m.forward(X[idx])
            delta = ((sigmoid(z) - y[idx]) / len(idx))[:, None]
            for k in range(len(m.W) - 1, -1, -1):
                h_in, a = acts[k]
                gW, gb = delta.T @ h_in, delta.sum(0)
                if k:
                    delta = (delta @ m.W[k]) * (acts[k - 1][1] > 0)
                m.W[k] -= lr * 10 * gW
                m.b[k] -= lr * 10 * gb
        vl = float(logloss_obj(m.forward(Xv)[0], yv)[0].mean())
        if vl < best - 1e-6:
            best, state = vl, ([w.copy() for w in m.W], [b.copy() for b in m.b])
    m.W, m.b = state
    return m
