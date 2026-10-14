"""Step 3: gain-guided booster trees inside a frozen encoder.

Each round works like one XGBoost tree, but the "tree" adjusts a weight
matrix inside the encoder instead of predicting a value:

1. Gradients. On a random sample of training issues, compute the gradient of
   the loss for every candidate weight matrix (two per layer: attention output
   and feed-forward output). Its top singular vectors u, v give the single best
   rank-1 direction D = u v^T to move that matrix.
2. Per-example g and h. Nudge each candidate by +eps*D and -eps*D and measure
   every example's loss. Finite differences give each example's own gradient
   g_i and curvature h_i along D: exactly XGBoost's per-example statistics.
3. Pick a layer. Score each candidate with XGBoost's gain
   G^2 / (H + lambda). Only candidates whose average g is clearly non-zero
   (t-statistic >= t_min) may win. If none do, training stops by itself.
4. Grow a small tree. Split the examples on readable features (e.g. "body
   mentions an error") when the two sides pull in clearly different
   directions (Welch t-test). This is how one booster can push one group of
   issues one way and another group the other way.
5. Leaf weights. Each leaf gets XGBoost's Newton step -G / (H + lambda),
   scaled by a learning rate. The booster adds  w_leaf(x) * (u v^T)  to the
   weight matrix. A line search on the sample keeps only boosters that lower
   the loss.

No optimizer and no epochs: every number comes from g, h and the gain formula.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

FEATURE_NAMES = [
    "title starts with a tag like [X] or feat:",
    "text mentions an error, exception or traceback",
    "text has a code block",
    "text uses a form template (### headings or checkboxes)",
    "description length (characters)",
    "title length (words)",
    "meaning axis 1",
    "meaning axis 2",
    "meaning axis 3",
    "meaning axis 4",
]
BINARY_FEATURES = {0, 1, 2, 3}

_TAG = re.compile(r"^\s*(\[[^\]]{1,30}\]|[A-Za-z]{2,12}(\([^)]*\))?\s*:)")
_ERROR = re.compile(r"(error|exception|traceback|stack trace|panic|segfault|failed)", re.I)
_TEMPLATE = re.compile(r"(^|\n)\s*(#{2,4}\s|- \[[ x]\])")


def text_features(texts: list[str]) -> np.ndarray:
    rows = []
    for text in texts:
        title, _, body = text.partition("\n")
        rows.append([
            float(bool(_TAG.search(title))),
            float(bool(_ERROR.search(text))),
            float("```" in text),
            float(bool(_TEMPLATE.search(body))),
            float(len(body)),
            float(len(title.split())),
        ])
    return np.asarray(rows, dtype=np.float64)


# --------------------------------------------------------------------------
# Encoder wrapper
# --------------------------------------------------------------------------

class BoostedLinear(nn.Module):
    """A frozen Linear layer plus any number of rank-1 boosters.

    Booster k adds  w_k[example] * u_k (v_k . x)  to the output, i.e. it adds
    w_k * u_k v_k^T to the weight matrix, with a per-example scale w_k chosen
    by that booster's tree.
    """

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.us: list[torch.Tensor] = []
        self.vs: list[torch.Tensor] = []
        self.batch_w: list[torch.Tensor] = []  # per-example scales for the current batch
        self.probe: tuple[torch.Tensor, torch.Tensor, float] | None = None

    def forward(self, x):
        y = self.base(x)
        shape = (-1,) + (1,) * (x.dim() - 1)
        for u, v, w in zip(self.us, self.vs, self.batch_w):
            y = y + w.view(shape) * (x @ v).unsqueeze(-1) * u
        if self.probe is not None:
            u, v, s = self.probe
            y = y + s * (x @ v).unsqueeze(-1) * u
        return y


def candidate_modules(model: nn.Module) -> dict[str, tuple[nn.Module, str]]:
    """BERT-style encoders: per layer, the attention output and FFN output."""
    found = {}
    for i, layer in enumerate(model.encoder.layer):
        found[f"layer {i + 1} attention"] = (layer.attention.output, "dense")
        found[f"layer {i + 1} feed-forward"] = (layer.output, "dense")
    return found


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return nn.functional.normalize(pooled, dim=-1)


# --------------------------------------------------------------------------
# Trees over readable features
# --------------------------------------------------------------------------

def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float((a.mean() - b.mean()) / math.sqrt(va + vb + 1e-12))


def _stats(g, h, c):
    cw = c.sum()
    return (c * g).sum() / cw, (c * h).sum() / cw


def leaf_weight(g, h, c, lam, lr, w_max):
    mg, mh = _stats(g, h, c)
    w = -lr * mg / (max(mh, 0.0) + lam)
    return float(np.clip(w, -w_max, w_max))


def grow_tree(feats, g, h, c, idx, depth, cfg, thresholds):
    """Greedy XGBoost-style tree; splits must pass a Welch t-test on g."""
    node = {"n": int(len(idx))}
    if depth < cfg.max_depth and len(idx) >= 2 * cfg.min_leaf:
        best = None
        cg = c[idx] * g[idx]
        for j in range(feats.shape[1]):
            col = feats[idx, j]
            for thr in thresholds[j]:
                left = col <= thr
                nl = int(left.sum())
                if nl < cfg.min_leaf or len(idx) - nl < cfg.min_leaf:
                    continue
                t = _welch_t(cg[left], cg[~left])
                if abs(t) < cfg.t_split:
                    continue
                gl, hl = _stats(g[idx][left], h[idx][left], c[idx][left])
                gr, hr = _stats(g[idx][~left], h[idx][~left], c[idx][~left])
                gp, hp = _stats(g[idx], h[idx], c[idx])
                gain = (nl * gl ** 2 / (max(hl, 0) + cfg.lam)
                        + (len(idx) - nl) * gr ** 2 / (max(hr, 0) + cfg.lam)
                        - len(idx) * gp ** 2 / (max(hp, 0) + cfg.lam))
                if best is None or gain > best[0]:
                    best = (gain, j, float(thr), t)
        if best is not None and best[0] > 0:
            _, j, thr, t = best
            left = feats[idx, j] <= thr
            node.update({"feature": j, "threshold": thr, "t": round(t, 2),
                         "left": grow_tree(feats, g, h, c, idx[left], depth + 1, cfg, thresholds),
                         "right": grow_tree(feats, g, h, c, idx[~left], depth + 1, cfg, thresholds)})
            return node
    node["w"] = leaf_weight(g[idx], h[idx], c[idx], cfg.lam, cfg.lr, cfg.w_max)
    return node


def tree_leaves(node, path=()):
    if "w" in node:
        return [(path, node)]
    j, thr = node["feature"], node["threshold"]
    return (tree_leaves(node["left"], path + ((j, thr, "<="),))
            + tree_leaves(node["right"], path + ((j, thr, ">"),)))


def assign_leaves(node, feats) -> np.ndarray:
    """Per-example weight from a tree."""
    out = np.empty(len(feats))
    for path, leaf in tree_leaves(node):
        mask = np.ones(len(feats), dtype=bool)
        for j, thr, side in path:
            mask &= (feats[:, j] <= thr) if side == "<=" else (feats[:, j] > thr)
        out[mask] = leaf["w"]
    return out


def describe_condition(j, thr, side):
    name = FEATURE_NAMES[j] if j < len(FEATURE_NAMES) else f"feature {j}"
    if j in BINARY_FEATURES:
        return f"{name}: {'no' if side == '<=' else 'yes'}"
    return f"{name} {'≤' if side == '<=' else '>'} {thr:.3g}"


def scale_tree(node, factor):
    if "w" in node:
        node["w"] *= factor
    else:
        scale_tree(node["left"], factor)
        scale_tree(node["right"], factor)


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------

@dataclass
class BoostConfig:
    max_rounds: int = 12
    sample_size: int = 256
    batch_size: int = 32
    eps: float = 0.05          # probe size along the unit-norm direction D
    lam: float = 1.0           # XGBoost lambda (on per-example means)
    lr: float = 0.5            # XGBoost eta (shrinkage)
    w_max: float = 2.0         # cap on a leaf's step along D
    t_min: float = 2.0         # a layer may only win if its gradient is beyond noise
    t_split: float = 2.5       # a split must separate clearly different pulls
    max_depth: int = 2
    min_leaf: int = 25
    max_minutes: float = 150.0
    seed: int = 0


@dataclass
class Split:
    ids: torch.Tensor
    mask: torch.Tensor
    y: torch.Tensor
    feats: np.ndarray
    weight: np.ndarray  # class-balance weight per example


@dataclass
class Booster:
    module: str
    u: torch.Tensor
    v: torch.Tensor
    tree: dict
    gain: float
    t: float


class BoostTrainer:
    def __init__(self, model, head, train: Split, test: Split, cfg: BoostConfig, log=print):
        self.model, self.head, self.cfg, self.log = model, head, cfg, log
        self.splits = {"train": train, "test": test}
        for p in list(model.parameters()) + list(head.parameters()):
            p.requires_grad_(False)
        self.mods: dict[str, BoostedLinear] = {}
        for name, (parent, attr) in candidate_modules(model).items():
            wrapped = BoostedLinear(getattr(parent, attr))
            setattr(parent, attr, wrapped)
            self.mods[name] = wrapped
        self.boosters: list[Booster] = []
        self.rounds: list[dict] = []
        self.rng = np.random.default_rng(cfg.seed)
        self.thresholds = self._thresholds(train.feats)

    @staticmethod
    def _thresholds(feats):
        out = []
        for j in range(feats.shape[1]):
            if j in BINARY_FEATURES:
                out.append([0.5])
            else:
                out.append(sorted(set(np.quantile(feats[:, j], [0.2, 0.35, 0.5, 0.65, 0.8]).tolist())))
        return out

    # -- forward helpers ----------------------------------------------------
    def _set_batch(self, split: str, idx: np.ndarray):
        feats = self.splits[split].feats[idx]
        by_mod: dict[str, list[torch.Tensor]] = {n: [] for n in self.mods}
        for b in self.boosters:
            by_mod[b.module].append(torch.tensor(assign_leaves(b.tree, feats), dtype=torch.float32))
        for name, mod in self.mods.items():
            mod.batch_w = by_mod[name]

    def _logits(self, split: str, idx: np.ndarray):
        s = self.splits[split]
        self._set_batch(split, idx)
        t = torch.as_tensor(idx)
        hidden = self.model(input_ids=s.ids[t], attention_mask=s.mask[t]).last_hidden_state
        return self.head(mean_pool(hidden, s.mask[t]))

    def losses(self, split: str, idx: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start in range(0, len(idx), self.cfg.batch_size):
                b = idx[start:start + self.cfg.batch_size]
                logits = self._logits(split, b)
                y = self.splits[split].y[torch.as_tensor(b)]
                out.append(nn.functional.cross_entropy(logits, y, reduction="none").numpy())
        return np.concatenate(out).astype(np.float64)

    def predict(self, split: str) -> np.ndarray:
        idx = np.arange(len(self.splits[split].y))
        preds = []
        with torch.no_grad():
            for start in range(0, len(idx), self.cfg.batch_size):
                preds.append(self._logits(split, idx[start:start + self.cfg.batch_size]).argmax(-1).numpy())
        return np.concatenate(preds)

    def gradients(self, idx: np.ndarray) -> dict[str, torch.Tensor]:
        weights = {n: m.base.weight for n, m in self.mods.items()}
        for w in weights.values():
            w.requires_grad_(True)
            w.grad = None
        cw = torch.tensor(self.splits["train"].weight[idx], dtype=torch.float32)
        total = cw.sum()
        for start in range(0, len(idx), self.cfg.batch_size):
            b = idx[start:start + self.cfg.batch_size]
            logits = self._logits("train", b)
            y = self.splits["train"].y[torch.as_tensor(b)]
            loss = (nn.functional.cross_entropy(logits, y, reduction="none")
                    * cw[start:start + len(b)]).sum() / total
            loss.backward()
        grads = {n: w.grad.detach().clone() for n, w in weights.items()}
        for w in weights.values():
            w.requires_grad_(False)
            w.grad = None
        return grads

    # -- one boosting round -------------------------------------------------
    def round(self, r: int) -> bool:
        cfg = self.cfg
        train = self.splits["train"]
        n = min(cfg.sample_size, len(train.y))
        idx = self.rng.choice(len(train.y), size=n, replace=False)
        c = train.weight[idx]

        grads = self.gradients(idx)
        base_loss = self.losses("train", idx)
        cands = []
        for name, grad in grads.items():
            U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
            u, v = U[:, 0].contiguous(), Vh[0].contiguous()
            mod = self.mods[name]
            mod.probe = (u, v, cfg.eps)
            plus = self.losses("train", idx)
            mod.probe = (u, v, -cfg.eps)
            minus = self.losses("train", idx)
            mod.probe = None
            g = (plus - minus) / (2 * cfg.eps)
            h = (plus - 2 * base_loss + minus) / cfg.eps ** 2
            mg, mh = _stats(g, h, c)
            cg = c * g
            t = float(cg.mean() / (cg.std(ddof=1) / math.sqrt(n) + 1e-12))
            gain = float(n * mg ** 2 / (max(mh, 0.0) + cfg.lam))
            cands.append({"module": name, "gain": gain, "t": t, "u": u, "v": v, "g": g, "h": h,
                          "mean_h": float(mh)})

        gain_map = {cd["module"]: round(cd["gain"], 5) for cd in cands}
        eligible = sorted([cd for cd in cands if abs(cd["t"]) >= cfg.t_min],
                          key=lambda cd: cd["gain"], reverse=True)
        record = {"round": r, "gains": gain_map, "sample_loss_before": float((c * base_loss).sum() / c.sum())}
        if not eligible:
            record["stopped"] = "no layer's gain is beyond its noise (all |t| < %.1f)" % cfg.t_min
            self.rounds.append(record)
            return False

        for cd in eligible[:3]:  # try the best few; keep the first that lowers the loss
            tree = grow_tree(train.feats[idx], cd["g"], cd["h"], c, np.arange(n), 0, cfg,
                             self.thresholds)
            # train.feats[idx] is indexed by position within the sample
            booster = Booster(cd["module"], cd["u"], cd["v"], tree, cd["gain"], cd["t"])
            for _ in range(4):
                self.boosters.append(booster)
                self.mods[booster.module].us.append(booster.u)
                self.mods[booster.module].vs.append(booster.v)
                after = self.losses("train", idx)
                after_loss = float((c * after).sum() / c.sum())
                if after_loss < record["sample_loss_before"]:
                    leaves = [{"rule": " and ".join(describe_condition(*p) for p in path) or "all issues",
                               "issues": leaf["n"], "weight": round(leaf["w"], 4)}
                              for path, leaf in tree_leaves(tree)]
                    record.update({"module": cd["module"], "gain": round(cd["gain"], 5),
                                   "t": round(cd["t"], 2), "leaves": leaves,
                                   "sample_loss_after": after_loss})
                    self.rounds.append(record)
                    return True
                # undo, halve the step, retry
                self.boosters.pop()
                self.mods[booster.module].us.pop()
                self.mods[booster.module].vs.pop()
                scale_tree(tree, 0.5)
        record["stopped"] = "no candidate lowered the loss after shrinking its step"
        self.rounds.append(record)
        return False

    def fit(self, on_round=None):
        start = time.time()
        reason = f"reached the round limit ({self.cfg.max_rounds})"
        for r in range(1, self.cfg.max_rounds + 1):
            if (time.time() - start) / 60 > self.cfg.max_minutes:
                reason = "time limit"
                break
            kept = self.round(r)
            rec = self.rounds[-1]
            if on_round:
                on_round(rec)
            self.log(f"round {r}: {rec.get('module', '-')} gain={rec.get('gain')} "
                     f"t={rec.get('t')} loss {rec['sample_loss_before']:.4f}"
                     f" -> {rec.get('sample_loss_after', float('nan')):.4f}")
            if not kept:
                reason = rec["stopped"]
                break
        return reason
