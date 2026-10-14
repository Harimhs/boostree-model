"""The real experiment: train at a cutoff date T, when recent labels are incomplete.

At cutoff T, a commit is "known buggy" only if its fix was committed before T
(real fix dates from ApacheJIT's SZZ links). Everything else looks clean, even
if a bug is discovered later. Then we test on the next year's commits using
their final labels.

  q(e) = chance a bug would have been found within e days of the commit,
         estimated ONLY from older commits whose delays were observable by T.

Variants (all use the same from-scratch boosting engine and 13 metrics):
  naive         : log-loss on labels as known at T (what everyone does)
  wait-90       : drop commits younger than 90 days, then log-loss (Cabral-style)
  delay raw-h   : delay-aware loss with its raw hessian (clipped when <= 0)
  delay Fisher  : delay-aware loss with the Fisher hessian (ours)
  oracle        : log-loss on FINAL labels (impossible in practice; the ceiling)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from riskycommit.engine import Booster, Params, logistic_obj, sigmoid
from riskycommit.features import BASE, load

DAY = 86400.0


def with_discovery(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    links = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(data_dir, "commit_links_*.csv"))])
    first = links.groupby("bug_hash").fix_date.min()
    df = df.copy()
    df["found_at"] = df.commit_id.map(first).astype(float)
    df.loc[df.buggy == 0, "found_at"] = np.inf
    df["found_at"] = np.maximum(df.found_at, df.author_date)  # 2 tiny clock-skew cases
    return df


def delay_curve(df: pd.DataFrame, T: float, horizon_days=3 * 365, window_days=2 * 365):
    """Empirical CDF of discovery delay, from commits authored in [T-H-W, T-H): for those,
    any bug found within H days was already visible at T, so the curve uses no future info."""
    lo, hi = T - (horizon_days + window_days) * DAY, T - horizon_days * DAY
    old = df[(df.author_date >= lo) & (df.author_date < hi) & (df.found_at <= T)]
    d = np.sort(((old.found_at - old.author_date) / DAY).to_numpy())
    d = d[d <= horizon_days]
    def q(e_days):
        return np.clip(np.searchsorted(d, e_days, side="right") / max(1, len(d)), 1e-3, 1.0)
    return q, d


class DelayObjective:
    """loss = -log p            if a bug was already found
              -log(1 - p*q)      if no bug found yet (q = chance it would have been found by now)."""

    def __init__(self, q, fisher=True):
        self.q, self.fisher = q, fisher
        self.last_raw_h = None

    def __call__(self, z, y):
        p, q = sigmoid(z), self.q
        pq = np.clip(p * q, 1e-12, 1 - 1e-9)
        g = np.where(y == 1, p - 1, p * (1 - p) * q / (1 - pq))
        raw = np.where(y == 1, p * (1 - p), p * (1 - p) * q * (1 - 2 * p + p * p * q) / (1 - pq) ** 2)
        self.last_raw_h = raw
        if self.fisher:
            h = np.where(y == 1, p * (1 - p), q * p * (1 - p) ** 2 / (1 - pq))
        else:
            h = raw
        return g, np.maximum(h, 1e-6)

    def loss(self, z, y, q):
        p = sigmoid(z)
        return float(np.mean(np.where(y == 1, -np.log(p + 1e-12), -np.log(1 - np.clip(p * q, 0, 1 - 1e-9)))))


def evaluate(z, y):
    p = sigmoid(z)
    k = int(round(len(y) * 0.2))
    top = np.argsort(-z, kind="stable")[:k]
    return {"auc": round(float(roc_auc_score(y, z)), 4),
            "recall_top20": round(float(y[top].sum() / y.sum()), 4),
            "logloss": round(float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))), 4),
            "mean_predicted": round(float(p.mean()), 4), "actual_rate": round(float(y.mean()), 4)}


def run_cutoff(df, T, log=print):
    pool = df[df.author_date < T].copy()
    pool["y_obs"] = (pool.found_at <= T).astype(float)
    pool["elapsed"] = (T - pool.author_date) / DAY
    test = df[(df.author_date >= T) & (df.author_date < T + 365 * DAY)]
    q_fn, delays = delay_curve(df, T)
    pool["q"] = q_fn(pool.elapsed.to_numpy())
    cut = pool.author_date.quantile(0.9)          # last 10% of the pool = early-stopping set
    tr, va = pool[pool.author_date < cut], pool[pool.author_date >= cut]
    X = lambda d: d[BASE].to_numpy(float)
    yt = test.buggy.to_numpy(float)
    out = {"cutoff": pd.Timestamp(T, unit="s").strftime("%Y-%m-%d"), "train_commits": len(pool),
           "test_commits": len(test), "test_buggy": int(yt.sum()),
           "known_buggy_at_cutoff": round(float(pool.y_obs.mean()), 4),
           "final_buggy_in_pool": round(float(pool.buggy.mean()), 4),
           "last_year_known_vs_final": [round(float(pool[pool.elapsed < 365].y_obs.mean()), 4),
                                        round(float(pool[pool.elapsed < 365].buggy.mean()), 4)],
           "delay_curve_commits": len(delays), "variants": {}}

    def fit(name, objective, ytr, yva, trd=tr, vad=va, val_loss=None, base=None):
        m = Booster(Params(), objective=objective).fit(X(trd), ytr, X(vad), yva, log=lambda *_: None,
                                                       val_loss=val_loss, base=base)
        res = evaluate(m.raw(X(test)), yt)
        res["trees"] = len(m.trees)
        if m.stats["hessian_values"]:
            res["negative_raw_hessians"] = round(m.stats["negative_hessians"] / m.stats["hessian_values"], 4)
        out["variants"][name] = res
        log(f"  {name:14s} {res}")

    fit("naive", logistic_obj, tr.y_obs.to_numpy(), va.y_obs.to_numpy())
    w_tr, w_va = tr[tr.elapsed >= 90], va[va.elapsed >= 90]
    if len(w_va) < 500:  # validation window is the most recent 10%; fall back to a time split of the matured pool
        wpool = pool[pool.elapsed >= 90]
        c2 = wpool.author_date.quantile(0.9)
        w_tr, w_va = wpool[wpool.author_date < c2], wpool[wpool.author_date >= c2]
    fit("wait-90", logistic_obj, w_tr.y_obs.to_numpy(), w_va.y_obs.to_numpy(), w_tr, w_va)
    ybar, qbar = tr.y_obs.mean(), tr.q.mean()
    base = float(np.log(min(ybar / qbar, 0.9) / (1 - min(ybar / qbar, 0.9))))
    for name, fisher in (("delay raw-h", False), ("delay Fisher", True)):
        obj = DelayObjective(tr.q.to_numpy(), fisher=fisher)
        vq, vy = va.q.to_numpy(), va.y_obs.to_numpy()
        fit(name, obj, tr.y_obs.to_numpy(), vy, val_loss=lambda z, vy=vy, vq=vq: obj.loss(z, vy, vq), base=base)
    fit("oracle", logistic_obj, tr.buggy.to_numpy(float), va.buggy.to_numpy(float))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    df = with_discovery(load(args.csv), args.data_dir)
    cutoffs = [pd.Timestamp(f"{y}-01-01", tz="UTC").timestamp() for y in (2015, 2016, 2017)]
    results = []
    for T in cutoffs:
        print(f"cutoff {pd.Timestamp(T, unit='s').date()}")
        results.append(run_cutoff(df, T))
    names = list(results[0]["variants"])
    summary = {n: {k: round(float(np.mean([r["variants"][n][k] for r in results])), 4)
                   for k in ("auc", "recall_top20", "logloss", "mean_predicted", "actual_rate")} for n in names}
    out = {"cutoffs": results, "mean_over_cutoffs": summary, "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
