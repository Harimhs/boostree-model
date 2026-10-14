"""Where label delay really bites: adapting to the MOST RECENT year.

For each cutoff T:
  base  : commits before T-1y (labels mostly matured)  -> small net trained from scratch, then FROZEN
  adapt : commits in [T-1y, T) - about half of their bugs are not found yet at T
  test  : commits in [T, T+1y) with FINAL labels

Adaptation variants on the frozen net (weights never change):
  boosters naive   : Newton boosters with plain log-loss on labels known at T
  boosters delay   : Newton boosters with the delay-aware loss + Fisher hessian (ours)
  boosters oracle  : Newton boosters with FINAL labels (ceiling, impossible in practice)
  SGD naive        : ordinary fine-tuning of all weights on labels known at T
Plus boosted trees trained only on the recent year (naive / delay / oracle).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from riskycommit.delay import DAY, DelayObjective, delay_curve, evaluate, with_discovery
from riskycommit.engine import Booster as TreeBooster, Params, logistic_obj
from riskycommit.features import BASE, READABLE, DenseEncoder, load
from riskycommit.frozen_boost import boost, delay_aware_obj, finetune, logloss_obj, train_mlp


def run(df, T, seed=0, log=print):
    q_fn, _ = delay_curve(df, T)
    known = lambda d: (d.found_at <= T).astype(float).to_numpy()
    base = df[df.author_date < T - 365 * DAY]
    ad = df[(df.author_date >= T - 365 * DAY) & (df.author_date < T)].copy()
    ad["q"] = q_fn(((T - ad.author_date) / DAY).to_numpy())
    test = df[(df.author_date >= T) & (df.author_date < T + 365 * DAY)]
    yt = test.buggy.to_numpy(float)
    cut = int(len(ad) * 0.8)
    atr, ava = ad.iloc[:cut], ad.iloc[cut:]
    bcut = int(len(base) * 0.9)
    btr, bva = base.iloc[:bcut], base.iloc[bcut:]

    enc = DenseEncoder().fit(btr)
    X, rn = enc.transform, list(READABLE)
    R = lambda d: d[rn].to_numpy(float)
    net = train_mlp(X(btr), known(btr), X(bva), known(bva), seed=seed, log=lambda *_: None)
    out = {"cutoff": pd.Timestamp(T, unit="s").strftime("%Y-%m-%d"), "seed": seed,
           "adapt_known_rate": round(float(known(ad).mean()), 4), "adapt_final_rate": round(float(ad.buggy.mean()), 4),
           "variants": {}}
    ev = lambda name, z: out["variants"].__setitem__(name, evaluate(z, yt))
    ev("frozen net", net.forward(X(test))[0])

    for name, obj, ytr, yva in (
            ("boosters naive", logloss_obj, known(atr), known(ava)),
            ("boosters delay (ours)", delay_aware_obj, known(atr), known(ava)),
            ("boosters oracle", logloss_obj, atr.buggy.to_numpy(float), ava.buggy.to_numpy(float))):
        bn, hist = boost(net, X(atr), R(atr), ytr, X(ava), R(ava), yva, rn, obj=obj,
                         q=atr.q.to_numpy(), qv=ava.q.to_numpy(), log=lambda *_: None)
        ev(name, bn.forward(X(test), R(test))[0])
        out["variants"][name]["boosters"] = len(hist)
    ft = finetune(net, X(atr), known(atr), X(ava), known(ava), seed=seed)
    ev("SGD fine-tune naive", ft.forward(X(test))[0])

    # boosted trees trained on the recent year only
    Xb = lambda d: d[BASE].to_numpy(float)
    for name, mk, ytr, yva in (
            ("trees recent-year naive", lambda: logistic_obj, known(atr), known(ava)),
            ("trees recent-year delay (ours)", lambda: DelayObjective(atr.q.to_numpy()), known(atr), known(ava)),
            ("trees recent-year oracle", lambda: logistic_obj, atr.buggy.to_numpy(float), ava.buggy.to_numpy(float))):
        obj = mk()
        vl = None
        base_logit = None
        if isinstance(obj, DelayObjective):
            vq = ava.q.to_numpy()
            vl = lambda z, vy=yva, vq=vq, o=obj: o.loss(z, vy, vq)
            r = min(ytr.mean() / atr.q.mean(), 0.9)
            base_logit = float(np.log(r / (1 - r)))
        m = TreeBooster(Params(), objective=obj).fit(Xb(atr), ytr, Xb(ava), yva, log=lambda *_: None,
                                                     val_loss=vl, base=base_logit)
        ev(name, m.raw(Xb(test)))
    for n, v in out["variants"].items():
        log(f"  {n:32s} AUC {v['auc']:.4f}  top20 {v['recall_top20']:.3f}  logloss {v['logloss']:.4f}  "
            f"pred {v['mean_predicted']:.3f} vs actual {v['actual_rate']:.3f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args(argv)
    t0 = time.time()
    df = with_discovery(load(args.csv), args.data_dir)
    runs = []
    for y in (2015, 2016, 2017):
        T = pd.Timestamp(f"{y}-01-01", tz="UTC").timestamp()
        for s in range(args.seeds):
            print(f"cutoff {y}-01-01 seed {s}  (adapt year: known vs final bug rate)")
            runs.append(run(df, T, seed=s))
    names = list(runs[0]["variants"])
    summary = {n: {k: round(float(np.mean([r["variants"][n][k] for r in runs])), 4)
                   for k in ("auc", "recall_top20", "logloss", "mean_predicted", "actual_rate")} for n in names}
    wins = {n: int(sum(r["variants"][n]["logloss"] < r["variants"][n.replace("delay (ours)", "naive")]["logloss"]
                       for r in runs)) for n in names if "delay" in n}
    out = {"runs": runs, "mean": summary, "delay_beats_naive_logloss_in_runs": wins, "n_runs": len(runs),
           "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({"mean": summary, "wins": wins, "n_runs": len(runs)}, indent=1))


if __name__ == "__main__":
    main()
