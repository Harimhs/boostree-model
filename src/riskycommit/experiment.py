"""Frozen net vs fine-tuning vs booster trees, same data, same test years.

  base net   : trained from scratch on 2003-2014 (early stop on 2015), then FROZEN
  adaptation : 2016 commits (first 80% to fit, last 20% held out for line search)
  test       : 2017-2019, never touched
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from riskycommit.engine import Booster as TreeBooster, Params
from riskycommit.features import BASE, READABLE, DenseEncoder, load
from riskycommit.frozen_boost import boost, finetune, logloss_obj, train_mlp


def score(z, y):
    k = int(round(len(y) * 0.2))
    top = np.argsort(-z, kind="stable")[:k]
    return {"auc": round(float(roc_auc_score(y, z)), 4),
            "recall_top20": round(float(y[top].sum() / y.sum()), 4),
            "test_logloss": round(float(logloss_obj(z, y)[0].mean()), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    t0 = time.time()
    df = load(args.csv)
    base_tr, base_va = df[df.year <= 2014], df[df.year == 2015]
    ad = df[df.year == 2016]
    cut = int(len(ad) * 0.8)
    ad_tr, ad_va = ad.iloc[:cut], ad.iloc[cut:]
    te = df[df.year >= 2017]

    enc = DenseEncoder().fit(base_tr)
    X = lambda d: enc.transform(d)
    rnames = list(READABLE)
    R = lambda d: d[rnames].to_numpy(float)
    y = lambda d: d.buggy.to_numpy(float)

    net = train_mlp(X(base_tr), y(base_tr), X(base_va), y(base_va))
    frozen_w = [w.copy() for w in net.W]

    res = {"features": {"dense_inputs": len(enc.names), "names": enc.names, "readable_for_trees": rnames},
           "split": {"base_train": "2003-2014", "base_val": "2015", "adapt": "2016", "test": "2017-2019"},
           "counts": {"base_train": len(base_tr), "adapt": len(ad), "test": len(te)}, "variants": {}}
    yt = y(te)
    res["variants"]["frozen net (no update)"] = score(net.forward(X(te))[0], yt)

    ft = finetune(net, X(ad_tr), y(ad_tr), X(ad_va), y(ad_va))
    res["variants"]["fine-tune all weights (SGD)"] = score(ft.forward(X(te))[0], yt)

    bnet, hist = boost(net, X(ad_tr), R(ad_tr), y(ad_tr), X(ad_va), R(ad_va), y(ad_va), rnames)
    assert all(np.array_equal(a, b) for a, b in zip(frozen_w, bnet.W)), "frozen weights changed!"
    res["variants"]["frozen net + booster trees"] = score(bnet.forward(X(te), R(te))[0], yt)
    res["boosters"] = hist

    # strong reference: from-scratch gradient-boosted trees on the 13 base metrics, same data up to 2016
    tb = TreeBooster(Params()).fit(df[df.year <= 2015][BASE].to_numpy(float), y(df[df.year <= 2015]),
                                   ad[BASE].to_numpy(float), y(ad), log=lambda *_: None)
    res["variants"]["reference: boosted trees on 13 metrics"] = score(tb.raw(te[BASE].to_numpy(float)), yt)
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("features",)}, indent=1)[:6000])


if __name__ == "__main__":
    main()
