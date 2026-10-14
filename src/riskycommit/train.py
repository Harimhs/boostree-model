"""Train the risky-commit model on ApacheJIT and export the demo data.

Time-based split (no peeking at the future):
    train: commits up to 2015      validation (early stopping): 2016
    test : 2017-2019 (the model never saw these years)

Usage:
    python -m riskycommit.train --csv apachejit_total.csv --out docs/xray/data.json \
        --model models/riskycommit_v0.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from riskycommit.engine import Booster, Params

FEATURES = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp", "fix"]
LABELS = {
    "la": "lines added", "ld": "lines deleted", "nf": "files touched", "nd": "directories touched",
    "ns": "subsystems touched", "ent": "how spread-out the change is", "ndev": "developers who touched these files",
    "age": "days since these files last changed", "nuc": "past changes to these files",
    "aexp": "author's past commits", "arexp": "author's recent activity", "asexp": "author's commits in this subsystem",
    "fix": "is a bug-fix commit",
}


def load(csv):
    df = pd.read_csv(csv)
    df["fix"] = df["fix"].astype(str).str.lower().eq("true").astype(float)
    df["buggy"] = df["buggy"].astype(str).str.lower().eq("true").astype(int)
    df = df.sort_values("author_date").reset_index(drop=True)
    return df


def recall_at(scores, y, frac=0.2):
    """Review only the riskiest `frac` of commits: what share of buggy commits is caught?"""
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-scores, kind="stable")[:k]
    return float(y[top].sum() / max(1, y.sum())), float(y[top].mean())


def metrics(scores, y):
    r20, p20 = recall_at(scores, y, 0.2)
    return {"auc": round(float(roc_auc_score(y, scores)), 4), "recall_top20": round(r20, 4),
            "precision_top20": round(p20, 4)}


def explain(model, x, feature_names):
    bias, c = model.contributions(x)
    total = bias + c.sum()
    p = 1 / (1 + np.exp(-total))
    out = []
    for j, name in enumerate(feature_names):
        if abs(c[j]) < 1e-9:
            continue
        p_without = 1 / (1 + np.exp(-(total - c[j])))
        out.append({"feature": name, "label": LABELS[name], "value": float(x[j]),
                    "effect_pts": round(float(p - p_without) * 100, 1)})
    out.sort(key=lambda d: -abs(d["effect_pts"]))
    return float(p), out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--compare-sklearn", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    df = load(args.csv)
    tr, va, te = df[df.year <= 2015], df[df.year == 2016], df[df.year >= 2017]
    X = lambda d: d[FEATURES].to_numpy(float)
    ytr, yva, yte = tr.buggy.to_numpy(), va.buggy.to_numpy(), te.buggy.to_numpy()

    model = Booster(Params()).fit(X(tr), ytr, X(va), yva)
    p_te = model.predict_proba(X(te))
    result = {
        "generated_at": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "dataset": "ApacheJIT (14 Apache projects, real commits, SZZ labels)",
        "split": {"train": "2003-2015", "validation": "2016", "test": "2017-2019"},
        "counts": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te)),
                   "test_buggy": int(yte.sum()), "projects": int(df.project.nunique())},
        "trees": len(model.trees),
        "models": {
            "ours": metrics(p_te, yte),
            "lines_added_rule": metrics(te.la.to_numpy(float), yte),
            "random": {"auc": 0.5, "recall_top20": 0.2, "precision_top20": round(float(yte.mean()), 4)},
        },
    }
    if args.compare_sklearn:
        from sklearn.ensemble import HistGradientBoostingClassifier
        sk = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=5,
                                            early_stopping=False, random_state=0)
        sk.fit(np.vstack([X(tr), X(va)]), np.concatenate([ytr, yva]))
        result["models"]["sklearn_hgb_reference"] = metrics(sk.predict_proba(X(te))[:, 1], yte)

    # calibration-style buckets: "commits we rated X% were actually buggy Y% of the time"
    buckets = []
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        m = (p_te >= lo) & (p_te < hi)
        if m.sum():
            buckets.append({"range": f"{int(lo*100)}–{min(100, int(hi*100))}%", "commits": int(m.sum()),
                            "actually_buggy": round(float(yte[m].mean()), 3)})
    result["buckets"] = buckets

    # demo commits: a fair mix of hits AND misses from the test years
    rng = np.random.default_rng(7)
    te = te.assign(risk=p_te)
    picks = []
    groups = [("high risk, really buggy", (te.risk >= .6) & (te.buggy == 1), 10),
              ("low risk, really clean", (te.risk <= .15) & (te.buggy == 0), 6),
              ("missed: low risk but buggy", (te.risk <= .25) & (te.buggy == 1), 4),
              ("false alarm: high risk but clean", (te.risk >= .6) & (te.buggy == 0), 4)]
    for name, mask, n in groups:
        sub = te[mask]
        for i in rng.choice(len(sub), size=min(n, len(sub)), replace=False):
            row = sub.iloc[i]
            p, reasons = explain(model, row[FEATURES].to_numpy(float), FEATURES)
            picks.append({"sha": row.commit_id, "project": row.project, "date": pd.to_datetime(int(row.author_date), unit="s").strftime("%Y-%m-%d"),
                          "risk": round(p, 3), "buggy": bool(row.buggy), "group": name,
                          "reasons": reasons[:5], "la": int(row.la), "ld": int(row.ld), "nf": int(row.nf)})
    rng.shuffle(picks)
    result["commits"] = picks
    result["seconds"] = round(time.time() - t0, 1)

    for path, obj in [(args.out, result), (args.model, model.to_json(FEATURES))]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=1 if path == args.out else None)
    print(json.dumps({k: v for k, v in result.items() if k != "commits"}, indent=2))


if __name__ == "__main__":
    main()
