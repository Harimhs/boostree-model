"""Build docs/xray/dash.json: training trends, with/without comparison, and live training samples.

Usage:
    python -m riskycommit.dashboard_data --csv apachejit_total.csv --data-dir apachejit/data \
        --research docs/xray/delay_adapt.json --out docs/xray/dash.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from riskycommit.delay import DAY, delay_curve, with_discovery
from riskycommit.engine import Booster as TreeBooster, Params
from riskycommit.features import BASE, READABLE, DenseEncoder, load
from riskycommit.frozen_boost import boost, delay_aware_obj, logloss_obj, train_mlp


def top20(z, y):
    k = int(round(len(y) * 0.2))
    return float(y[np.argsort(-z, kind="stable")[:k]].sum() / y.sum())


def booster_trajectories(df, year=2016, seed=1):
    """Test AUC after every booster that is added, for naive vs delay-aware vs oracle."""
    T = pd.Timestamp(f"{year}-01-01", tz="UTC").timestamp()
    q_fn, _ = delay_curve(df, T)
    known = lambda d: (d.found_at <= T).astype(float).to_numpy()
    base = df[df.author_date < T - 365 * DAY]
    ad = df[(df.author_date >= T - 365 * DAY) & (df.author_date < T)].copy()
    ad["q"] = q_fn(((T - ad.author_date) / DAY).to_numpy())
    test = df[(df.author_date >= T) & (df.author_date < T + 365 * DAY)]
    yt = test.buggy.to_numpy(float)
    cut, bcut = int(len(ad) * 0.8), int(len(base) * 0.9)
    atr, ava, btr, bva = ad.iloc[:cut], ad.iloc[cut:], base.iloc[:bcut], base.iloc[bcut:]
    enc = DenseEncoder().fit(btr)
    X, rn = enc.transform, list(READABLE)
    R = lambda d: d[rn].to_numpy(float)
    net = train_mlp(X(btr), known(btr), X(bva), known(bva), seed=seed, log=lambda *_: None)
    Xt, Rt = X(test), R(test)
    z0 = net.forward(Xt)[0]
    start = {"auc": round(float(roc_auc_score(yt, z0)), 4), "top20": round(top20(z0, yt), 4)}
    out = {"cutoff": f"{year}-01-01", "test_year": year, "start": start, "series": {}}
    for name, obj, ytr, yva in (("naive", logloss_obj, known(atr), known(ava)),
                                ("ours", delay_aware_obj, known(atr), known(ava)),
                                ("oracle", logloss_obj, atr.buggy.to_numpy(float), ava.buggy.to_numpy(float))):
        pts = [{"boosters": 0, **start}]
        def cb(m, rec, pts=pts):
            z = m.forward(Xt, Rt)[0]
            pts.append({"boosters": len(pts), "auc": round(float(roc_auc_score(yt, z)), 4),
                        "top20": round(top20(z, yt), 4), "layer": rec["layer"]})
        boost(net, X(atr), R(atr), ytr, X(ava), R(ava), yva, rn, obj=obj, q=atr.q.to_numpy(),
              qv=ava.q.to_numpy(), log=lambda *_: None, on_round=cb)
        out["series"][name] = pts
    return out, ad


def tree_curve(df):
    tr, va, te = df[df.year <= 2015], df[df.year == 2016], df[df.year >= 2017]
    m = TreeBooster(Params()).fit(tr[BASE].to_numpy(float), tr.buggy.to_numpy(float),
                                  va[BASE].to_numpy(float), va.buggy.to_numpy(float), log=lambda *_: None)
    B = m.binner.transform(te[BASE].to_numpy(float))
    yt = te.buggy.to_numpy(float)
    z = np.full(len(te), m.base)
    pts = []
    for i, arrs in enumerate(m._arrays, 1):
        z += m.p.learning_rate * m._tree_predict(arrs, B)
        if i % 4 == 0 or i == len(m._arrays):
            pts.append({"trees": i, "test_auc": round(float(roc_auc_score(yt, z)), 4),
                        "val_logloss": m.history[i - 1]["val_logloss"]})
    return {"points": pts, "kept": len(m.trees), "ran": len(m.history)}


def titles(rows):
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for repo, sha in rows:
            path = f"{tmp}/{repo.replace('/', '_')}"
            subprocess.run(["git", "init", "-q", path])
            r = subprocess.run(["git", "-C", path, "fetch", "-q", "--depth=1", "--filter=blob:none",
                                f"https://github.com/{repo}", sha], capture_output=True, timeout=120)
            if r.returncode == 0:
                t = subprocess.run(["git", "-C", path, "log", "-1", "--format=%s", "FETCH_HEAD"],
                                   capture_output=True, text=True).stdout.strip()
                import re
                t = re.sub(r"\.?\s*Contributed by.*$", "", t)
                t = re.sub(r"\s*\([^()]*reviewed by[^()]*\)?\s*$", "", t)
                out[sha] = t[:110]
    return out


def training_samples(ad, cutoff, n_each=4, seed=5):
    rng = np.random.default_rng(seed)
    T = pd.Timestamp(cutoff, tz="UTC").timestamp()
    ad = ad.assign(age_days=((T - ad.author_date) / DAY).round().astype(int),
                   known=(ad.found_at <= T), found_days=((ad.found_at - ad.author_date) / DAY))
    groups = {
        "bug already found": ad[ad.known],
        "hidden bug (found later)": ad[(~ad.known) & (ad.buggy == 1)],
        "really clean": ad[ad.buggy == 0],
    }
    picks = []
    for g, sub in groups.items():
        for i in rng.choice(len(sub), size=min(n_each, len(sub)), replace=False):
            r = sub.iloc[i]
            picks.append({"sha": r.commit_id, "project": r.project, "date": pd.Timestamp(int(r.author_date), unit="s").strftime("%Y-%m-%d"),
                          "age_days": int(r.age_days), "q": round(float(r.q), 3), "group": g,
                          "known_bug": bool(r.known), "final_bug": bool(r.buggy),
                          "found_after_days": None if not np.isfinite(r.found_days) else int(r.found_days),
                          "la": int(r.la), "ld": int(r.ld), "nf": int(r.nf)})
    t = titles([(p["project"], p["sha"]) for p in picks])
    for p in picks:
        p["title"] = t.get(p["sha"])
    order = rng.permutation(len(picks))
    counts = {"adapt_commits": int(len(ad)), "known_buggy": int(ad.known.sum()),
              "hidden_buggy": int(((~ad.known) & (ad.buggy == 1)).sum()), "clean": int((ad.buggy == 0).sum())}
    return [picks[i] for i in order], counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--research", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    df = with_discovery(load(args.csv), args.data_dir)
    traj, ad = booster_trajectories(df)
    samples, counts = training_samples(ad, traj["cutoff"])
    R = json.load(open(args.research))
    per_cutoff = {}
    for r in R["runs"]:
        y = r["cutoff"][:4]
        for k, v in r["variants"].items():
            per_cutoff.setdefault(k, {}).setdefault(y, []).append(v["auc"])
    per_cutoff = {k: {y: round(float(np.mean(v)), 4) for y, v in d.items()} for k, d in per_cutoff.items()}
    out = {"trajectory": traj, "tree_curve": tree_curve(df), "samples": samples, "sample_counts": counts,
           "per_cutoff_auc": per_cutoff,
           "bug_rate_by_year": [{"year": int(y), "rate": round(float(g.buggy.mean()), 3)}
                                for y, g in df[df.year >= 2010].groupby("year")]}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("sample_counts", "per_cutoff_auc")}, indent=1))
    for n, s in traj["series"].items():
        print(n, [(p["boosters"], p["auc"]) for p in s])


if __name__ == "__main__":
    main()
