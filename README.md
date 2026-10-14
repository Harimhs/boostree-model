# Commit X-ray: delay-aware boosting for risky-commit prediction

**Live demo (phone-friendly):** https://harimhs.github.io/boostree-model/xray/

Commit X-ray scores every code change for bug risk the moment it is pushed.
Its research contribution is in the **gradient and hessian**. It deals with the
fact that a commit's true label arrives late: real bugs take a median of
**154 days** to be found.

## The problem: labels arrive late

When a later "fix" commit lands, SZZ (`git blame`) traces it back to the commit
that introduced the bug. In ApacheJIT's real fix dates, about **half the bugs in
the most recent year are not yet known**: 12–17% of commits look buggy, against
27–30% in the end. Any model retrained on recent data learns "recent = clean".

## The method

For a commit with no bug found yet, let p be the risk and q the chance its bug
would have surfaced by now. q is learned from older commits only.

```
loss            = −log(1 − p·q)
gradient        = p(1−p)·q / (1−p·q)
raw hessian     = p(1−p)·q·(1−2p+p²q) / (1−p·q)²   ← can be negative: breaks the Newton step
Fisher hessian  = q·p·(1−p)² / (1−p·q)               ← always positive; equals p(1−p) at q = 1
```

The objective plugs into:

1. **A gradient-boosting engine written from scratch in NumPy**
   (`src/riskycommit/engine.py`). Its AUC is 0.8647, against 0.8635 for
   scikit-learn on the same split.
2. **Booster trees on a frozen neural network** (`src/riskycommit/frozen_boost.py`).
   - A small MLP is trained from scratch and then frozen.
   - Adaptation adds only rank-1 Newton-step updates (`−G/(H+λ)`) to its
     weights.
   - Small trees over readable commit features decide which commits receive
     each update.

## Results (ApacheJIT, 106k real commits, 15 Apache projects)

| Experiment | Result |
|---|---|
| Trees, train 2003–2015 → test 2017–2019 | AUC 0.865. Reviewing the riskiest 20% catches 63% of bug-inducing commits (lines-added rule 47%, random 20%) |
| Frozen net adapted on the most recent year, labels as known (6 runs) | naive boosters **hurt** the model: AUC 0.848 → 0.820. **Delay-aware boosters → 0.852**, ahead of naive boosters in 6/6 runs on both AUC and recall@20% (ceiling with future labels: 0.862) |
| Full 10+ year history | delay barely affects ranking. Delay-aware mainly corrects the predicted bug rate (log-loss 0.421 → 0.399, oracle 0.393) |

Leakage check: one-hot project identity raised AUC from 0.84 to 0.89. It is a
per-project shortcut that does not transfer to a new repository, so it was
removed. Feature scaling is fitted on training years only, and every split is
by time.

## Reproduce

```bash
git clone --filter=blob:none --sparse https://github.com/harsh8398/apachejit && cd apachejit && git sparse-checkout set dataset data && cd ..
PYTHONPATH=src python -m riskycommit.train --csv apachejit/dataset/apachejit_total.csv --out docs/xray/data.json --model models/riskycommit_v0.json
PYTHONPATH=src python -m riskycommit.experiment --csv apachejit/dataset/apachejit_total.csv --out docs/xray/frozen.json
PYTHONPATH=src python -m riskycommit.delay --csv apachejit/dataset/apachejit_total.csv --data-dir apachejit/data --out docs/xray/delay.json
PYTHONPATH=src python -m riskycommit.delay_adapt --csv apachejit/dataset/apachejit_total.csv --data-dir apachejit/data --out docs/xray/delay_adapt.json
```

Only NumPy, pandas and scikit-learn (for metrics) are needed. Everything runs
on CPU in minutes.

## Earlier work in this repo

`src/boostree/` is the first experiment: gain-guided booster trees inside a
frozen text encoder for GitHub issue labels. Its gains were small, and it led
to this project.

## License

MIT
