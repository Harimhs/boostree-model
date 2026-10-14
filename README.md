# boostree

**Fine-tune a small transformer like XGBoost grows trees.**

boostree adds tiny rank-1 "booster" adapters to a frozen encoder, one round at a
time. XGBoost's own math decides everything:

- **Where to train:** each layer is scored with the gain `G² / (H + λ)`, and only
  the best layer gets the next booster.
- **Whom it applies to:** each booster is a small decision tree over the inputs,
  grown with XGBoost's split gain from per-example gradients and Hessians. It
  splits only where examples pull in opposite directions.
- **When to stop:** when no layer's gain beats its own noise, training stops.

## Showcase: GitHub issue triage that learns each repo's style

Every project labels issues differently (`bug`, `kind/bug`, `type: defect`,
and "how do I…" posts that one repo calls `question` and another `support`).
boostree learns one shared model plus booster trees that split per repo only
where repos disagree. It also reports where a repo's own labels are
inconsistent.

- Trained continuously on **real public issues** from [GH Archive](https://www.gharchive.org)
- Runs on **CPU**, free GitHub Actions runners
- No issue text is sent to any outside AI service

## Status

- [x] Step 1: pull real labeled issues and summarize them
- [ ] Step 2: baseline (frozen tiny encoder + label head)
- [ ] Step 3: booster trees
- [ ] Step 4: scheduled continuous training, versions on Hugging Face
- [ ] Step 5: live dashboard (curves, gain heatmap, split rules, drift, consistency report)

Live dashboard: https://harimhs.github.io/boostree-model/ · Latest data report: [data branch](https://github.com/Harimhs/boostree-model/blob/data/latest.md)

## Run step 1

GitHub: **Actions → Pull data → Run workflow**.

Locally:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m boostree.data.pull_gharchive --hours 2 --out issues.parquet --stats-out stats.json
PYTHONPATH=src python -m boostree.data.summarize --data issues.parquet --stats stats.json \
  --out-md summary.md --out-json summary.json
```

## License

MIT
