"""Run step 3 on the collected issues and save results for the dashboard.

Same task and same train/test split as the baselines, so the numbers compare
directly. The starting point is exactly the MiniLM baseline (frozen encoder +
logistic-regression head); booster trees are then added inside the encoder.

Usage:
    python -m boostree.model.train_boost --out model/latest_boost.json \
        --history model/boost_history.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from boostree.model.baseline import ENCODER, score
from boostree.model.boost import (BoostConfig, BoostTrainer, FEATURE_NAMES, Split, mean_pool,
                                  text_features)
from boostree.model.dataset import build_family_dataset, load_issues_from_hf, time_split

MAX_LEN = 128


def encode(model, ids, mask, batch=64) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(ids), batch):
            hidden = model(input_ids=ids[s:s + batch], attention_mask=mask[s:s + batch]).last_hidden_state
            out.append(mean_pool(hidden, mask[s:s + batch]).numpy())
    return np.concatenate(out)


def head_from_logreg(clf: LogisticRegression, dim: int) -> nn.Linear:
    """Copy a fitted scikit-learn head into torch (handles the 2-class case)."""
    k = len(clf.classes_)
    head = nn.Linear(dim, k)
    with torch.no_grad():
        if clf.coef_.shape[0] == 1:  # binary: logit for class 1 vs a fixed 0 for class 0
            head.weight.zero_()
            head.bias.zero_()
            head.weight[1] = torch.tensor(clf.coef_[0], dtype=torch.float32)
            head.bias[1] = float(clf.intercept_[0])
        else:
            head.weight.copy_(torch.tensor(clf.coef_, dtype=torch.float32))
            head.bias.copy_(torch.tensor(clf.intercept_, dtype=torch.float32))
    return head


def build(model, tokenizer, train_df, test_df, classes):
    """Tokens, frozen embeddings, starting head and routing features."""
    def tok(texts):
        t = tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return t["input_ids"], t["attention_mask"]

    tr_ids, tr_mask = tok(train_df["text"].tolist())
    te_ids, te_mask = tok(test_df["text"].tolist())
    tr_emb, te_emb = encode(model, tr_ids, tr_mask), encode(model, te_ids, te_mask)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    clf.fit(tr_emb, train_df["target"])
    head = head_from_logreg(clf, tr_emb.shape[1])

    pca = PCA(n_components=4, random_state=0).fit(tr_emb)

    def feats(df, emb):
        return np.hstack([text_features(df["text"].tolist()), pca.transform(emb)])

    cls_index = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cls_index[t] for t in train_df["target"]])
    yte = np.array([cls_index[t] for t in test_df["target"]])
    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=ytr)
    train = Split(tr_ids, tr_mask, torch.tensor(ytr), feats(train_df, tr_emb), cw[ytr])
    test = Split(te_ids, te_mask, torch.tensor(yte), feats(test_df, te_emb), cw[yte])
    return head, train, test


def macro(trainer, split, classes, df):
    pred = trainer.predict(split)
    return score(df["target"].tolist(), [classes[p] for p in pred], classes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--dataset", default="boostree-issues")
    parser.add_argument("--max-rounds", type=int, default=30)
    parser.add_argument("--max-minutes", type=float, default=140)
    args = parser.parse_args(argv)

    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    started = time.time()
    raw = load_issues_from_hf(args.dataset, token=os.environ.get("HF_TOKEN"))
    ds = build_family_dataset(raw)
    result = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "encoder": ENCODER, "features": FEATURE_NAMES,
              "data": {"usable": int(len(ds))}, "rounds": []}
    if len(ds) < 60 or ds["target"].nunique() < 2:
        result["status"] = "not enough data yet"
    else:
        train_df, test_df = time_split(ds)
        classes = sorted(ds["target"].unique())
        tokenizer = AutoTokenizer.from_pretrained(ENCODER)
        model = AutoModel.from_pretrained(ENCODER).eval()
        head, train, test = build(model, tokenizer, train_df, test_df, classes)

        cfg = BoostConfig(max_rounds=args.max_rounds, max_minutes=args.max_minutes)
        trainer = BoostTrainer(model, head, train, test, cfg)
        start_scores = macro(trainer, "test", classes, test_df)

        def on_round(rec):
            if "module" in rec:
                rec["test_macro_f1"] = macro(trainer, "test", classes, test_df)["macro_f1"]

        reason = trainer.fit(on_round=on_round)
        final = macro(trainer, "test", classes, test_df)
        result.update({
            "status": "ok",
            "classes": classes,
            "data": {"usable": int(len(ds)), "train": int(len(train_df)), "test": int(len(test_df)),
                     "classes": {k: int(v) for k, v in ds["target"].value_counts().items()}},
            "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
            "start": start_scores,
            "final": final,
            "boosters": len(trainer.boosters),
            "stopped_because": reason,
            "rounds": trainer.rounds,
            "modules": list(trainer.mods),
        })
    result["seconds"] = round(time.time() - started, 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    with open(args.history, "a") as fh:
        fh.write(json.dumps({
            "time": result["generated_at"], "status": result["status"],
            "start_macro_f1": result.get("start", {}).get("macro_f1"),
            "boost_macro_f1": result.get("final", {}).get("macro_f1"),
            "boosters": result.get("boosters", 0),
            "stopped_because": result.get("stopped_because"),
        }) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rounds"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
