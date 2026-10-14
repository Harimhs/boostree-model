"""Step 2: baselines to beat.

1. TF-IDF + logistic regression (classic word-count model).
2. Frozen MiniLM encoder + logistic regression head (the encoder that the
   booster trees will later adapt).

Both are trained on older issues and tested on the newest ones.

Usage:
    python -m boostree.model.baseline --out model/latest_baseline.json \
        --history model/model_history.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from boostree.model.dataset import build_family_dataset, load_issues_from_hf, time_split

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def score(y_true, y_pred, classes) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=classes, average="macro",
                                         zero_division=0)), 4),
        "per_class": {
            c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
    }


def run_tfidf(train, test, classes) -> dict:
    start = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    xtr = vec.fit_transform(train["text"])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(xtr, train["target"])
    pred = clf.predict(vec.transform(test["text"]))
    return {**score(test["target"], pred, classes), "seconds": round(time.time() - start, 1)}


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ENCODER, device="cpu")
    model.max_seq_length = 256
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                        normalize_embeddings=True)


def run_minilm(train, test, classes) -> dict:
    start = time.time()
    xtr = embed(train["text"].tolist())
    xte = embed(test["text"].tolist())
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
    clf.fit(xtr, train["target"])
    pred = clf.predict(xte)
    return {**score(test["target"], pred, classes), "seconds": round(time.time() - start, 1),
            "encoder": ENCODER}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--dataset", default="boostree-issues")
    parser.add_argument("--min-per-class", type=int, default=15)
    args = parser.parse_args(argv)

    raw = load_issues_from_hf(args.dataset, token=os.environ.get("HF_TOKEN"))
    ds = build_family_dataset(raw, min_per_class=args.min_per_class)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "issue family: " + " / ".join(sorted(ds["target"].unique())) if len(ds) else "",
        "data": {
            "issues_total": int(len(raw)),
            "usable": int(len(ds)),
            "repos": int(ds["repo"].nunique()) if len(ds) else 0,
            "classes": {k: int(v) for k, v in ds["target"].value_counts().items()},
        },
        "models": {},
    }
    if len(ds) < 60 or ds["target"].nunique() < 2:
        result["status"] = "not enough data yet"
    else:
        train, test = time_split(ds)
        classes = sorted(ds["target"].unique())
        result["data"].update({"train": int(len(train)), "test": int(len(test))})
        result["models"]["tfidf"] = run_tfidf(train, test, classes)
        result["models"]["minilm"] = run_minilm(train, test, classes)
        result["status"] = "ok"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    with open(args.history, "a") as fh:
        fh.write(json.dumps({
            "time": result["generated_at"],
            "status": result["status"],
            "usable": result["data"]["usable"],
            "train": result["data"].get("train", 0),
            "test": result["data"].get("test", 0),
            **{f"{name}_macro_f1": m["macro_f1"] for name, m in result["models"].items()},
            **{f"{name}_accuracy": m["accuracy"] for name, m in result["models"].items()},
        }) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
