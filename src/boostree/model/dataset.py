"""Turn accumulated raw issues into a supervised dataset.

The baseline task: label an issue as one of the shared "families"
(bug / feature / docs / question), whatever each repo calls them.
Issues whose labels map to exactly one target family are used; issues with
none or several are skipped (they are ambiguous for a single-label baseline).
"""

from __future__ import annotations

import glob
import os

import pandas as pd

from boostree.data.labels import family

TARGET_FAMILIES = ["bug", "feature", "docs", "question"]
MAX_BODY_CHARS = 1000


def load_issues_from_hf(dataset: str = "boostree-issues", token: str | None = None,
                        local_dir: str = "hf-data") -> pd.DataFrame:
    """Download every pulled batch from the Hugging Face dataset and merge them."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/{dataset}"
    path = snapshot_download(repo_id, repo_type="dataset", token=token,
                             allow_patterns=["issues/**"], local_dir=local_dir)
    files = sorted(glob.glob(os.path.join(path, "issues", "**", "*.parquet"), recursive=True))
    if not files:
        return pd.DataFrame(columns=["repo", "number", "title", "body", "labels", "event_time"])
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return latest_per_issue(df)


def latest_per_issue(df: pd.DataFrame) -> pd.DataFrame:
    """The same issue can appear in several hours; keep its newest labels."""
    df = df.sort_values("event_time")
    return df.drop_duplicates(["repo", "number"], keep="last").reset_index(drop=True)


def issue_family(labels) -> str | None:
    fams = {family(l) for l in labels} & set(TARGET_FAMILIES)
    return fams.pop() if len(fams) == 1 else None


def build_family_dataset(df: pd.DataFrame, min_per_class: int = 15) -> pd.DataFrame:
    """Return rows with `text`, `target`, `repo`, `event_time`, keeping only
    classes with at least `min_per_class` examples."""
    out = df.copy()
    out["target"] = out["labels"].map(lambda ls: issue_family(list(ls)))
    out = out.dropna(subset=["target"])
    out["text"] = (out["title"].fillna("") + "\n" +
                   out["body"].fillna("").str.slice(0, MAX_BODY_CHARS)).str.strip()
    out = out[out["text"].str.len() > 0]
    counts = out["target"].value_counts()
    keep = counts[counts >= min_per_class].index
    out = out[out["target"].isin(keep)]
    return out[["repo", "number", "text", "target", "event_time"]].reset_index(drop=True)


def time_split(ds: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on older issues, test on the newest ones (like real deployment)."""
    ds = ds.sort_values("event_time").reset_index(drop=True)
    cut = int(len(ds) * (1 - test_frac))
    return ds.iloc[:cut].reset_index(drop=True), ds.iloc[cut:].reset_index(drop=True)
