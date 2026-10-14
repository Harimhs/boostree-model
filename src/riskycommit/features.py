"""Turn each ApacheJIT commit into a dense feature vector.

13 base metrics (counted from git) + derived ratios, log scales, time-of-day,
and project identity. Everything is known at the moment of the commit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE = ["la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp", "fix"]

# Readable features the booster trees are allowed to split on (raw units).
READABLE = {
    "fix": "is a bug-fix commit",
    "la": "lines added", "ld": "lines deleted", "churn": "lines changed in total",
    "nf": "files touched", "nd": "directories touched", "ns": "subsystems touched",
    "ent": "how spread-out the change is", "ndev": "developers who touched these files",
    "age": "days since these files last changed", "nuc": "past changes to these files",
    "aexp": "author's past commits", "asexp": "author's commits in this subsystem",
    "subsys_share": "share of author's work in this subsystem",
    "del_ratio": "share of the change that is deletion",
    "hour": "hour of day (UTC)",
}


def load(csv: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["fix"] = df["fix"].astype(str).str.lower().eq("true").astype(float)
    df["buggy"] = df["buggy"].astype(str).str.lower().eq("true").astype(int)
    df = df.sort_values("author_date").reset_index(drop=True)
    df["churn"] = df.la + df.ld
    df["la_per_file"] = df.la / df.nf.clip(lower=1)
    df["del_ratio"] = df.ld / (df.churn + 1)
    df["subsys_share"] = df.asexp / (df.aexp + 1)
    df["recent_share"] = df.arexp / (df.aexp + 1)
    df["hist_per_dev"] = df.nuc / (df.ndev + 0.1)
    df["dir_per_file"] = df.nd / df.nf.clip(lower=1)
    ts = pd.to_datetime(df.author_date, unit="s", utc=True)
    df["hour"] = ts.dt.hour.astype(float)
    df["weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    return df


LOG_COLS = ["la", "ld", "churn", "nf", "nd", "ns", "ndev", "age", "nuc", "aexp", "arexp", "asexp",
            "la_per_file", "hist_per_dev"]
PLAIN_COLS = ["ent", "fix", "del_ratio", "subsys_share", "recent_share", "dir_per_file", "weekend"]


class DenseEncoder:
    """Fixed (non-learned) transform to a standardized dense vector for the network."""

    def __init__(self, include_project: bool = False):
        # Project identity is OFF by default: it lifted test AUC from 0.84 to 0.89 in an
        # ablation, but it is a per-project shortcut (each project labels bugs differently)
        # and cannot transfer to a new repository.
        self.include_project = include_project

    def fit(self, df: pd.DataFrame) -> "DenseEncoder":
        self.projects = sorted(df.project.unique()) if self.include_project else []
        raw = self._raw(df)
        self.mu, self.sd = raw.mean(0), raw.std(0) + 1e-6
        return self

    def _raw(self, df):
        cols = [np.log1p(df[c].clip(lower=0).to_numpy(float)) for c in LOG_COLS]
        cols += [df[c].to_numpy(float) for c in PLAIN_COLS]
        h = df.hour.to_numpy(float) / 24 * 2 * np.pi
        cols += [np.sin(h), np.cos(h)]
        return np.column_stack(cols)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = (self._raw(df) - self.mu) / self.sd
        if not self.projects:
            return x.astype(np.float64)
        onehot = np.column_stack([(df.project == p).to_numpy(float) for p in self.projects])
        return np.hstack([x, onehot]).astype(np.float64)

    @property
    def names(self):
        return LOG_COLS + PLAIN_COLS + ["hour_sin", "hour_cos"] + [f"project={p}" for p in self.projects]
