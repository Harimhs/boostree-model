"""Write a human-readable summary of pulled issues (renders nicely on GitHub mobile).

Usage:
    python -m boostree.data.summarize --data data/issues.parquet \
        --stats data/pull_stats.json --out-md reports/data_summary.md \
        --out-json reports/data_summary.json
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import pandas as pd

from boostree.data.labels import family

MIN_ISSUES_FOR_REPO = 5


def summarize(df: pd.DataFrame, stats: dict) -> dict:
    label_counts = Counter(label for labels in df["labels"] for label in labels)
    per_repo = df.groupby("repo").size().sort_values(ascending=False)

    # How many different raw names does each label family have across repos?
    family_names: dict[str, Counter] = defaultdict(Counter)
    for label, count in label_counts.items():
        family_names[family(label)][label] += count

    labels_per_issue = df["labels"].map(len) if len(df) else pd.Series(dtype=int)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pull_stats": stats,
        "issues": int(len(df)),
        "repos": int(df["repo"].nunique()) if len(df) else 0,
        "repos_with_5plus_issues": int((per_repo >= MIN_ISSUES_FOR_REPO).sum()),
        "distinct_labels": len(label_counts),
        "avg_labels_per_issue": round(float(labels_per_issue.mean()), 2) if len(df) else 0,
        "issues_with_body": int((df["body"].str.len() > 0).sum()) if len(df) else 0,
        "top_labels": label_counts.most_common(25),
        "top_repos": [[repo, int(n)] for repo, n in per_repo.head(15).items()],
        "family_name_variety": {
            fam: {"distinct_names": len(names), "top_names": names.most_common(8)}
            for fam, names in sorted(family_names.items())
        },
        "examples": df.head(5)[["repo", "number", "title", "labels"]]
        .assign(labels=lambda d: d["labels"].map(list))
        .to_dict(orient="records")
        if len(df)
        else [],
    }


def to_markdown(s: dict) -> str:
    p = s["pull_stats"]
    lines = [
        "# Data summary",
        "",
        f"_Generated {s['generated_at']} from GH Archive hours: "
        f"{', '.join(map(str, p.get('sources', [])))}_",
        "",
        "## Pull",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| All events scanned | {p.get('events', 0):,} |",
        f"| Issue events | {p.get('issue_events', 0):,} |",
        f"| Issue events with labels | {p.get('labeled_issue_events', 0):,} |",
        f"| Issue events missing issue details | {p.get('missing_issue_payload', 0):,} |",
        f"| Hours not yet published / failed | {p.get('hours_missing', 0)} / {p.get('hours_failed', 0)} |",
        "",
        "## Dataset",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Unique labeled issues | {s['issues']:,} |",
        f"| Repos | {s['repos']:,} |",
        f"| Repos with 5+ labeled issues | {s['repos_with_5plus_issues']:,} |",
        f"| Distinct label names | {s['distinct_labels']:,} |",
        f"| Avg labels per issue | {s['avg_labels_per_issue']} |",
        f"| Issues with a description | {s['issues_with_body']:,} |",
        "",
        "## Same idea, different names",
        "",
        "How many different label names repos use for each kind of label. "
        "This is the variety the booster trees must learn.",
        "",
        "| Family | Distinct names | Most common names |",
        "|---|---|---|",
    ]
    for fam, info in s["family_name_variety"].items():
        names = ", ".join(f"`{n}` ({c})" for n, c in info["top_names"][:5])
        lines.append(f"| {fam} | {info['distinct_names']} | {names} |")
    lines += ["", "## Top labels", "", "| Label | Issues |", "|---|---|"]
    lines += [f"| `{label}` | {n} |" for label, n in s["top_labels"][:15]]
    lines += ["", "## Top repos", "", "| Repo | Labeled issues |", "|---|---|"]
    lines += [f"| {repo} | {n} |" for repo, n in s["top_repos"][:10]]
    lines += ["", "## Examples", ""]
    for ex in s["examples"]:
        labels = ", ".join(f"`{x}`" for x in ex["labels"])
        lines.append(f"- **{ex['repo']}#{ex['number']}**: {ex['title'][:100]} → {labels}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.data)
    with open(args.stats) as fh:
        stats = json.load(fh)
    summary = summarize(df, stats)
    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    with open(args.out_md, "w") as fh:
        fh.write(to_markdown(summary))
    print(to_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
