"""Keep a running history of pulls so the dashboard can show trends.

Each run appends one JSON line to history.jsonl (on the `data` branch).

Usage:
    python -m boostree.data.history --summary reports/data_summary.json \
        --history data-branch/history.jsonl
"""

import argparse
import json
import os

MAX_RECORDS = 5000


def record_from_summary(summary: dict) -> dict:
    stats = summary.get("pull_stats", {})
    return {
        "time": summary.get("generated_at"),
        "hours": stats.get("sources", []),
        "events": stats.get("events", 0),
        "issue_events": stats.get("issue_events", 0),
        "labeled_issue_events": stats.get("labeled_issue_events", 0),
        "missing_issue_payload": stats.get("missing_issue_payload", 0),
        "issues": summary.get("issues", 0),
        "repos": summary.get("repos", 0),
        "distinct_labels": summary.get("distinct_labels", 0),
        "family_distinct": {
            fam: info["distinct_names"]
            for fam, info in summary.get("family_name_variety", {}).items()
        },
    }


def read_history(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def last_hour(path: str) -> str | None:
    """Newest GH Archive hour already pulled, e.g. '2026-10-14-2'."""
    hours = [h for rec in read_history(path) for h in rec.get("hours", [])]
    if not hours:
        return None
    from boostree.data.pull_gharchive import parse_hour_label

    return max(hours, key=parse_hour_label)


def append(history_path: str, record: dict) -> int:
    records = read_history(history_path)
    records.append(record)
    records = records[-MAX_RECORDS:]
    with open(history_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return len(records)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--print-last-hour", action="store_true",
                        help="only print the newest pulled hour and exit")
    args = parser.parse_args(argv)

    if args.print_last_hour:
        print(last_hour(args.history) or "")
        return 0
    with open(args.summary) as fh:
        summary = json.load(fh)
    n = append(args.history, record_from_summary(summary))
    print(f"history now has {n} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
