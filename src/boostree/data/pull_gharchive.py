"""Pull real, labeled GitHub issues from GH Archive (https://www.gharchive.org).

GH Archive publishes one gzipped JSON file per hour with every public GitHub
event. We stream each file line by line (never loading it fully in memory),
keep IssuesEvents that carry labels, skip pull requests, and keep the latest
known labels per issue.

Usage:
    python -m boostree.data.pull_gharchive --hours 6 --out data/issues.parquet
    python -m boostree.data.pull_gharchive --local-file sample.json.gz --out x.parquet
"""

import argparse
import gzip
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

GHARCHIVE_URL = "https://data.gharchive.org/{date}-{hour}.json.gz"
MAX_BODY_CHARS = 2000
# GH Archive usually publishes an hour's file about an hour after it ends.
DEFAULT_LAG_HOURS = 2


def hour_urls(hours: int, lag: int, now: datetime | None = None) -> list[tuple[str, str]]:
    """Return (label, url) for the last `hours` complete hours, newest first."""
    now = now or datetime.now(timezone.utc)
    start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=lag)
    out = []
    for i in range(hours):
        t = start - timedelta(hours=i)
        label = f"{t:%Y-%m-%d}-{t.hour}"
        out.append((label, GHARCHIVE_URL.format(date=f"{t:%Y-%m-%d}", hour=t.hour)))
    return out


def parse_hour_label(label: str) -> datetime:
    """'2026-10-14-2' -> datetime(2026, 10, 14, 2, tzinfo=UTC)."""
    date_part, hour = label.rsplit("-", 1)
    day = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return day + timedelta(hours=int(hour))


def hour_urls_after(last_label: str, lag: int, max_hours: int,
                    now: datetime | None = None) -> list[tuple[str, str]]:
    """Hours newer than `last_label` that should be published by now, oldest
    first, capped at `max_hours` (so a long gap catches up over several runs)."""
    now = now or datetime.now(timezone.utc)
    newest = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=lag)
    t = parse_hour_label(last_label) + timedelta(hours=1)
    out = []
    while t <= newest and len(out) < max_hours:
        label = f"{t:%Y-%m-%d}-{t.hour}"
        out.append((label, GHARCHIVE_URL.format(date=f"{t:%Y-%m-%d}", hour=t.hour)))
        t += timedelta(hours=1)
    return out


def label_names(raw_labels) -> list[str]:
    names = []
    for item in raw_labels or []:
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = item
        if name:
            names.append(str(name))
    return names


def parse_lines(lines, stats: dict) -> list[dict]:
    rows = []
    for line in lines:
        stats["events"] += 1
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            stats["bad_lines"] += 1
            continue
        if event.get("type") != "IssuesEvent":
            continue
        stats["issue_events"] += 1
        payload = event.get("payload") or {}
        issue = payload.get("issue") or {}
        if not issue:
            stats["missing_issue_payload"] += 1
            continue
        if "pull_request" in issue:
            stats["pull_requests_skipped"] += 1
            continue
        labels = label_names(issue.get("labels"))
        if not labels:
            continue
        stats["labeled_issue_events"] += 1
        rows.append(
            {
                "repo": (event.get("repo") or {}).get("name", ""),
                "number": issue.get("number"),
                "title": (issue.get("title") or "").strip(),
                "body": (issue.get("body") or "")[:MAX_BODY_CHARS],
                "labels": labels,
                "action": payload.get("action", ""),
                "event_time": event.get("created_at", ""),
            }
        )
    return rows


def stream_url(url: str, stats: dict, retries: int = 3) -> list[dict]:
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                if resp.status_code == 404:
                    print(f"  not published yet: {url}", file=sys.stderr)
                    stats["hours_missing"] += 1
                    return []
                resp.raise_for_status()
                with gzip.GzipFile(fileobj=resp.raw) as gz:
                    text = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
                    return parse_lines(text, stats)
        except (requests.RequestException, OSError, EOFError) as err:
            print(f"  attempt {attempt} failed for {url}: {err}", file=sys.stderr)
            time.sleep(5 * attempt)
    stats["hours_failed"] += 1
    return []


def dedupe(rows: list[dict]) -> pd.DataFrame:
    """Keep the latest event per (repo, issue number)."""
    if not rows:
        return pd.DataFrame(
            columns=["repo", "number", "title", "body", "labels", "action", "event_time"]
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("event_time").drop_duplicates(["repo", "number"], keep="last")
    return df.reset_index(drop=True)


def new_stats() -> dict:
    keys = [
        "events", "bad_lines", "issue_events", "missing_issue_payload",
        "pull_requests_skipped", "labeled_issue_events", "hours_missing", "hours_failed",
    ]
    return {k: 0 for k in keys}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--lag", type=int, default=DEFAULT_LAG_HOURS)
    parser.add_argument("--after", default="",
                        help="continue after this hour label (e.g. 2026-10-14-2); "
                             "--hours then caps how many hours to catch up")
    parser.add_argument("--local-file", help="parse a local .json.gz instead of downloading")
    parser.add_argument("--out", required=True, help="output parquet path")
    parser.add_argument("--stats-out", help="optional JSON file for pull stats")
    args = parser.parse_args(argv)

    stats = new_stats()
    rows: list[dict] = []
    sources = []
    if args.local_file:
        with gzip.open(args.local_file, "rt", encoding="utf-8", errors="replace") as fh:
            rows = parse_lines(fh, stats)
        sources.append(args.local_file)
    else:
        if args.after:
            targets = hour_urls_after(args.after, args.lag, args.hours)
            if not targets:
                print(f"nothing new after {args.after} yet", file=sys.stderr)
                # Marker file lets the workflow skip the remaining steps.
                marker = os.path.join(os.path.dirname(args.out) or ".", "nothing_new")
                open(marker, "w").close()
                return 0
        else:
            targets = hour_urls(args.hours, args.lag)
        for label, url in targets:
            print(f"pulling {label} ...", file=sys.stderr)
            before = len(rows)
            rows.extend(stream_url(url, stats))
            print(f"  +{len(rows) - before} labeled issue events", file=sys.stderr)
            sources.append(label)

    df = dedupe(rows)
    df.to_parquet(args.out, index=False)
    stats["unique_labeled_issues"] = len(df)
    stats["sources"] = sources
    print(json.dumps(stats, indent=2))
    if args.stats_out:
        with open(args.stats_out, "w") as fh:
            json.dump(stats, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
