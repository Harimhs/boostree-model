import gzip
import json
from datetime import datetime, timezone

import pandas as pd

from boostree.data.labels import family
from boostree.data.pull_gharchive import dedupe, hour_urls, main, new_stats, parse_lines


def event(type_, repo, number, labels, action="labeled", t="2026-10-14T05:00:00Z", pr=False):
    issue = {"number": number, "title": f"Issue {number}", "body": "details",
             "labels": [{"name": l} for l in labels]}
    if pr:
        issue["pull_request"] = {"url": "x"}
    return json.dumps({"type": type_, "repo": {"name": repo}, "created_at": t,
                       "payload": {"action": action, "issue": issue}})


def test_parse_keeps_only_labeled_non_pr_issues():
    lines = [
        event("IssuesEvent", "a/x", 1, ["bug"]),
        event("IssuesEvent", "a/x", 2, []),                 # no labels -> skip
        event("IssuesEvent", "a/x", 3, ["bug"], pr=True),   # PR -> skip
        event("PushEvent", "a/x", 4, ["bug"]),              # not an issue -> skip
        "not json",
    ]
    stats = new_stats()
    rows = parse_lines(lines, stats)
    assert [r["number"] for r in rows] == [1]
    assert stats["issue_events"] == 3
    assert stats["pull_requests_skipped"] == 1
    assert stats["bad_lines"] == 1


def test_dedupe_keeps_latest_labels():
    rows = parse_lines([
        event("IssuesEvent", "a/x", 1, ["question"], t="2026-10-14T01:00:00Z"),
        event("IssuesEvent", "a/x", 1, ["bug"], t="2026-10-14T02:00:00Z"),
    ], new_stats())
    df = dedupe(rows)
    assert len(df) == 1 and list(df.loc[0, "labels"]) == ["bug"]


def test_hour_urls_format():
    now = datetime(2026, 10, 14, 9, 30, tzinfo=timezone.utc)
    urls = hour_urls(2, lag=2, now=now)
    assert urls[0][1].endswith("/2026-10-14-7.json.gz")
    assert urls[1][1].endswith("/2026-10-14-6.json.gz")


def test_hour_urls_after_catches_up_in_order_with_cap():
    from boostree.data.pull_gharchive import hour_urls_after

    now = datetime(2026, 10, 14, 9, 30, tzinfo=timezone.utc)
    labels = [l for l, _ in hour_urls_after("2026-10-14-3", lag=2, max_hours=6, now=now)]
    assert labels == ["2026-10-14-4", "2026-10-14-5", "2026-10-14-6", "2026-10-14-7"]
    capped = [l for l, _ in hour_urls_after("2026-10-13-20", lag=2, max_hours=3, now=now)]
    assert capped == ["2026-10-13-21", "2026-10-13-22", "2026-10-13-23"]
    assert hour_urls_after("2026-10-14-7", lag=2, max_hours=6, now=now) == []


def test_history_append_and_last_hour(tmp_path):
    from boostree.data import history

    path = tmp_path / "history.jsonl"
    summary = {"generated_at": "t1", "issues": 5, "repos": 2, "distinct_labels": 3,
               "pull_stats": {"sources": ["2026-10-14-9", "2026-10-14-10"], "events": 9},
               "family_name_variety": {"bug": {"distinct_names": 4, "top_names": []}}}
    history.append(str(path), history.record_from_summary(summary))
    rec = history.read_history(str(path))[0]
    assert rec["family_distinct"] == {"bug": 4}
    assert history.last_hour(str(path)) == "2026-10-14-10"  # numeric, not string order


def test_family():
    assert family("type: bug") == "bug"
    assert family("kind/feature") == "feature"
    assert family("🐛 Bug") == "bug"
    assert family("documentation") == "docs"
    assert family("good first issue") == "other"
    assert family("help wanted") == "other"
    assert family("needs pull request") == "other"
    assert family("feature request") == "feature"


def test_end_to_end_local_file(tmp_path):
    src = tmp_path / "hour.json.gz"
    with gzip.open(src, "wt") as fh:
        fh.write("\n".join([event("IssuesEvent", "a/x", 1, ["bug"]),
                            event("IssuesEvent", "b/y", 7, ["type: bug", "P1"])]))
    out = tmp_path / "issues.parquet"
    assert main(["--local-file", str(src), "--out", str(out)]) == 0
    df = pd.read_parquet(out)
    assert sorted(df["repo"]) == ["a/x", "b/y"]
