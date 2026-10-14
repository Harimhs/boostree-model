import pandas as pd

from boostree.model.dataset import build_family_dataset, issue_family, latest_per_issue, time_split


def test_issue_family_needs_exactly_one_target():
    assert issue_family(["type: bug", "P1"]) == "bug"
    assert issue_family(["enhancement", "good first issue"]) == "feature"
    assert issue_family(["bug", "enhancement"]) is None   # ambiguous
    assert issue_family(["P2", "sprint-1"]) is None       # no target family


def test_latest_per_issue_keeps_newest():
    df = pd.DataFrame([
        {"repo": "a/x", "number": 1, "labels": ["question"], "event_time": "2026-10-14T01:00:00Z"},
        {"repo": "a/x", "number": 1, "labels": ["bug"], "event_time": "2026-10-14T03:00:00Z"},
    ])
    out = latest_per_issue(df)
    assert len(out) == 1 and list(out.loc[0, "labels"]) == ["bug"]


def test_build_and_split():
    rows = []
    for i in range(40):
        fam = ["bug", "enhancement"][i % 2]
        rows.append({"repo": f"r/{i % 5}", "number": i, "title": f"t{i}", "body": "b",
                     "labels": [fam], "event_time": f"2026-10-14T{i:02d}:00:00Z"[:20]})
    rows.append({"repo": "r/9", "number": 99, "title": "doc", "body": "", "labels": ["docs"],
                 "event_time": "2026-10-15T00:00:00Z"})
    ds = build_family_dataset(pd.DataFrame(rows), min_per_class=5)
    assert set(ds["target"]) == {"bug", "feature"}  # docs dropped: only 1 example
    train, test = time_split(ds, test_frac=0.25)
    assert len(train) == 30 and len(test) == 10
    assert train["event_time"].max() <= test["event_time"].min()
