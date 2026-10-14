"""Append a pulled batch to a Hugging Face dataset repo, so data accumulates
across runs (GitHub runners are temporary machines).

Needs the HF_TOKEN environment variable (a GitHub Actions secret).

Usage:
    python -m boostree.data.push_hf --file data/issues.parquet --dataset boostree-issues
"""

import argparse
import os
from datetime import datetime, timezone

from huggingface_hub import HfApi


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--dataset", default="boostree-issues", help="dataset repo name")
    args = parser.parse_args(argv)

    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.dataset}"
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y/%m/%d/%H%M%S")
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo=f"issues/{stamp}.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add batch {stamp}",
    )
    print(f"uploaded to https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
