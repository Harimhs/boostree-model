#!/usr/bin/env bash
# Commit and push the `data` branch checkout, retrying if another workflow
# pushed first (both "Pull data" and "Train" write to it, different files).
# Usage: scripts/push_data_branch.sh <checkout-dir> "<commit message>"
set -euo pipefail
cd "$1"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add -A
if git diff --cached --quiet; then
  echo "Nothing to publish"
  exit 0
fi
git commit -m "$2"
for attempt in 1 2 3 4 5; do
  if git push origin HEAD:data; then
    exit 0
  fi
  echo "Push rejected (attempt $attempt), rebasing on the latest data branch"
  sleep $((attempt * 5))
  git pull --rebase origin data
done
echo "Could not push after 5 attempts" >&2
exit 1
