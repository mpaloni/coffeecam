#!/usr/bin/env bash
# Export coffeecam to the public GitHub mirror (mpaloni/coffeecam) with
# internal-only content stripped from the *entire history*, not just HEAD:
#   - deploy/                 (systemd units naming homelab hosts)
#   - docs/CAPTURE.md, PIPELINE.md, deployment.md, dataset-experiments-log.md
#   - TODO.md                 (internal dev notes, references deploy/ heavily)
#   - dataset/images/, dataset/previews/, crops/   (raw camera captures)
#   - captures/annotations.jsonl                   (bbox labels tied to capture timestamps)
#   - docs/dataset-experiments-tools/              (k8s-worker training job configs/scripts)
#   - models/*.pt                                  (trained weights)
#
# Gitea (git@gitea:manfred/coffeecam.git) stays the untouched source of truth.
# This script never touches that repo — it only reads from it into a scratch
# clone, rewrites history there with git-filter-repo, and force-pushes the
# result to GitHub.
#
# Re-run any time you want to resync GitHub; it always rebuilds the scratch
# clone from scratch and force-pushes, so GitHub's history is fully replaced
# each time (expected — it's a filtered mirror, not a shared history).

set -euo pipefail

GITEA_REMOTE="git@gitea:manfred/coffeecam.git"
# github-coffeecam is an SSH config alias (~/.ssh/config) pinned to the
# coffeecam-only deploy key, so this push can't use any other identity.
GITHUB_REMOTE="git@github-coffeecam:mpaloni/coffeecam.git"
SCRATCH_DIR="$(mktemp -d /tmp/coffeecam-github-export.XXXXXX)"

EXCLUDE_PATHS=(
  deploy
  docs/CAPTURE.md
  docs/PIPELINE.md
  docs/deployment.md
  docs/dataset-experiments-log.md
  TODO.md
  dataset/images
  dataset/previews
  crops
  models/best-trackB-v1.pt
  models/best-v6-datafix-raw.pt
  dataset/shift_previews.gif
  captures/annotations.jsonl
  docs/dataset-experiments-tools
)

cleanup() { rm -rf "$SCRATCH_DIR"; }
trap cleanup EXIT

echo "==> Cloning $GITEA_REMOTE into scratch dir"
git clone "$GITEA_REMOTE" "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

echo "==> Rewriting history to drop internal-only paths"
invert_args=()
for p in "${EXCLUDE_PATHS[@]}"; do
  invert_args+=(--path "$p")
done
git filter-repo --force "${invert_args[@]}" --invert-paths

echo "==> Verifying no excluded paths survive in any commit"
for p in "${EXCLUDE_PATHS[@]}"; do
  if git log --all --oneline -- "$p" | grep -q .; then
    echo "ERROR: '$p' still present in rewritten history" >&2
    exit 1
  fi
done

echo "==> Verifying no tracked images/model weights remain on HEAD"
if git ls-files | grep -iE '\.(pt|pth|onnx|png|jpg|jpeg|gif|bmp|tiff?|webp|npy|pkl|h5|weights)$'; then
  echo "ERROR: binary artifacts still tracked on HEAD (see above)" >&2
  exit 1
fi
echo "OK: history is clean"

echo "==> Pushing filtered history to $GITHUB_REMOTE"
# filter-repo strips the 'origin' remote by design; re-add pointing at github
git remote add github "$GITHUB_REMOTE"
git push --force github main

echo "==> Done. GitHub main now reflects the filtered history."
