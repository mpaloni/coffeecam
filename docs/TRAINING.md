# Detector: dataset, training loop, and current status

```
label frames (/annotate) → promote to dataset/ → augment → promote again → train → compare → promote weights
```

## Current state — verified against `models/CHECKPOINT` and git history

**Live model: `models/best-v6-datafix-raw.pt`** (`models/CHECKPOINT` pointer
updated in `6c602bd`, 2026-09-04). This supersedes `trackB-v1`, which earlier
docs and the README's "Train" section describe as live — that's now stale;
`trackB-v1` was demoted the same day it was promoted-past, once the
session-held-out evaluation below showed it generalized far worse than its
per-frame test score suggested.

v6-datafix-raw's own reported numbers (68-frame `test_sess.txt`, clean
re-labeled data, no synthetic augmentation): **mAP50 0.982 / mAP50-95 0.667**
— a wide margin over every earlier checkpoint on the same split
(`trackB-v1`: 0.247/0.128). A same-day sibling run, `v7-datafix-aug` (same
data + 1:3 synthetic augmentation), tied it (0.668 vs 0.667 mAP50-95) —
another confirmation augmentation isn't the lever here.

**A queued follow-up, `v8-datafix-imgsz960`, was launched 2026-09-04 and never
finished or promoted** — no `models/best-v8*` file exists and `CHECKPOINT`
still points at v6-datafix-raw. `docs/dataset-experiments-log.md` (the
append-only training log) ends mid-run on that experiment; if picking this
back up, check whether the k8s pod that ran it still exists before assuming
lost work.

## The generalization-gap lesson (read this before trusting any mAP number)

The single most important finding from the 2026-09-03/04 experiment burst:
**the original per-frame hash train/val/test split was leaking.** Frames from
the same capture session (same lighting, same brew event, seconds apart)
could land in both train and test, inflating scores. Measured directly on
`trackB-v1`:

| eval split | frames | mAP50 | mAP50-95 |
|---|---|---|---|
| old 43-frame per-frame hash split | 43 | 0.603 | 0.438 |
| session-held-out test | 72 | **0.161** | **0.091** |

A ~3-4× gap between "looks great" and "actually generalizes." **All model
selection since has used session-bucketed splits** (`data_sess*.yaml`,
`coffeecam/dataset-experiments-tools/session_split.py`) — 13 distinct capture
sessions, held-out buckets assigned by day + time-of-day diversity rather
than per-frame. If you see a detector mAP number quoted anywhere in this
repo, check which split it's on before trusting it.

Confirmed conclusions from that round of experiments (session-held-out
evals):

1. **More real frames from the same scenes helps.** capped-v1 (72 reals) →
   realbase-sess (325 reals, same 7 sessions): +0.12 mAP50 / +0.04 mAP50-95.
2. **Synthetic augmentation is a wash** once real data is in the
   hundreds-of-frames range — confirmed independently on trackA-v1 (per-frame
   split), trackB-sess vs realbase-sess, and v7-datafix-aug vs v6-datafix-raw
   (session split). Stop spending training time on augmentation volume/recipe.
3. **The real bottleneck is capture-session / lighting diversity**, not frame
   count or augmentation — 13 sessions total, 7 in train for the mid-scoring
   runs. Next real gain is more *distinct* sessions (different days, lighting,
   pot states).

## How the pipeline works

### 1. Labeling — see [ANNOTATION.md](ANNOTATION.md)

### 2. Promote — `coffeecam/dataset.py`

`promote(...)` turns the `captures/annotations.jsonl` sidecar into a
trainable `dataset/`: copies frames to `dataset/images/`, writes YOLO labels,
regenerates `dataset/{train,val,test}.txt` + `data.yaml`. Deterministic
70/15/15 SHA1-of-`rel` split (`_split_bucket`, seed 0) — synthetic `_shift_*`
frames are pinned to train; only real frames are val/test candidates.
Idempotent.

```
.venv/bin/python -m coffeecam.dataset promote [--dry-run] [--drop-kahvi] [--drop-kahvi-aug] [--no-negatives]
```

`--drop-kahvi` excludes the original off-camera seed screenshot
(`dataset/images/kahvi.png`) and its augmented derivatives — it once
dominated train and is now retired from the "real" set.

### 3. Augment — `coffeecam/augment_shift.py`

Shift + rotate + occlude, composable, label-preserving. Augmented copies
carry a `_shift_x…` stem; `dataset._existing_synthetic` pins them to train so
a val/test frame's augmentation can never leak across the split.

```
.venv/bin/python -m coffeecam.augment_shift --generate-from dataset/train.txt --balanced \
    --n-shift 3 --n-rotate 3 --n-occlude 1 --occ-min-cover 0.1 --occ-max-cover 0.6
```

Given conclusion (3) above, treat this step as available but not currently
worth leaning on for score gains.

### 4. Compare — `coffeecam/compare.py` + `/summary?set=` + `/compare`

`compare.score_models(...)` runs each checkpoint's real `.val()` on a split
for mAP/P/R. `build_comparison_gif(...)` renders a per-model panel GIF.
Browser route `/compare` does the visual diff only (no mAP — too slow for a
request).

```
.venv/bin/python -m coffeecam.compare --model v6=models/best-v6-datafix-raw.pt --set test
```

### 5. Train — k8s worker2/worker1

See `docs/dataset-experiments-tools/` (excluded from the public GitHub
mirror — homelab k8s specifics) for the current pod specs and launch
scripts. worker2 (10 CPU / 15.8 GiB) is the node for imgsz-640 runs; the
coffeecam box itself OOMs above imgsz 320, worker1 above imgsz 416.

```bash
.venv/bin/python -m coffeecam.train --epochs 20
```

writes to `runs/detect/runs/<name>/weights/best.pt`; promote a run by
pointing `models/CHECKPOINT` at it.

## Gotchas carried over from the experiment log

- `data.yaml`'s `path: dataset` resolves against the trainer's CWD — must be
  `/work` when training in a k8s pod.
- Weight files (`.pt`) are gitignored except the two committed exceptions
  under `models/`; `models/CHECKPOINT` is a one-line pointer, not a weight —
  a fresh clone has no `runs/` until you train or copy weights in.
- `detect.resolve_weights()` treats missing weights as a **hard error**
  (unlike the fullness classifier, which falls back to `NullFullness`).

## Tests

```
.venv/bin/pytest tests/ -q
```
