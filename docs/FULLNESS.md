# Pot-fullness classifier

Reads how full the coffee pot is from the detector's crop. Replaces the old
`BrightnessFullness` luminance heuristic (kept only for offline comparison —
lighting dominated its signal far more than the coffee did).

**Current as of 2026-09-10** (last commit touching `coffeecam/fullness*.py`:
`443fbd5` "keep absent frames in the fullness dataset"). Shipped and live —
not a plan. Live model: `models/FULLNESS_CHECKPOINT` → `fullness-v1-4`
(superseded `fullness-v1`, the version documented in earlier handoff notes).

## How it works

One crop transform, used identically at train and inference time
(`coffeecam/fullness_crop.py::prepare_crop`):

1. expand the box (GT box in training, detector box at inference,
   `DEFAULT_POT_BOX = (291, 113, 373, 205)` — the median hand-drawn box — when
   the detector finds nothing) to a fixed `83:92` aspect,
2. clamp to frame bounds,
3. letterbox-pad to a square (never stretch — a squashed carafe changes the
   fill geometry),
4. resize to `CROP_SIZE = 96`. Never returns `None`.

| module | role |
|---|---|
| `coffeecam/fullness_crop.py` | `prepare_crop` + `DEFAULT_POT_BOX` + `CROP_SIZE`. The one crop transform. |
| `coffeecam/fullness_labels.py` | Sidecar store `captures/fullness.jsonl`. `LABELS` = 5-level scale (`empty/low/half/high/full`) + `absent` (carafe off the warmer — its own trainable class) + `unsure` (legible crop, fill level genuinely unjudgeable — glare/angle/blur; also its own class so the model learns to say "unsure" instead of guessing wrong). `upsert/remove/skip/skip_many/load/counts/positive_box_rels`; CLI `stats`, `skip-unlabeled`. |
| `coffeecam/fullness_dataset.py` | Builds `fullness_dataset/{train,val,test}/{class}/*.jpg`. Split = same deterministic hash-of-`rel` as the detector (`coffeecam.dataset._split_bucket`) — no cross-task leakage. `--merge {none,coarse,binary}`. `absent`/box-less `unsure` frames fall back to `DEFAULT_POT_BOX` rather than being dropped. `--balance` oversamples *train* to ~1:1:1 with jittered crops; val/test keep natural prevalence. Tree wiped+rebuilt each run. |
| `coffeecam/fullness_train.py` | Rebuilds tree (`--merge coarse --balance` default) → fine-tunes `yolov8n-cls.pt` imgsz 96 → writes `models/FULLNESS_CHECKPOINT`. `--no-build` to reuse the tree. |
| `coffeecam/fullness.py` | `ModelFullness` (lazy YOLO), `resolve_fullness_weights()` (`None` on a fresh clone, not an error), `default_estimator()`. `score` = prob-weighted fill scalar via `_FILL_SCALAR`, `None` when the model is confident it's `absent`/`unsure` (off the fill scale). |
| `coffeecam/fullness_compare.py` | `build_test_gif()` — fullness model vs the retired brightness heuristic over the test split. |
| `coffeecam/fullness_confidence.py` | Reads `captures/pipeline/state-*.jsonl`, summarizes the live classifier's own top-1 confidence — distribution, per-level means, detector-miss rate, worst-confidence frames. `python -m coffeecam.fullness_confidence [--low 0.6] [--csv out.csv]`. |

### Server routes (`coffeecam/server.py`)

All frame/crop routes are addressed by `?rel=<path>` (not a positional index —
see [ANNOTATION.md](ANNOTATION.md) for why that matters) and send
`Cache-Control: no-store`.

- `GET /fullness` + `/fullness/{queue.json, crop.jpg, frame.jpg, label,
  label/delete, skip, skip-queue, suggest.json}` — labeling UI, mirrors
  `/annotate`. Queue = frames with a positive box in `annotations.jsonl`.
  `suggest.json` shows the model's own read of the frame before you commit a
  label (mirrors `/annotate/suggest.json`). Queue supports `sort=confidence`.
- `GET /fullness/compare.gif` + `.json` — model vs. brightness-heuristic
  walkthrough over the test split.
- The pipeline worker builds `default_estimator()` **once**, reused across
  every `run_pipeline` call.

## Current data

Re-derive with `.venv/bin/python -m coffeecam.fullness_labels stats` — don't
trust a stale snapshot. As of this writing: **1116 total labels** — `empty
226 · low 219 · half 132 · high 174 · full 32 · absent 74 · unsure 61 ·
watched 198`. `full` is still the scarcest class by a wide margin.

## What's next

- **Data, not model, is the lever** — same conclusion as the detector (see
  [TRAINING.md](TRAINING.md)). `full` needs more distinct brew events, not
  more frames of the same few events (heartbeat dupes don't add signal).
  Actively seed it: run `capture.py --mode stream --interval 5` for ~15 min
  right after someone brews.
- No fullness-specific eval has been re-run against the session-held-out
  methodology that caught the detector's leaky per-frame split (see
  TRAINING.md) — worth checking whether the same 3-4× optimism applies here
  before trusting the per-frame test numbers as a live-behavior estimate.
- Retrain: `.venv/bin/python -m coffeecam.fullness_train`, then eyeball
  `/fullness/compare.gif`.

## Labeling guidance

The `/fullness` crop is `prepare_crop(frame, GT_box)` where `GT_box` is the
hand-drawn box from `/annotate`, not the live detector — the crop is only
wrong when that box was drawn sloppily.

- Crop loose/clipped but still on the carafe → label from whichever image
  reads clearer; a correct level on a slightly-off crop is still good signal.
- Crop on the wrong thing entirely (mis-drawn/stale box) → don't label; fix
  the box in `/annotate` instead, or `skip`.
- `skip`/watched when neither image lets you judge, or for a wrong-box frame
  you won't re-annotate now.
- `unsure` when the crop is legible but the level genuinely can't be judged.
- `absent` when the carafe isn't on the warmer at all.

## Env / commands

```bash
cd services/coffeecam
.venv/bin/pytest tests/ -q
.venv/bin/python -m coffeecam.fullness_labels stats
.venv/bin/python -m coffeecam.fullness_dataset --merge coarse --balance
.venv/bin/python -m coffeecam.fullness_train                # rebuild + train + promote
.venv/bin/python -m coffeecam.fullness_compare               # -> scratchpad/*.gif
systemctl --user restart coffeecam-web.service                # dashboard on :8000
```

Not versioned: `captures/` (frames + label stores), `runs/` (all weights incl.
`fullness-v1-4`), `fullness_dataset/` (derived, rebuilt on demand). Back up
`captures/fullness.jsonl` + `captures/annotations.jsonl` manually — they're
unbacked-up hand-labeled work living in a gitignored directory.
