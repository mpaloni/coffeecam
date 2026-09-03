# Dataset & training loop

How coffee-pot detector training data is made and how models are compared. The loop is:

```
label frames (/annotate)  →  promote to dataset/  →  augment  →  promote again  →  train  →  compare  →  promote weights
```

Reference for the machinery. Current model, in-flight runs and next actions live in
`training-status.md`. Original design notes: `annotate-endpoint-plan.md`.

Most of this landed on branch **`feat/annotate-endpoint`** (still uncommitted / no PR as of
2026-09-02 — see `training-status.md`).

---

## 1. Labeling: `/annotate`

Labels are written to a **sidecar** `captures/annotations.jsonl`, one upsertable row per
frame keyed by its path under `captures/`. **`captures/` is never mutated.** Row types:

| type | JSON | meaning |
|---|---|---|
| positive | `{"rel": ..., "boxes": [[cx,cy,w,h], ...]}` | one or more boxes (normalised) |
| negative | `{"rel": ..., "boxes": []}` | explicit background — becomes an empty YOLO label on promote |
| watched | `{"rel": ..., "boxes": [], "skip": true}` | seen, deliberately not labeled; `promote` **drops these entirely**, so parking uncertain frames here can't poison training |

`skip` is serialised only when true, so pre-existing rows are untouched on rewrite.

### Files

| File | What |
|---|---|
| `coffeecam/annotations.py` | Sidecar store: `load` / `upsert` / `remove` / `validate_boxes` / `skip` / `skip_many` / `_iter_captures`. Pure, lock-free, atomic tmp-file + `os.replace`, rows sorted by `rel`. CLI: `python -m coffeecam.annotations skip-unlabeled [--captures-dir] [--start YYYY-MM-DD] [--dry-run]` — marks every capture with no row as watched (clears an `unlabeled` backlog non-destructively). |
| `coffeecam/annotate.py` | `write_example(image_path, boxes, *, images_dir, labels_dir, class_id=0, dest_name=None)` — shared "copy image + write multi-box YOLO label" primitive. `annotate_image()` CLI (with preview PNG) still works. |
| `coffeecam/server.py` | `/annotate` page + routes: `queue.json`, `frame/<i>.jpg` (raw, no box drawn), `suggest/<i>.json` (live-model prefill, 503 if no model loaded), `POST label`, `POST label/delete`, `POST skip`, `POST skip-queue?start=`, `POST promote?confirm=1`. `filter=unlabeled` (default — excludes labeled **and** watched) / `all` / `watched`. `counts` includes `watched`. Env `COFFEECAM_DATASET_DIR` (default `dataset/`). Linked from `/` footer. Global `_annot_lock`. |

### UI

Inline canvas. Keys: **Space** save + next · **x** negative · **s** skip/watched · **h**
hint (model prefill) · **z** undo. "Skip rest of queue" button (→ `skip-queue`, ignores
`stride` so it leaves no gaps). `watched` filter + amber badge. Un-skip = delete-label /
Backspace on a `filter=watched` frame.

**Frame index `i` in the API is ephemeral** — it indexes the current filtered/strided queue.
Clients always send `rel`, never `i`, on writes.

### Gaps

- **No "distinct-only" filter** (plan §6.2). Near-duplicate heartbeat frames recur in the
  queue until labeled or skipped. `s` / "skip rest of queue" / `skip-unlabeled` are the
  non-destructive clear. The deferred real fix is a ~20-line luma-diff filter using
  `image_summary.py` mean-luma.
- **UI has zero execution coverage** — `_ANNOTATE_PAGE` is only asserted to serve the right
  HTML strings. Canvas coordinate mapping, corner-handle math, key handlers have never run
  in a headless browser. Only 4 corner handles; no multi-box add button.

---

## 2. Promote: `coffeecam/dataset.py`

`promote(...)` turns the sidecar into a trainable `dataset/`:

- reads `annotations.jsonl`, skips `watched` rows
- copies frames to `dataset/images/` as `cap_YYYYMMDD_HHMMSS[_fff].jpg`, writes YOLO labels
  (empty file for a negative)
- regenerates `dataset/{train,val,test}.txt`, maintains `data.yaml` (`train/val/test:
  {split}.txt`, `path: dataset`, one class `coffee_pot`)
- **deterministic 70/15/15 SHA1-of-`rel` split** (`_split_bucket`, seed 0). Synthetic
  `_shift_*` frames are pinned to train; **only real frames are val/test candidates.**
- idempotent

CLI: `python -m coffeecam.dataset promote [--dry-run] [--drop-kahvi] [--drop-kahvi-aug] [--no-negatives]`

- `--drop-kahvi` — exclude `kahvi.png` **and** every `kahvi*` derivative. `kahvi.png` was an
  early off-camera screenshot used as a synthetic seed; it came to dominate train and is now
  retired. `--drop-kahvi-aug` keeps the bare seed, drops only its augs.

`data.yaml`'s `path: dataset` **resolves against the trainer's CWD** — must be `/work` when
training in the k8s pod (see `training-status.md`).

The `/annotate` promote route honours `COFFEECAM_DATASET_DIR`; **the CLI always writes the
real `dataset/`** — be aware when experimenting.

---

## 3. Augment: `coffeecam/augment_shift.py`

Module name kept; extended from shift-only to **shift + rotate + occlude**.

| Function | What |
|---|---|
| `rotate_image` / `rotate_bbox` | rotate `--angle`° CCW about centre (border per `--fill`). Box = axis-aligned bounds of the 4 rotated corners. |
| `apply_occlusions` / `occlusions_for_coverage` | paint opaque rects (`black\|gray\|white\|mean`) over the frame — mimics a hand/mug hiding the pot. Label unchanged; a patch covering ≥90 % of the box is rejected. `occlusions_for_coverage` targets a fractional box coverage. |
| `augment_and_save(..., angle=, occlusions=)` | shift → rotate → occlude in one pass. Stem encodes the transform (`..._shift_x40_y25_rot-8_occ1`); manifest row carries `angle` + `occlusions`; `--replay` reproduces. |
| `generate(source, count, seed=)` / `--generate N` | seeded random batch from one image. |
| `generate_from_list(train_list, per, balanced=, n_shift=, n_rotate=, n_occlude=, occ_min_cover=, occ_max_cover=)` / `--generate-from dataset/train.txt` | augments every positive single-box **train** frame. Reads `train.txt`, so val/test are never touched; skips negatives / multi-box. `--balanced` uses disjoint blocks (pure shift / shift+rot / shift+≥1 occlusion), so the ratio is a knob, not a random draw. |

**Leakage safety:** augmented copies carry a `_shift_x…` stem; `dataset._existing_synthetic`
pins any such file to `train.txt`, so an aug of a val/test frame can never leak into train.
Re-verified 0 leaks on every build. (`_existing_synthetic` filters to image extensions — a
bug where `_shift_x` also matched ultralytics `.npy` cache files is fixed.)

**Rotate-box caveat:** the rotated box is axis-aligned bounds of the corners, looser than a
tight fit, which appears to teach sloppy localisation (see the trackA result below).

---

## 4. Compare: `coffeecam/compare.py` + `/summary?set=` + `/compare`

"Is the new model actually better" in one command, with a picture.

- **`/summary?set=captures|train|val|test`** (`summary.py` + `server.py`) — split values read
  `dataset/<split>.txt`, resolve each image + its YOLO label, and feed the same
  annotate→stitch→GIF path used for capture timelapses. Helpers: `collect_split_frames`,
  `resolve_frames`, `FRAMESETS`, `DEFAULT_DATASET_DIR`; `FrameRef.label_path`; meta carries
  `"frameset"`. Honours `COFFEECAM_DATASET_DIR`; set is in the cache key. CLI:
  `python -m coffeecam.summary --set test`.
- **`compare.py`**
  - `build_comparison_gif(refs, named_models, ...)` — one GIF, a per-model panel (box +
    conf tag + name caption) per frame; returns `(gif_bytes, stats)` with per-model
    `{strong, weak_only, no_box}` counts.
  - `score_models(named_weights, *, split, dataset_dir)` — runs each checkpoint's `.val()`
    on the split for real mAP / P / R at its own train-time imgsz.
  - CLI: `--model NAME=PATH` (repeatable; PATH is a `.pt` or a run dir), `--set` any
    `FRAMESETS` value. mAP auto-skipped for `--set captures` or `--no-map`.
    ```
    .venv/bin/python -m coffeecam.compare \
      --model realdata-v1=runs/detect/runs/train-realdata-v1 \
      --model balanced-v1=models/best-balanced-v1.pt \
      --set test
    ```
- **`/compare` + `/compare.json`** — browser two-model diff. `?model=NAME=spec` repeatable
  (default: `train-nomosaic-2` vs the live `models/CHECKPOINT`); also `?set=`, `frames`,
  `ms`, `scale`, `conf`, `rebuild`. **No mAP** — `.val()` is too slow for a request; the CLI
  is the scored path. Bad spec → 400, empty frame set → 404.

---

## 5. Model experiments — held-out test split

Scored with `compare --set test` at each checkpoint's own imgsz.

| model | train frames | test mAP50 | test mAP50-95 | verdict |
|---|---|---|---|---|
| **nomosaic-2** | 1 real + synth | 0.216 | 0.065 | the old "overfits one scene / mAP ~0.99 = memorization" model; lands a confident box on only 4/18 real frames |
| realdata-v1 | 81 real | 0.562 | 0.443 | first real-data run |
| **balanced-v1** | 550 (81 real + 469 balanced aug) | **0.573** | **0.479** | **promoted**; ~48 % strong boxes on live capture vs nomosaic-2's ~14 % |
| trackA-v1 | 1954 (67 real + 1873 synth, ~1:24) | 0.590 | 0.420 | **rejected** — more confident (P 0.817) but localises *worse*; decision gate was ≥ +0.03 mAP50-95, got −0.059 |
| trackB-v1 | 1202 (147 real + 1026 synth, ~1:7) | *pending* | *pending* | training — see `training-status.md` |

Notes:

- balanced-v1 / trackA-v1 scores were on an **old 18-frame** test split. trackB's split is a
  **different 43 frames** (the labeled set grew), so incumbents **must be re-scored** on the
  new split before any comparison.
- Recurring conclusion: **the bottleneck is real scene count, not augmentation volume or
  model choice.** The detector still doesn't generalise cleanly at ~150–240 real positives.
  Pushing synthetic past ~1:7 real:synth hurt mAP50-95 (trackA).
- `kahvi.png` and its augs were dropped once they dominated train; `--drop-kahvi` enforces it.

---

## Bundled work

The `feat/annotate-endpoint` branch also folds in the previously-uncommitted
**capture-timelapse** foundation this builds on: `coffeecam/summary.py` (+ `image_summary.py`),
`coffeecam/viewer.py`, their `/summary` and `/viewer` routes and tests. The `server.py` hunks
aren't cleanly separable. Consider splitting summary/viewer and compare from annotate when
opening the PR.

Test suite: **152 passing** (`.venv/bin/python -m pytest -q`).
