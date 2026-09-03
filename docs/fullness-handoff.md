# Handoff — pot-fullness classifier

**State as of 2026-09-03.** Focus shifted from detector to **fullness (full/empty)
recognition**. The brightness heuristic (`fullness.BrightnessFullness`) is being
replaced. Full design: [`docs/fullness-plan.md`](fullness-plan.md) — read it first.

## Where things stand

- `main` is at `35b51b0` (pushed to Gitea). Contains: annotate/dataset tooling,
  augment overhaul, `compare.py`, the `fullness-plan.md`/`dataset-and-training.md`/
  `training-status.md` docs, and the committed detector weight.
- **Detector weight is now in git**: `models/best-trackB-v1.pt` (6.0 MB, sha256
  `2d6948b2…`), byte-identical to `runs/detect/runs/trackB-v1/weights/best.pt`.
  `.gitignore` has a `!models/*.pt` exception. `runs/` is still untracked.
- **Step 1 done** (build order below): `coffeecam/fullness_crop.py` —
  `prepare_crop(frame, box)` + `DEFAULT_POT_BOX = (291, 113, 373, 205)` +
  `CROP_SIZE = 96`. Expand box to fixed aspect (83:92) → clamp → letterbox → resize,
  never returns None. Wired into `pipeline.py`: `result.crop` (and the
  `COFFEECAM_HARVEST` output) is now the `prepare_crop` output, and a detector miss
  classifies `prepare_crop(frame, DEFAULT_POT_BOX)` instead of `unknown`. Tests:
  `tests/test_fullness_crop.py` (10). Not yet committed/pushed.
- **Step 2 done**: `coffeecam/fullness_labels.py` (sidecar store for
  `captures/fullness.jsonl`, mirrors `annotations.py`: `upsert`/`remove`/`skip`/
  `skip_many`/`load`/`counts`/`positive_box_rels`, CLI `stats` + `skip-unlabeled`;
  `skip` row = watched). **Class set: went 5-level, not 3.** `LEVELS` reuses
  `coffeecam.fullness.LEVELS` = `empty`/`low`/`half`/`high`/`full`, presented in the
  UI as a **1-5 scale** (1 empty .. 5 full), hotkeys `1`-`5`. (Supersedes the
  "start 3-way" call in `fullness-plan.md` — user asked for the finer scale, and it
  matches the enum `FullnessResult` already uses.) `GET /fullness` page +
  `/fullness/{queue.json,crop/<i>.jpg,frame/<i>.jpg,label,label/delete,skip,skip-queue}`
  in `server.py`, mirroring `/annotate`; queue = frames with a positive box in
  `annotations.jsonl` (373 of them). Crop endpoint serves `prepare_crop(frame, GT_box)`
  so the labeller sees exactly the model input. Sixth label `absent` ("no pot in
  frame", key `w`) for carafe-removed frames — stored like a level, distinct from
  `skip`. `s` skip, filter/stride controls.
  Tests: `tests/test_fullness_labels.py` (14), `tests/test_server.py` fullness cases (8).
- **No labels yet** — `captures/fullness.jsonl` still needs a first ~30-event batch;
  seed the fuller levels first (see TODO.md). Add a `.gitignore` exception for the
  jsonl sidecar when it exists.
- Steps 3–6 not started.
- Tests green: `.venv/bin/python -m pytest -q` → 184 passing.

## Labeling guidance

The `/fullness` crop is `prepare_crop(frame, GT_box)` where `GT_box` is the
**hand-drawn** box from `/annotate`, not the live detector's output — so the crop
is only wrong when that box was drawn sloppily.

- **Crop loose / clipped but still on the carafe** → label from whichever image
  reads clearer (the full frame is shown alongside). A correct level on a
  slightly-off crop is still good signal — train-time box jitter (step 4) is
  built to cover exactly that sloppiness.
- **Crop on the wrong thing entirely** (wall, mug, bare counter — a mis-drawn or
  stale GT box): **do not label it.** The crop *is* the training input, so a
  level here would teach the model "this wall patch = half full". Either re-draw
  the box in `/annotate` (regenerates the crop) then label, or `skip` it to keep
  the frame out of the dataset. This is not `absent` — that means the carafe is
  genuinely off the warmer, not that the box missed it.
- **`skip` / watched** when *neither* image lets you judge the level (glare on
  the glass, motion blur, too dark), or for a wrong-box frame you don't want to
  re-annotate now.
- **`w` / `absent`** when the carafe isn't on the warmer at all.

## Not versioned (know this before relying on it)

- **Real captured frames** (`captures/`, `dataset/images/cap_*.jpg`) — local-only.
- **The labels** `captures/annotations.jsonl` (373 boxes / 45 negatives) — under
  gitignored `captures/`, untracked. The planned `captures/fullness.jsonl` would
  inherit this. Consider a `.gitignore` exception for the jsonl sidecars early.
- **Other weights**: `models/best-balanced-v1.pt` and everything under `runs/`
  (279 MB, 10+ runs) — local-only, not backed up except an ad-hoc
  `scratchpad/seen-frames-backup-20260902.tgz`.

## Data reality

- 373 frames with a coffee-pot box. Box size stable (`w` 83±10, `h` 92±8 px),
  position mostly stable (median `291,113→373,205`) with a nudged/off-angle tail.
- **No fullness labels exist.** First blocker.
- **Class imbalance**: pot is empty most of the day; non-empty frames cluster into
  a few brew events with many near-duplicate heartbeat frames each. Count
  *distinct brew events*, not frames.

## Build order (from the plan)

1. **`coffeecam/fullness_crop.py`** — `prepare_crop(frame, box)` +
   `DEFAULT_POT_BOX = (291, 113, 373, 205)`. Fixed aspect → letterbox → resize to
   `CROP_SIZE=96`, never returns None. Unit tests for pad/resize invariants and
   the static-fallback path. Wire into `pipeline.py` so `result.crop` (and the
   `COFFEECAM_HARVEST` output) is already the model input. Detector-miss → classify
   `prepare_crop(frame, DEFAULT_POT_BOX)` instead of `unknown`.
2. **`GET /fullness` labeling endpoint** in `server.py`, mirroring `/annotate`:
   shows `prepare_crop(frame, GT_box)` + full frame, hotkeys for classes, `s` skip,
   `stride` control. Sidecar **`captures/fullness.jsonl`**, one upsertable row per
   `rel`. Queue = frames that already have a positive box in `annotations.jsonl`.
   CLI fallback like `annotations.py skip-unlabeled`.
   - **Class set:** start 3-way `empty` / `partial` / `full`; drop to binary
     `empty` / `has_coffee` if `partial` stays too rare after the first batch.
   - Label a first ~30-event batch. **Seed `partial`/`full` first** — run
     `capture.py --mode stream --interval 5` for ~15 min right after someone
     brews, a few times across lighting. Add this to `TODO.md`.
3. **`coffeecam/fullness_dataset.py`** — reads `fullness.jsonl` + `annotations.jsonl`,
   writes an ImageFolder tree `fullness_dataset/{train,val,test}/{class}/…` of
   `prepare_crop` outputs. Reuse `dataset.py`'s deterministic hash-of-path split so
   a frame lands in the **same** split as for the detector (no brew-event leakage).
4. **`coffeecam/fullness_train.py`** — `yolov8n-cls` (`yolo classify train
   model=yolov8n-cls.pt data=fullness_dataset imgsz=96`). Oversample minority
   classes in the *train* tree to ~1:1:1; leave val/test at natural prevalence.
   Add `models/FULLNESS_CHECKPOINT` pointer, parallel to `models/CHECKPOINT`.
   Train-time aug: jitter the GT box before `prepare_crop` (scale ×0.85–1.20,
   translate ±12 px) to mimic detector sloppiness.
5. **`ModelFullness` in `fullness.py`** — same `.estimate()` interface, returns
   `FullnessResult(level=argmax, score=P(full-ish), method="yolov8n-cls",
   detail={"probs": …})`. Make it the pipeline default when
   `FULLNESS_CHECKPOINT`/weights resolve; keep `BrightnessFullness` as documented
   fallback, `NullFullness` when no weights. Record the 3×3 confusion matrix +
   **balanced accuracy** + `empty`-vs-`has_coffee` recall on the real held-out
   test split in `docs/training-status.md`. **Never report raw accuracy** (~80 %
   empty ⇒ trivial baseline).
6. Iterate on data (more brew events) before a bigger model. Fallbacks if
   `yolov8n-cls` underperforms: frozen `mobilenet_v3_small` + 3-way head
   (torchvision is installed), or a fill-line `y`-regressor.

## Open questions

- Does the privacy crop (`capture.DEFAULT_INSETS`) clip the carafe's right edge?
  `TODO.md` flags it — check before locking `DEFAULT_POT_BOX`.
- `partial` learnable or binary from the start? Decide after batch 1's class counts.
- ~~Carafe-removed frames: own class or out of scope?~~ **Decided: own label
  `absent`** (keyed `w` in the UI, "no pot in frame"). Stored like a level, not a `skip`;
  `fullness_dataset` decides keep-as-class vs drop.
- Optional cleanup: wire `detect.resolve_weights()` to fall back to `models/*.pt`
  so a fresh clone can run the detector without `runs/`.

## Env / commands

```bash
cd services/coffeecam
.venv/bin/python -m pytest -q                       # 152 passing
.venv/bin/python -m coffeecam.server                # dashboard + /annotate on :8000
COFFEECAM_HARVEST=1 .venv/bin/python -m coffeecam.server   # accumulate crops
```

Branch note: `services/coffeecam` clones can sit on a stale feature branch —
`git branch --show-current` before editing. Currently on `main`.
