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
- Nothing of the fullness classifier is built yet — the plan doc is the only
  artifact.
- Tests green: `.venv/bin/python -m pytest -q` → 152 passing.

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
- Carafe-removed frames: own class or out of scope?
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
