# Pot-fullness classification — plan

Replaces the uncalibrated `BrightnessFullness` luminance heuristic with a trained
classifier that reads the detected pot crop. This doc is the design; nothing here
is built yet.

## Why the heuristic fails

`BrightnessFullness` maps mean luminance of the crop's lower-centre to a level
(darker ⇒ fuller). It breaks because:

- **Lighting dominates the signal.** Overhead lights on vs. off, sun vs. overcast,
  evening corridor spill — all move mean luminance far more than the coffee does.
- **The crop is not stable.** The detector box varies (`w` 83±10, `h` 92±8 px, but
  `x1` ranges 0–307), so "lower-centre" samples different parts of the carafe /
  brewer body frame to frame.
- **No calibration.** `dark_lum=60 / bright_lum=170` are guesses; `method` admits it.
- **Empty ≠ bright, reliably.** An empty glass carafe shows whatever is *behind*
  it (dark brewer housing, warmer plate), not a bright background.

The fix is a learned model over the whole crop, plus a **stable, well-defined
crop** feeding it identically at train and inference time.

## Data on hand

- `captures/annotations.jsonl` — 373 frames with a `coffee_pot` bounding box, 45
  explicit negatives, 244 watched/skip. All real camera frames, fixed camera.
- Box size is stable (`w` 83±10, `h` 92±8); box *position* mostly stable
  (median `291,113 → 373,205`) with a tail of nudged / off-angle frames.
- `captures/pipeline/` — a few dozen harvested `_crop.jpg` from live runs
  (`COFFEECAM_HARVEST=1`).
- **No fullness labels yet.** That is the first blocker.
- **Class balance is bad**: the pot is empty most of the working day. Expect the
  raw frame set to be ~70–85 % empty, and the non-empty frames to cluster into a
  handful of distinct brew events (many near-duplicate heartbeat frames per event).

## The crop: one transform, both paths

The single most important decision. Define **one** function and use it everywhere:

```python
# coffeecam/fullness_crop.py
DEFAULT_POT_BOX = (291, 113, 373, 205)   # median of all GT boxes; static fallback
CROP_SIZE = 96                            # square N×N fed to the classifier

def prepare_crop(frame: Image.Image, box: tuple[int,int,int,int] | None) -> Image.Image:
    """box in `frame` pixels (GT box in training, detector box at inference,
    DEFAULT_POT_BOX when the detector found nothing). Expand to a fixed aspect,
    letterbox to square, resize to CROP_SIZE. Never returns None."""
```

Rules:

1. **Fixed aspect / padding.** Take the box, expand it to a fixed aspect ratio
   (roughly the median `83:92`), clamp to frame bounds, letterbox-pad to square,
   resize to `CROP_SIZE`. No stretching — a squashed carafe changes the fill
   geometry.
2. **Static fallback box.** When the detector returns nothing (happens on ~7/43
   held-out frames today), classify `prepare_crop(frame, DEFAULT_POT_BOX)` instead
   of returning `unknown`. The camera is fixed; the pot is almost always in the
   same ~80×90 px window. This alone makes fullness far more available than it is
   now.
3. **Train/serve skew is handled by augmentation, not by hoping the boxes match.**
   Training crops come from the *GT* box; the detector's live box is looser and
   jittery. So during training, jitter the GT box before `prepare_crop`: random
   scale ×[0.85, 1.20], random translate ±12 px, independent per side. This
   teaches the classifier to tolerate exactly the sloppiness the detector adds.

`pipeline.py` already produces `result.crop` from the detector box — change it to
carry the *box* through and call `prepare_crop` in the classify stage (or in
`FullnessEstimator.estimate`), so the resize/pad is identical to training.

## Labels

Add a labeling endpoint mirroring `/annotate` (`docs/annotate-endpoint-plan.md`):

- **`GET /fullness`** — shows `prepare_crop(frame, GT_box)` (so the labeller sees
  exactly what the model will see), plus the full frame for context. Hotkeys for
  the class set; `s` to skip; auto-advance. `stride` control to skip heartbeat
  near-duplicates.
- Sidecar **`captures/fullness.jsonl`**, one upsertable row per frame `rel`,
  `{"rel": ..., "level": "empty", "labeled_at": ...}`. `captures/` untouched.
- Only frames that already have a positive box in `annotations.jsonl` are in the
  queue (need the box to crop).
- CLI fallback to bulk-label / inspect, like `annotations.py skip-unlabeled`.

### Class set — start coarse

Ship **3 classes**: `empty` / `partial` / `full` (drop to binary `empty` /
`has_coffee` if `partial` stays too rare to learn). The aspirational 5-level
`LEVELS` needs far more data than we'll have soon. `absent` (carafe removed) is a
detector concern — if the detector fires anyway, fold those into their own class
or exclude at label time.

### How much

- Target ~**40–60 distinct brew events** with `partial`/`full` frames, not 40
  frames (heartbeat dupes inflate the count without adding information).
- **Actively seed the minority classes.** Empty accumulates for free. For
  `partial`/`full`: run `capture.py --mode stream --interval 5` for ~15 min right
  after someone brews, a few times, across different lighting. Note this in
  `TODO.md`.

## Splitting — no scene leakage across tasks

Reuse `dataset.py`'s deterministic hash-of-path split so a given frame lands in
the **same** split for the fullness task as for the detector task. Prevents a
brew event's frames straddling train/test. `_shift_*` augmentation copies are
irrelevant here (fullness augments on the fly).

Build an ImageFolder tree for training:

```
fullness_dataset/
  train/ {empty,partial,full}/ cap_YYYYMMDD_HHMMSS.jpg   # prepare_crop output
  val/   ...
  test/  ...
```

`coffeecam/fullness_dataset.py`: reads `fullness.jsonl` + `annotations.jsonl`,
writes the tree, idempotent — same shape as `dataset.promote`.

## Model

### v1 — `yolo classify` (recommended first build)

Ultralytics is already the toolchain (`runs/`, `models/CHECKPOINT`,
`detect.resolve_weights()` all reusable). `yolov8n-cls` on 96 px crops is
~1.5 M params, trains in minutes on CPU, and drops straight into the existing
run-dir / CHECKPOINT machinery.

- `coffeecam/fullness_train.py` → `yolo classify train model=yolov8n-cls.pt
  data=fullness_dataset imgsz=96 ...`.
- **Imbalance:** oversample `partial`/`full` in the train tree (copy with jitter
  seeds) to ~1:1:1; keep val/test at natural prevalence.
- `models/FULLNESS_CHECKPOINT` pointer, parallel to `models/CHECKPOINT`.

### v2 — `ModelFullness` in `fullness.py`

Whatever v1 produces, wrap it behind the existing interface:

```python
@dataclass
class ModelFullness:
    weights: Path
    def estimate(self, crop: Image.Image | None) -> FullnessResult:
        # crop is already prepare_crop output from the pipeline
        # -> FullnessResult(level, score, method="yolov8n-cls", detail={"probs": ...})
```

`FullnessResult` gains nothing structural; `score` becomes the model's
`P(full)`-style scalar (or `argmax` prob), `level` the argmax class, `detail`
carries the full prob vector. `NullFullness` stays the honest default when no
weights are present. `BrightnessFullness` stays in the tree as a documented
fallback but is no longer the pipeline default once weights exist.

### If v1 underperforms

- Transfer-learn a `mobilen_v3_small` (torchvision is installed) backbone, frozen,
  + a 3-way linear head. More robust than from-scratch on small data.
- Or a **fill-line regressor**: the camera is fixed, so the liquid meniscus is a
  near-horizontal edge whose `y` encodes volume. Regress that `y`, map to level.
  Elegant, needs continuous labels — defer unless classes plateau.

## Metrics — never report raw accuracy

~80 % empty means a "always empty" baseline scores 80 %. Report:

- **Per-class recall** and the **3×3 confusion matrix** on the real held-out test
  split.
- **Balanced accuracy** (mean per-class recall) as the headline number.
- `empty` vs `has_coffee` recall specifically — that's the user-facing signal
  ("is there coffee right now?").
- Include detector-miss frames (classified via `DEFAULT_POT_BOX`) in the test set
  so the number reflects live behaviour.

## Build order

1. `fullness_crop.py` — `prepare_crop` + `DEFAULT_POT_BOX`; unit tests (pad/resize
   invariants, fallback path). Wire into `pipeline.py` so the crop the harvest
   saves is already the model input.
2. `/fullness` labeling endpoint + `fullness.jsonl` + CLI. Label a first
   ~30-event batch, seeding `partial`/`full` captures first.
3. `fullness_dataset.py` — build the ImageFolder tree, shared split with
   `dataset.py`.
4. `fullness_train.py` — `yolov8n-cls`, oversampled train, `FULLNESS_CHECKPOINT`.
5. `ModelFullness` in `fullness.py`; make it the pipeline default when weights
   resolve. Record test confusion matrix in `docs/training-status.md`.
6. Iterate on data (more brew events) before reaching for a bigger model.

## Open questions

- Does the privacy crop (`capture.DEFAULT_INSETS`) clip the carafe's right edge?
  `TODO.md` already flags this — check before locking `DEFAULT_POT_BOX`, since a
  clipped carafe loses fill signal.
- Is `partial` learnable, or collapse to binary from the start? Decide after the
  first label batch shows the class counts.
- Carafe-removed frames: separate class, or out of scope for fullness?
