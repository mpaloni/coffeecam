# How the two models are used

`coffeecam` runs **two independent models** on every camera frame:

1. a **detector** (`yolov8n`) that *finds* the coffee pot, and
2. a **fullness classifier** (`yolov8n-cls`) that reads *how full* it is.

They are wired together in `coffeecam/pipeline.py::run_pipeline`, called on a
timer by `coffeecam/server.py`. Design background: [FULLNESS.md](FULLNESS.md)
(classifier), [TRAINING.md](TRAINING.md) (detector). Live routes:
[PIPELINE.md](PIPELINE.md).

```
snapshot
  │  apply_transform          rotate 180° + privacy crop        → frame
  │  match_training_frame      black-pad to training aspect      → normalized, pad   (if normalize=True)
  │
  ├─▶ DETECTOR  detect_pot(normalized or frame, model, conf=0.15, imgsz=640)
  │     model.predict → keep highest-confidence `coffee_pot` box → Detection | None
  │     map_bbox_back(bbox, pad) so the box is in `frame` pixels
  │
  │  prepare_crop(frame, detection.bbox  or  DEFAULT_POT_BOX)   → 96×96 square crop
  │
  └─▶ CLASSIFIER  estimator.estimate(crop)
        model.predict → argmax class + prob vector → FullnessResult(level, score, detail)
```

Every stage is wrapped: a stage that raises is recorded in
`PipelineResult.errors` and the pipeline continues **degraded** rather than
crashing the server loop.

---

## 1. Finding the pot — the detector

**Module:** `coffeecam/detect.py` · **Model:** fine-tuned `yolov8n` object
detector · **Class emitted:** one box for `coffee_pot`.

### Weight resolution

Weights are ~24 MB and live in git-ignored `runs/`, so they are **not
committed**. `detect.resolve_weights()` picks them in this order:

1. explicit path argument, else
2. the one-line pointer file `models/CHECKPOINT` (currently
   `models/best-v6-datafix-raw.pt` — see TRAINING.md for how that got
   promoted over the earlier `trackB-v1`), else
3. newest `runs/**/weights/best.pt` by mtime (`find_latest_weights`).

Missing weights here **is a hard error** (unlike the classifier). `load_model()`
constructs the `ultralytics.YOLO` object **once**; the pipeline worker holds it
and passes it into every `run_pipeline` call — never reloaded per frame.

### Inference

`detect_pot(image, model, *, conf=0.25, imgsz=640)`:

- `model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)[0]`
- `imgsz` **must match the checkpoint's training size** — `detect.DEFAULT_IMGSZ = 640`.
- The pipeline passes `conf=0.15` (`pipeline.DEFAULT_CONF`), lower than the
  CLI default — live frames detect at ~0.29 bare / ~0.41 normalized, so 0.15
  clears the normalized path with margin.
- Of every returned box it keeps the **single highest-confidence** one.
- Returns `Detection(x1, y1, x2, y2, confidence)` or `None` (no boxes).

### Normalization round-trip

When `normalize=True` (default), `match_training_frame(frame)` black-pads the
frame to the detector's training aspect ratio and returns `(normalized, pad)`.
The detector predicts on `normalized`; the box is then mapped back into real
`frame` coordinates with `map_bbox_back(raw_det.bbox, pad)`. With
`normalize=False` the detector runs directly on `frame` and no mapping is
needed. Degenerate mapped boxes (`x2 <= x1` or `y2 <= y1`) are discarded →
`detection = None`.

---

## 2. Bridging the models — `prepare_crop`

**Module:** `coffeecam/fullness_crop.py`. Full detail in FULLNESS.md. Given
the detector box (or `DEFAULT_POT_BOX` when nothing was detected), expands to
a fixed aspect, clamps, letterbox-pads to a square, resizes to 96px — the
same transform used at fullness-training time, so train/serve skew is bounded.

---

## 3. Classifying the state — the fullness model

**Module:** `coffeecam/fullness.py` · **Model:** `yolov8n-cls` over the 96 px
crop, trained by `coffeecam.fullness_train`.

`default_estimator()` returns **`ModelFullness`** when
`resolve_fullness_weights()` finds weights (`models/FULLNESS_CHECKPOINT`,
currently pointing at `fullness-v1-4`), else **`NullFullness`** — honest
`level="unknown", score=None` on a fresh clone with no `runs/`. **Absence is
not an error here** (contrast the detector). `BrightnessFullness` is the
retired placeholder, kept only for offline comparison. The pipeline worker
builds the estimator **once** and passes it into every `run_pipeline` call.

`ModelFullness.estimate`: `crop is None` → `unknown`; else predicts, takes
argmax class as `level`, a probability-weighted 0..1 fill scalar as `score`
(`None` when the model is confident it's off-scale — `absent` or `unsure`),
full probability vector in `detail["probs"]`. A classify stage that raises
degrades to `FullnessResult("unknown", None, "error")` — the loop keeps
running.

---

## Output — `PipelineResult`

| field | what it is |
|---|---|
| `frame` | privacy-cropped frame (rotate 180 + crop); the raw snapshot is never retained |
| `normalized` | black-padded frame the detector saw (`None` if `normalize=False` or the stage failed) |
| `bounded` | `frame` with the detection box drawn in red (a copy of `frame` when nothing detected) |
| `crop` | `prepare_crop` output — the classifier's exact input; `None` only if `apply_transform` failed |
| `detection` | `Detection` in `frame` coordinates, or `None` |
| `fullness` | `FullnessResult(level, score, method, detail)` |
| `timings_ms` | per-stage wall time |
| `errors` | non-fatal stage failures; `ok` is `not errors` |

`coffeecam/server.py` renders these: `/bounded.jpg` and `/crop.jpg` show the
two models' visible output, `/fullness.json` returns
`{level, score, method, detail}`, `/pipeline.json` returns everything plus
`timings_ms`, `errors[]` and `stale_seconds`.

---

## Retraining touch-points

| model | rebuild + retrain | promote |
|---|---|---|
| detector | TRAINING.md (k8s worker2, imgsz 640) | `detect.promote_weights()` / edit `models/CHECKPOINT` |
| fullness | `python -m coffeecam.fullness_train` (rebuilds `fullness_dataset/`, fine-tunes `yolov8n-cls.pt`, writes the pointer) | written automatically by `fullness_train` |

Neither model's weights are in git; `models/CHECKPOINT` and
`models/FULLNESS_CHECKPOINT` (committed one-line pointers) are how a checkout
finds them, and both fall back gracefully when `runs/` is absent (detector →
newest-by-mtime or error; fullness → `NullFullness`).
