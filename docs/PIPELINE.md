# Detection pipeline + dashboard

`coffeecam/server.py` runs `acquire → normalize → detect → crop → classify` on a
timer and serves every stage over HTTP, so the detector and crop are visible now
and a real fullness classifier can slot in later without touching the server.

```
snapshot ─▶ apply_transform ─▶ match_training_frame ─▶ detect_pot ─▶ frame.crop ─▶ FullnessEstimator
           (rotate 180 +       (black-pad to train    (YOLO, imgsz   (bbox mapped   (BrightnessFullness
            privacy crop)       aspect ratio)          320)           back to frame) placeholder)
```

Run it:

```bash
.venv/bin/python -m coffeecam.server          # http://<host>:8000/
```

| route | content |
|---|---|
| `GET /` | HTML: the four stage images + fullness verdict, self-refreshing |
| `GET /frame.jpg` | acquire output — rotated + privacy-cropped ("privacy filtered version") |
| `GET /normalized.jpg` | black-padded frame the detector actually sees |
| `GET /bounded.jpg` | `frame` with the detection box drawn |
| `GET /crop.jpg` | detected pot region (`404` when nothing detected) |
| `GET /fullness.json` | `{level, score, method, detail}` |
| `GET /pipeline.json` | everything + `timings_ms` + `errors[]` + `stale_seconds` |
| `GET /healthz` | `200` ok / `503` degraded (camera unreachable or result stale) |

Env: `COFFEECAM_SOURCE_URL`, `COFFEECAM_REFRESH_SECS` (10), `COFFEECAM_CONF`
(0.15), `COFFEECAM_NORMALIZE` (1), `COFFEECAM_HARVEST` (0),
`COFFEECAM_HOST`/`COFFEECAM_PORT` (`0.0.0.0`/`8000`).

Deploy: `deploy/systemd/coffeecam-web.service` (user unit, same pattern as the
capture units). Needs a LAN route to `192.168.50.10:8888`.

## `/annotate` — browser labeling

The same server hosts a labeling UI that turns the raw frames in `captures/` into
a training set. Full design: `docs/annotate-endpoint-plan.md`.

- **`GET /annotate`** — canvas over the frame: drag a `coffee_pot` box, corner
  handles + drag-to-move, arrow-key nudge (`Shift` = 10px). `Space` saves and
  advances, `x` saves an explicit negative (no pot), `h` fetches a live-model
  prefill box, `z` undoes. `filter` (unlabeled/labeled/all) and `stride` (label
  every Nth frame — `captures/` is full of near-duplicate heartbeat frames) in
  the side column.
- Labels are written to a **sidecar `captures/annotations.jsonl`**
  (`coffeecam/annotations.py`), one upsertable row per frame keyed by its path
  under `captures/`; `boxes: []` is a kept negative. `captures/` itself is never
  mutated.

| route | content |
|---|---|
| `GET /annotate` | the labeling page |
| `GET /annotate/queue.json` | `{frames:[{i,rel,ts,labeled,boxes}], counts}`; params `filter`, `stride`, `start=YYYY-MM-DD` |
| `GET /annotate/frame/<i>.jpg` | raw frame bytes, no box drawn (`i` indexes the current queue) |
| `GET /annotate/suggest/<i>.json` | `{boxes, source, conf}` from the loaded detector; `503` if no model |
| `POST /annotate/label` | `{rel, boxes}` → upsert the sidecar row (`[]` = negative) |
| `POST /annotate/label/delete` | `{rel}` → `{removed: bool}` |
| `POST /annotate/promote?confirm=1` | runs `dataset.promote` (see below) |

Extra env: `COFFEECAM_DATASET_DIR` (`dataset/`) — where `promote` writes.

**Promote to a trainable dataset:**

```bash
.venv/bin/python -m coffeecam.dataset promote [--dry-run] [--no-negatives]
```

`coffeecam.dataset.promote` reads `annotations.jsonl`, copies each labeled frame
into `dataset/images/` as `cap_YYYYMMDD_HHMMSS[_fff].jpg`, writes its YOLO label
(empty file for a negative), and regenerates `dataset/{train,val,test}.txt`. The
split is a deterministic hash of the frame path (`val_frac`/`test_frac` default
0.15 each); synthetic `kahvi*` frames always stay in `train.txt`, real captured
frames are the only val/test candidates. Idempotent — re-run as labels
accumulate, then retrain.

## Weights

Not committed (~24 MB, git-ignored under `runs/`). `models/CHECKPOINT` is a
one-line pointer to the chosen run directory; `detect.resolve_weights()` reads it,
falling back to the newest `runs/**/weights/best.pt` by mtime. Repoint with:

```bash
.venv/bin/python -c "from coffeecam.detect import promote_weights; from pathlib import Path; \
  promote_weights(Path('runs/detect/runs/<run>'))"
```

Current: `train-nomosaic-2` (150 ep, `mosaic=0`, `imgsz=320`) — val mAP50 0.995 /
mAP50-95 0.971, and `detect.py` on `kahvi.png` now reports conf **0.997** (the old
~0.016 confidence-calibration bug is fixed).

## Known limitations

- **The detector is trained on one scene.** `dataset/` is `kahvi.png` + shift
  augmentations; the val split is 5 shifted copies of that same frame, so mAP
  ~0.99 measures memorization, not generalization. On a live frame the detector
  reports conf ~0.3 bare / ~0.45 normalized and tends to lock onto the nearest
  carafe-shaped object (the mug or the brewer body).
- **`normalize` is a bridge.** Padding the live crop out to the training aspect
  ratio (no downscaling) empirically lifts live confidence ~40%. Downscaling into
  a small letterbox — mimicking the training screenshot literally — makes it
  *worse*. Delete this stage once the detector is retrained on real frames.
- **`fullness` is not a model.** `BrightnessFullness` maps mean luminance of the
  crop's lower-centre to a level (darker ⇒ fuller). Uncalibrated; `method` says
  so. Set `COFFEECAM_HARVEST=1` to accumulate labelled crops, then swap in a
  `ModelFullness` with the same `.estimate()`.
- The privacy crop clips the right edge of the brewer; revisit
  `capture.DEFAULT_INSETS` once real frames show where the carafe actually sits.
