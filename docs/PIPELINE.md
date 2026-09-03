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
  advances, `x` saves an explicit negative (no pot), `s` marks the frame
  *watched* (seen, not labelled — advances without recording a label), `h`
  fetches a live-model prefill box, `z` undoes. `filter`
  (unlabeled/labeled/watched/all) and `stride` (label every Nth frame —
  `captures/` is full of near-duplicate heartbeat frames) in the side column,
  plus a **skip rest of queue** button that marks every still-unlabelled frame
  watched in one go.
- Labels are written to a **sidecar `captures/annotations.jsonl`**
  (`coffeecam/annotations.py`), one upsertable row per frame keyed by its path
  under `captures/`; `boxes: []` is a kept negative, `{"skip": true}` is a
  watched row. `captures/` itself is never mutated.
- **Three row states**: a box (positive), `[]` (explicit negative → a background
  frame in training), and `skip` (watched → `promote` drops it entirely, so it
  is neither). Use `x` only when you can see there is no pot; use `s` for
  "not now / unsure / duplicate".

| route | content |
|---|---|
| `GET /annotate` | the labeling page |
| `GET /annotate/queue.json` | `{frames:[{i,rel,ts,labeled,skip,boxes}], counts}`; params `filter`, `stride`, `start=YYYY-MM-DD` |
| `GET /annotate/frame/<i>.jpg` | raw frame bytes, no box drawn (`i` indexes the current queue) |
| `GET /annotate/suggest/<i>.json` | `{boxes, source, conf}` from the loaded detector; `503` if no model |
| `POST /annotate/label` | `{rel, boxes}` → upsert the sidecar row (`[]` = negative) |
| `POST /annotate/label/delete` | `{rel}` → `{removed: bool}` (also un-skips a watched row) |
| `POST /annotate/skip` | `{rel}` → mark one frame watched |
| `POST /annotate/skip-queue?start=` | mark every currently-unlabelled frame watched → `{skipped: N}` |
| `POST /annotate/promote?confirm=1` | runs `dataset.promote` (see below) |

CLI to clear an `unlabeled` backlog without the browser:
`python -m coffeecam.annotations skip-unlabeled [--start YYYY-MM-DD] [--dry-run]`.

Extra env: `COFFEECAM_DATASET_DIR` (`dataset/`) — where `promote` writes.

**Promote to a trainable dataset:**

```bash
.venv/bin/python -m coffeecam.dataset promote [--dry-run] [--no-negatives]
```

`coffeecam.dataset.promote` reads `annotations.jsonl`, copies each labeled frame
into `dataset/images/` as `cap_YYYYMMDD_HHMMSS[_fff].jpg`, writes its YOLO label
(empty file for a negative), and regenerates `dataset/{train,val,test}.txt`. The
split is a deterministic hash of the frame path (`val_frac`/`test_frac` default
0.15 each); `_shift_*` augmentation copies always stay in `train.txt`, real
captured frames are the only val/test candidates. Idempotent — re-run as labels
accumulate, then retrain. `--drop-kahvi` excludes the `kahvi.png` seed screenshot
and every `kahvi*` derivative entirely (`--drop-kahvi-aug` keeps the bare seed).

Synthetic frames come from `coffeecam.augment_shift` — shift, rotate
(`--angle`), and occlude (`--occlude`, opaque patches over the pot) a labeled
image. Ways in: a single explicit transform; a seeded random batch from one
image (`--generate N --seed S`); `--generate-from dataset/train.txt --per K`,
which augments every positive single-box **training** frame (real `cap_*` frames
included) with `K` random shift+rotate+occlude combos; or `--generate-from
dataset/train.txt --balanced`, which instead lays down a fixed *recipe* per
frame — `--n-shift` pure translations + `--n-rotate` rotations + `--n-occlude`
occlusions whose covered fraction is swept across
`[--occ-min-cover, --occ-max-cover]` (each occluded copy gets at least one
patch). The recipe is the knob for dataset balance: originals + pure shifts as
the bulk, rotations as a large secondary block, occlusions a deliberate
minority. Because it reads `train.txt` and the outputs carry a `_shift_x…` stem
that `_existing_synthetic()` pins to train, augmented copies never leak into
val/test. Every sample is recorded in `dataset/augmentations.json` (`--replay`
rebuilds them). Re-run `dataset promote` afterwards to fold the new frames into
`train.txt`.

## Weights

Not committed (~24 MB, git-ignored under `runs/`). `models/CHECKPOINT` is a
one-line pointer to the chosen run directory; `detect.resolve_weights()` reads it,
falling back to the newest `runs/**/weights/best.pt` by mtime. Repoint with:

```bash
.venv/bin/python -c "from coffeecam.detect import promote_weights; from pathlib import Path; \
  promote_weights(Path('runs/detect/runs/<run>'))"
```

Current: **`trackB-v1`** (`yolov8n`, 48 ep early-stopped, `mosaic=0`, `imgsz=640`,
trained on k8s worker2). Dataset = 259 hand-labelled real frames (214 pos / 45 neg)
→ 176 real train + 1026 balanced shift/rotate/occlude augs, 40 val / 43 test
real-only. **Held-out test (43 frames): mAP50 0.603 / mAP50-95 0.438**, P 0.69 /
R 0.56, a confident box on 36/43. Replaces `balanced-v1` (which re-scored 0.323 /
0.246 on this split) and the earlier `nomosaic-2` (0.216 / 0.065).

## Known limitations

- **Still doesn't generalise cleanly.** test mAP50-95 0.438 on 43 held-out real
  frames — usable, not solid. ~214 real positives is the bottleneck (Track A
  showed piling on synthetic augs past ~1:7 real:synth *hurts* localisation).
  More real labelled scenes is the lever; keep the `/annotate` loop going.
- **`normalize` is a bridge.** Padding the live crop out to the training aspect
  ratio (no downscaling) empirically lifted live confidence ~40% for the
  screenshot-trained models. With `trackB-v1` trained on real camera frames this
  stage is probably now counter-productive — re-check and likely delete it
  (`TODO.md` follow-ups).
- **`fullness` is not a model.** `BrightnessFullness` maps mean luminance of the
  crop's lower-centre to a level (darker ⇒ fuller). Uncalibrated; `method` says
  so. Set `COFFEECAM_HARVEST=1` to accumulate labelled crops, then swap in a
  `ModelFullness` with the same `.estimate()`.
- The privacy crop clips the right edge of the brewer; revisit
  `capture.DEFAULT_INSETS` once real frames show where the carafe actually sits.
