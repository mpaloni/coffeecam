# Detection pipeline + dashboard

`coffeecam/server.py` runs `acquire → normalize → detect → crop → classify` on a
timer and serves every stage over HTTP, so the detector and crop are visible now
and a real fullness classifier can slot in later without touching the server.

> How the detector and fullness classifier are actually invoked — weight
> resolution, the `prepare_crop` bridge, inference calls, outputs — is in
> **`model-pipeline.md`**. (The ASCII diagram just below predates `prepare_crop`
> and imgsz 640; `model-pipeline.md` is current.)

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
| `GET /history`, `/history/long`, … | fullness-state history — see [Fullness-state history](#fullness-state-history) below |

Env: `COFFEECAM_SOURCE_URL`, `COFFEECAM_REFRESH_SECS` (10), `COFFEECAM_CONF`
(0.15), `COFFEECAM_NORMALIZE` (1), `COFFEECAM_HARVEST` (0),
`COFFEECAM_STATE_HISTORY` (1), `COFFEECAM_TRANSITION_ARTIFACTS` (1),
`COFFEECAM_HOST`/`COFFEECAM_PORT` (`0.0.0.0`/`8000`).

Deploy: `deploy/systemd/coffeecam-web.service` (user unit, same pattern as the
capture units). Needs a LAN route to `192.168.50.10:8888`.

## Fullness-state history

A record of what the classifier decided over time, so a bad run can be replayed
and picked apart after the fact instead of only watched live.

### The log

Every pipeline tick appends one line to
`captures/pipeline/state-YYYY-MM-DD.jsonl` (file keyed by the tick's own date, so
a run across midnight splits cleanly). Always on, independent of
`COFFEECAM_HARVEST`, ~150 B/row. Set `COFFEECAM_STATE_HISTORY=0` to disable.

```jsonc
{
  "ts": "2026-09-07T14:23:01",       // tick time, second precision
  "level": "lots",                   // classifier argmax class
  "score": 0.83,                     // 0..1 fill scalar (prob-weighted), null if "absent"
  "p": 0.51,                         // classifier top-1 probability (its confidence); null for Null/Brightness estimators
  "method": "yolov8n-cls",
  "conf": 0.94,                      // detector box confidence; null when the detector found nothing
  "bbox": [281, 129, 360, 228],      // detector box in frame coords; null on a miss
  "timings_ms": { "detect": 197.6, "classify": 17.6, ... },
  "errors": [],
  "frame": "2026-09-07/142301_552.jpg",   // source capture path — backfill only (the live worker keeps no frame)
  "transition": true,                     // present only on rows where level changed vs. the previous tick
  "artifact": "2026-09-07/142301_frame.jpg"  // ditto — the saved transition frame
}
```

On every **level change** the transition tick's `frame.jpg` + `crop.jpg` +
`.json` are also written under `captures/pipeline/<day>/` and referenced from the
row — the frames where the model flipped state are always kept even with
`COFFEECAM_HARVEST` off. Set `COFFEECAM_TRANSITION_ARTIFACTS=0` to skip the
image dump (the row is still written).

### `/history` — one day

| route | content |
|---|---|
| `GET /history` | HTML: a segmented bar of the day's state runs (width ∝ time held, colour per level) + a table of transitions with the saved frame thumbnails |
| `GET /history.json?date=&limit=` | `{date, dates[], rows, runs}` — consecutive same-level ticks collapsed into `runs:[{level, start, end, duration_s, frames, artifact}]`; `artifact` is the frame of the transition *into* that run |
| `GET /history/rows.json?date=&limit=` | the raw per-tick rows, no collapsing |
| `GET /history/artifact/<day>/<stem>_frame.jpg` | a saved transition frame / crop / json (traversal-guarded, jpg + json only) |

`date` defaults to the most recent day with a log; `limit` keeps the last N ticks.

### `/history/long` — N days

| route | content |
|---|---|
| `GET /history/long?days=N` | HTML: inline-SVG graph of fill level over the last N logged days (default 7, max 60). One column per day (00:00→24:00), y = `score` 0..1, stepped line broken across day/gap boundaries, dots coloured by level. "confidence (p)" checkbox overlays the classifier probability. Hover for a `time · level · score · p` readout. |
| `GET /history/long.json?days=N&max=` | `{days, dates[], points:[{t, date, tod, s, l, p}]}` where `tod` is seconds past midnight. `max` (default 4000) evenly strides the payload down for long spans; transition rows are always kept. |

### Backfill

Replay a past day's stored frames into the log — e.g. after adding the feature,
or to reprocess with a new checkpoint:

```bash
.venv/bin/python -m coffeecam.backfill_history 2026-09-07 --force
#   --captures-dir DIR   default captures/
#   --conf F             detector confidence floor (default 0.15)
#   --force              delete an existing state-<day>.jsonl first
```

Reads the kept frames from `captures/<day>/index.jsonl` (falls back to a
`*.jpg` glob), runs each through `run_pipeline(..., transform=False)` — the
stored frames are already privacy-cropped, so the rotate/crop step is skipped —
and feeds the results through the same code path the live worker uses. Backfilled
rows carry `frame` and `p`. **Stop `coffeecam-web` first**: both processes append
the same day file, and only for *today*'s date (past days are safe while it
runs).

### Confidence stats

```bash
.venv/bin/python -m coffeecam.fullness_confidence
#   --date YYYY-MM-DD   restrict to one day (repeatable); default: all logs
#   --low F             low-confidence threshold (default 0.6)
#   --limit N           how many worst frames to list (default 40)
#   --csv PATH          also dump every row to CSV
```

Reads the state logs and reports, over the classifier's own top-1 probability
(`p`):

- overall distribution (min / percentiles / mean) and a coarse histogram
- per-level mean `p` and the share below `--low` — the middle class is usually
  the weak one
- detector-miss count: when there's no box the classifier runs on the static
  `DEFAULT_POT_BOX` crop, and its `p` there is a good "garbage in" signal
- the lowest-`p` frames by path — bad-image candidates to eyeball

Rows written before `p` existed show as "no p"; re-run the backfill to refresh
them.

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
| `GET /annotate/frame.jpg?rel=<rel>` | raw frame bytes, no box drawn; addressed by `rel` (traversal-guarded), `Cache-Control: no-store` |
| `GET /annotate/suggest.json?rel=<rel>` | `{boxes, source, conf}` from the loaded detector; `503` if no model |
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
