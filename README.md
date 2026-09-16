# coffeecam

YOLOv8-based detector that finds the coffee pot in a camera frame and crops it out — step 1 toward a fullness-monitoring camera for the break-room coffee maker (`kahvi` = Finnish for coffee).

Detection + a placeholder fullness heuristic, for now. A trained fill-level model drops into the same pipeline slot later, once enough real frames (across fill levels) are collected.

Frames are pulled straight from the camera by `coffeecam/capture.py` (see [Capture](#capture) and [`docs/CAPTURE.md`](docs/CAPTURE.md)); the original dataset seed was a single screenshot (`dataset/images/kahvi.png`).

Gitea (`git@gitea:manfred/coffeecam.git`) is the source of truth, with the full history — deploy configs, internal runbooks, dataset artifacts, model weights. [`github.com/mpaloni/coffeecam`](https://github.com/mpaloni/coffeecam) is a filtered public mirror (MIT-licensed); `scripts/export-github.sh` rebuilds it from Gitea, stripping homelab-internal paths and binary artifacts — see the exclusion list at the top of that script for exactly what's cut.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`ultralytics` downloads COCO-pretrained YOLOv8n weights on first use — needs internet.

## Capture

Pull frames from the break-room camera (exposed on the homelab LAN via the
tunnel in homelab-docs `runbooks/coffeecam-tunnel.md`; address configured in
`hosts.env`, see `hosts.env.example`). Frames are rotated 180° and
privacy-cropped before they hit disk.

```bash
.venv/bin/python -m coffeecam.capture --once                 # one frame now
.venv/bin/python -m coffeecam.capture --interval 60 \
    --days mon-fri --hours 7 19                              # 1/min during the workday
.venv/bin/python -m coffeecam.capture --mode stream --interval 2 \
    --dedup --heartbeat-secs 300 --days mon-fri --hours 7 19 # keep only distinct frames
```

Frames land in `captures/YYYY-MM-DD/HHMMSS.jpg` (git-ignored) with a per-day
`index.jsonl` event log. Run it as a service with one of the units in
`deploy/systemd/`. Sizing, dedup keep-rates, and setup: [`docs/CAPTURE.md`](docs/CAPTURE.md).

## Usage

**1. Label a bounding box for an image** (no GUI labeling tool set up — pixel coords in, YOLO label + preview PNG out):

```bash
.venv/bin/python -m coffeecam.annotate dataset/images/kahvi.png <x1> <y1> <x2> <y2>
```

Check `dataset/previews/<name>_preview.png` to confirm the box looks right.

**1b. Grow the dataset from a labeled image** — three cheap, label-preserving transforms, composable in one call:

- **shift** — translate content (`--dx > 0` right, `--dy > 0` down); the box moves with it
- **rotate** — `--angle` degrees counter-clockwise about the frame centre; the box becomes the axis-aligned box of its rotated corners
- **occlude** — `--occlude x1,y1,x2,y2[,fill]` (repeatable, `fill` in `black|gray|white|mean`) paints an opaque patch over the frame to mimic a hand/mug partly hiding the pot; the label is left alone (patches covering ≥90 % of the box are rejected)

```bash
.venv/bin/python -m coffeecam.augment_shift dataset/images/kahvi.png --dx 40 --dy 25 --angle -8 --occlude 90,120,140,220,gray [--fill edge|reflect|wrap|black]

# a seeded batch of random shift+rotate+occlude combos from one image:
.venv/bin/python -m coffeecam.augment_shift --generate 60 --seed 0 [--max-shift 130 --max-angle 15]

# ...or from every real training frame at once (run right after `dataset promote`,
# so it reads dataset/train.txt and the copies stay pinned to train — no leakage):
.venv/bin/python -m coffeecam.augment_shift --generate-from dataset/train.txt --per 4

# ...or with a controlled balance per frame instead of --per random combos:
# N pure shifts + N rotations + N occlusions (>=1 patch each, "percent blocked"
# swept across [--occ-min-cover, --occ-max-cover]). Originals + shifts are the
# bulk, rotations "many", occlusions a deliberate minority:
.venv/bin/python -m coffeecam.augment_shift --generate-from dataset/train.txt \
    --balanced --n-shift 3 --n-rotate 3 --n-occlude 1 --occ-min-cover 0.1 --occ-max-cover 0.6
.venv/bin/python -m coffeecam.dataset promote --drop-kahvi   # fold copies in; kahvi.png + derivs excluded
```

Per sample it writes the transformed raw image to `dataset/images/<stem>.png` (stem encodes the transform, e.g. `kahvi_shift_x40_y25_rot-8_occ1`), the updated YOLO label to `dataset/labels/…`, a boxed preview to `dataset/previews/…`, and appends the full transform (dx/dy/fill/angle/occlusions + old & new bbox) to `dataset/augmentations.json`. That manifest is the record of every augmentation; `--replay` regenerates all of them from it. Augmented images/labels live alongside the originals; run `dataset promote` afterwards to fold them into `train.txt`.

Stitch every shift preview into one animated GIF (box sweeps around the frame, each frame stamped with its dx/dy):

```bash
.venv/bin/python -m coffeecam.preview_anim   # -> dataset/shift_previews.gif
```

**1c. Label lots of real frames in the browser** — for the ~250 frames in `captures/`, the `/annotate` endpoint on the dashboard beats hand-typing pixel corners: draw a box per frame (live-model prefill with `h`, `x` for an explicit "no pot" negative, `s` to mark a frame *watched* — seen but not labelled, so it drops out of the queue without becoming a training negative, `Space` to save + advance). Labels go to a sidecar `captures/annotations.jsonl`, not straight into `dataset/`. Then build the trainable set:

```bash
.venv/bin/python -m coffeecam.dataset promote            # annotations.jsonl -> dataset/images,labels + train/val/test.txt
```

`promote` is deterministic and idempotent (hash split; real frames only in val/test, `_shift_*` copies pinned to train) — re-run it as labels accumulate. `--drop-kahvi` excludes the original `kahvi.png` screenshot and every `kahvi*` derivative from `train.txt` (the real captures now carry training); `--drop-kahvi-aug` is the softer form that keeps the bare seed frame. Details: [`docs/PIPELINE.md`](docs/PIPELINE.md#annotate--browser-labeling).

**2. Train:**

```bash
.venv/bin/python -m coffeecam.train --epochs 20
```

Writes to `runs/detect/runs/<name>/weights/best.pt`; `models/CHECKPOINT` points at the promoted run. The live model is **`trackB-v1`** — `yolov8n` trained on 259 hand-labelled real frames + balanced augmentation (imgsz 640, `mosaic=0`, on k8s), scoring **mAP50 0.603 / mAP50-95 0.438 on 43 held-out real frames**. Usable but not solid; real labelled scene count is the bottleneck, so keep feeding the `/annotate` loop (imgsz 640 needs a box with more than ~2 GiB RAM — the homelab k8s workers, not the camera Pi).

**3. Detect + crop:**

```bash
.venv/bin/python -m coffeecam.detect dataset/images/kahvi.png
```

Prints the detected bbox/confidence and saves the cropped pot region to `crops/`.
Weights come from `models/CHECKPOINT` (a pointer to a `runs/` directory — the
`.pt` files aren't committed), falling back to the newest run.

## Pipeline dashboard

`coffeecam/server.py` runs `acquire → normalize → detect → crop → classify` on a
timer and serves each stage over HTTP — the detector/crop are visible now and a
real fullness classifier drops in later. The classify stage is currently a
brightness-heuristic placeholder.

```bash
.venv/bin/python -m coffeecam.server        # http://<host>:8000/  (+ /pipeline.json, /crop.jpg, …)
```

The server also logs every tick's fullness verdict to
`captures/pipeline/state-<day>.jsonl` and serves it back: **`/history`** (one
day, segmented bar + transition frames), **`/history/long?days=N`** (an N-day
graph). `python -m coffeecam.backfill_history <day>` fills the log from stored
frames; `python -m coffeecam.fullness_confidence` reports how confident the
classifier is and lists the worst frames.

Details, routes, env vars, and current model limitations: [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Tests

```bash
.venv/bin/pytest tests/
```

No network/GPU required — bbox math, label I/O, shift augmentation, capture
transform/dedup, and the pipeline + server (with a fake model). Training and live
detection are exercised manually (see Usage) since they need `ultralytics` plus weights.
