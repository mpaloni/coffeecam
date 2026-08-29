# coffeecam

YOLOv8-based detector that finds the coffee pot in a camera frame and crops it out — step 1 toward a fullness-monitoring camera for the break-room coffee maker (`kahvi` = Finnish for coffee).

Detection only, for now. A second model/heuristic to classify how full the pot is comes later, once this pipeline is validated and more sample frames (across fill levels) are collected. Pulling frames live from the camera is also a follow-up — today's dataset seed is a single screenshot (`dataset/images/kahvi.png`).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`ultralytics` downloads COCO-pretrained YOLOv8n weights on first use — needs internet.

## Usage

**1. Label a bounding box for an image** (no GUI labeling tool set up — pixel coords in, YOLO label + preview PNG out):

```bash
.venv/bin/python -m coffeecam.annotate dataset/images/kahvi.png <x1> <y1> <x2> <y2>
```

Check `dataset/previews/<name>_preview.png` to confirm the box looks right.

**1b. Grow the dataset by shifting a labeled image** (cheap translation augmentation — `dx > 0` moves content right, `dy > 0` down):

```bash
.venv/bin/python -m coffeecam.augment_shift dataset/images/kahvi.png --dx 40 --dy 25 [--fill edge|reflect|wrap|black]
```

Per shift it writes the shifted raw image to `dataset/images/<stem>_shift_x<dx>_y<dy>.png`, the updated YOLO label to `dataset/labels/…`, a boxed preview to `dataset/previews/…`, and appends the shift (dx/dy/fill + old & new bbox) to `dataset/augmentations.json`. That manifest is the record of every augmentation; `--replay` regenerates all of them from it. Shifted images/labels live alongside the originals so `train` picks them up automatically.

Stitch every shift preview into one animated GIF (box sweeps around the frame, each frame stamped with its dx/dy):

```bash
.venv/bin/python -m coffeecam.preview_anim   # -> dataset/shift_previews.gif
```

**2. Train:**

```bash
.venv/bin/python -m coffeecam.train --epochs 20
```

Writes to `runs/train/weights/best.pt`. With only one labeled image the model will overfit badly — this validates the pipeline, not detection quality. Add more labeled frames (varied lighting, angles, fill levels) before expecting real accuracy.

**3. Detect + crop:**

```bash
.venv/bin/python -m coffeecam.detect dataset/images/kahvi.png
```

Prints the detected bbox/confidence and saves the cropped pot region to `crops/`.

## Tests

```bash
.venv/bin/pytest tests/
```

Covers bbox math, label I/O, and shift augmentation only — no network/GPU required. Training and detection are exercised manually (see Usage above) since they need `ultralytics` plus downloaded weights.
