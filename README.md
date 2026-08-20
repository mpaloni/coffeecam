# coffeecam

YOLOv8-based detector that finds the coffee pot in a camera frame and crops it out — step 1 toward a fullness-monitoring camera for the break-room coffee maker (`kahvi` = Finnish for coffee).

Detection only, for now. A second model/heuristic to classify how full the pot is comes later, once this pipeline is validated and more sample frames (across fill levels) are collected. Pulling frames live from the camera is also a follow-up — today's dataset seed is a single screenshot (`dataset/images/kahvi.png`).

**Camera framing note:** in `kahvi.png` the coffee maker sits off to the right side of the frame rather than centered. When the live-capture follow-up sets up the camera feed, aim/crop it so the pot is roughly centered both horizontally and vertically — better and more consistent framing for both detection and (later) fullness classification.

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

Covers bbox math and label I/O only — no network/GPU required. Training and detection are exercised manually (see Usage above) since they need `ultralytics` plus downloaded weights.
