"""Run the trained detector on an image and save the cropped coffee_pot region."""

from __future__ import annotations

import argparse
from pathlib import Path


def find_latest_weights(runs_dir: Path = Path("runs")) -> Path:
    # ultralytics nests output depth varies by version/args (e.g.
    # runs/detect/runs/train/weights/best.pt), so search recursively.
    candidates = sorted(runs_dir.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No trained weights found under {runs_dir}/**/weights/best.pt — run train.py first."
        )
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and crop the coffee pot in an image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=None, help="Defaults to the most recently trained weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path("crops"))
    args = parser.parse_args()

    from PIL import Image
    from ultralytics import YOLO  # lazy import: keeps --help usable without the dep installed

    weights = args.weights or find_latest_weights()
    model = YOLO(str(weights))
    result = model.predict(source=str(args.image), conf=args.conf, verbose=False)[0]

    if len(result.boxes) == 0:
        print("No coffee_pot detected.")
        return

    best_box = max(result.boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = (int(v) for v in best_box.xyxy[0].tolist())
    confidence = float(best_box.conf[0])
    print(f"coffee_pot detected: bbox=({x1}, {y1}, {x2}, {y2}) conf={confidence:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    crop_path = args.out / f"{args.image.stem}_crop.png"
    Image.open(args.image).crop((x1, y1, x2, y2)).save(crop_path)
    print(f"Saved crop: {crop_path}")


if __name__ == "__main__":
    main()
