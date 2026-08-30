"""Run the trained detector on an image: `python -m coffeecam.detect <image>`.

The reusable guts (`load_model`, `detect_pot`) are imported by `coffeecam.pipeline`;
the CLI below just wraps them with file I/O.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# The detector trained at imgsz 320 (runs/detect/runs/train-nomosaic-2); predict
# with the same size or confidence drops.
DEFAULT_IMGSZ = 320
DEFAULT_CONF = 0.25
CHECKPOINT_FILE = Path("models/CHECKPOINT")


def find_latest_weights(runs_dir: Path = Path("runs")) -> Path:
    # ultralytics nests output depth varies by version/args (e.g.
    # runs/detect/runs/train/weights/best.pt), so search recursively.
    candidates = sorted(runs_dir.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No trained weights found under {runs_dir}/**/weights/best.pt — run train.py first."
        )
    return candidates[-1]


def resolve_weights(explicit: Path | None = None) -> Path:
    """Pick weights: explicit arg > models/CHECKPOINT pointer > newest by mtime.

    Weights are not committed (they're ~24 MB and live in git-ignored runs/), so
    `models/CHECKPOINT` is a one-line text file naming the chosen run directory,
    e.g. `runs/detect/runs/train-nomosaic-2`. `weights/best.pt` is appended.
    """
    if explicit is not None:
        return explicit
    if CHECKPOINT_FILE.exists():
        run = CHECKPOINT_FILE.read_text().strip()
        if run:
            weights = Path(run)
            if weights.suffix != ".pt":
                weights = weights / "weights" / "best.pt"
            if not weights.exists():
                raise FileNotFoundError(
                    f"{CHECKPOINT_FILE} points at {run!r} but {weights} does not exist"
                )
            return weights
    return find_latest_weights()


def promote_weights(run: Path) -> Path:
    """Write models/CHECKPOINT so `resolve_weights()` locks onto `run` from now on."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(str(run) + "\n")
    return resolve_weights()


def load_model(weights: Path | None = None):
    """Load a YOLO model once; callers should hold the result, not reload per frame."""
    from ultralytics import YOLO  # lazy: keeps --help / imports usable without the dep

    return YOLO(str(resolve_weights(weights)))


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


def detect_pot(image, model, *, conf: float = DEFAULT_CONF, imgsz: int = DEFAULT_IMGSZ) -> Detection | None:
    """Highest-confidence coffee_pot box in `image` (a PIL image, ndarray or path), or None."""
    result = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)[0]
    if len(result.boxes) == 0:
        return None
    best = max(result.boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = (int(v) for v in best.xyxy[0].tolist())
    return Detection(x1, y1, x2, y2, float(best.conf[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and crop the coffee pot in an image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, default=None, help="Defaults to models/CHECKPOINT, else newest run")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--out", type=Path, default=Path("crops"))
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    model = load_model(args.weights)
    det = detect_pot(str(args.image), model, conf=args.conf, imgsz=args.imgsz)

    if det is None:
        print("No coffee_pot detected.")
        return

    print(f"coffee_pot detected: bbox={det.bbox} conf={det.confidence:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image)

    crop_path = args.out / f"{args.image.stem}_crop.png"
    image.crop(det.bbox).save(crop_path)
    print(f"Saved crop: {crop_path}")

    bounded = image.convert("RGB").copy()
    ImageDraw.Draw(bounded).rectangle(det.bbox, outline="red", width=4)
    bounded_path = args.out / f"{args.image.stem}_bounded.png"
    bounded.save(bounded_path)
    print(f"Saved bounded image: {bounded_path}")


if __name__ == "__main__":
    main()
