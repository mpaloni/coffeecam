"""Fine-tune a YOLOv8 model on the coffeecam dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the coffee_pot YOLO detector.")
    parser.add_argument("--data", type=Path, default=Path("dataset/data.yaml"))
    parser.add_argument("--weights", default="yolov8n.pt", help="Base weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15, help="Early-stop after N epochs without val improvement")
    parser.add_argument(
        "--cache",
        default="ram",
        help="Image cache: 'ram', 'disk', or 'none' (the dataset is tiny, so RAM is safe and much faster on a slow CPU)",
    )
    parser.add_argument("--device", default=None, help="e.g. 'cpu', '0'; ultralytics auto-detects if unset")
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augmentation probability. Pass 0 to disable — mosaic stitches 4 images and "
        "tends to destabilize training on a tiny single-object dataset.",
    )
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic for the final N epochs")
    parser.add_argument("--plots", action="store_true", help="Write training plots (needs a working polars; off by default)")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

    from ultralytics import YOLO  # lazy import: keeps --help usable without the dep installed

    model = YOLO(args.weights)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        cache=False if args.cache == "none" else args.cache,
        device=args.device,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        plots=args.plots,
        project=args.project,
        name=args.name,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best weights: {best}")


if __name__ == "__main__":
    main()
