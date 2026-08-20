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
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

    from ultralytics import YOLO  # lazy import: keeps --help usable without the dep installed

    model = YOLO(args.weights)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best weights: {best}")


if __name__ == "__main__":
    main()
