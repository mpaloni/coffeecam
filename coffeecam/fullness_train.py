"""Train the pot-fullness classifier: ``yolov8n-cls`` on 96 px `prepare_crop`s.

Rebuilds the ImageFolder tree (`coffeecam.fullness_dataset.build`, `--merge
coarse` + `--balance` by default), fine-tunes `yolov8n-cls.pt`, and writes
`models/FULLNESS_CHECKPOINT` (a one-line pointer at the run dir, mirroring
`models/CHECKPOINT` for the detector) so `ModelFullness` can find the weights.

`yolov8n-cls.pt` (~5 MB) auto-downloads from the Ultralytics release on first
run — needs outbound network once.

    python -m coffeecam.fullness_train --epochs 80
    python -m coffeecam.fullness_train --no-build          # reuse fullness_dataset/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coffeecam.fullness_dataset import DEFAULT_OUT, MERGES, build

FULLNESS_CHECKPOINT_FILE = Path("models/FULLNESS_CHECKPOINT")


def promote_run(run_dir: Path) -> None:
    """Point `models/FULLNESS_CHECKPOINT` at ``run_dir``."""
    FULLNESS_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    FULLNESS_CHECKPOINT_FILE.write_text(str(run_dir) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_OUT, help="ImageFolder tree")
    ap.add_argument("--captures-dir", type=Path, default=Path("captures"))
    ap.add_argument("--merge", choices=sorted(MERGES), default="coarse")
    ap.add_argument("--no-build", action="store_true", help="reuse the existing tree")
    ap.add_argument("--no-balance", action="store_true", help="skip train oversampling")
    ap.add_argument("--weights", default="yolov8n-cls.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default=None, help="'cpu', '0'; auto if unset")
    # ultralytics appends the task ("classify/") itself -> runs/classify/<name>,
    # matching the detector's runs/detect/... layout.
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="fullness-v1")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args(argv)

    if not args.no_build:
        summary = build(
            captures_dir=args.captures_dir,
            out_dir=args.data,
            merge=args.merge,
            balance=not args.no_balance,
        )
        print(summary)

    from ultralytics import YOLO  # lazy: keep --help usable without the dep

    model = YOLO(args.weights)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=args.plots,
    )

    run_dir = Path(results.save_dir)
    promote_run(run_dir)
    print(f"best weights : {run_dir / 'weights' / 'best.pt'}")
    print(f"FULLNESS_CHECKPOINT -> {run_dir}")


if __name__ == "__main__":
    main()
