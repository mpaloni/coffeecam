"""YOLO-format label I/O and pixel/normalized bbox conversion helpers.

Also hosts :func:`promote` — the step that turns the ``captures/annotations.jsonl``
sidecar (written by the ``/annotate`` endpoint) into a trainable ``dataset/``:
copied images, YOLO labels, and regenerated ``{train,val,test}.txt`` split
manifests. It is deterministic and idempotent; re-run it as labels accumulate.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


def clamp_bbox(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(x1, img_w))
    y1 = max(0.0, min(y1, img_h))
    x2 = max(0.0, min(x2, img_w))
    y2 = max(0.0, min(y2, img_h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def bbox_pixel_to_yolo(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Pixel-space (x1, y1, x2, y2) -> normalized YOLO (cx, cy, w, h)."""
    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, img_w, img_h)
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def bbox_yolo_to_pixel(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Normalized YOLO (cx, cy, w, h) -> pixel-space (x1, y1, x2, y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, img_w, img_h)
    return round(x1), round(y1), round(x2), round(y2)


def write_label(label_path: Path, class_id: int, cx: float, cy: float, w: float, h: float) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def read_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        class_id, cx, cy, w, h = line.split()
        boxes.append((int(class_id), float(cx), float(cy), float(w), float(h)))
    return boxes


# --- promote: annotations.jsonl -> trainable dataset/ ------------------------

DEFAULT_DATASET_DIR = Path("dataset")
# Synthetic training frames (kahvi.png + its shift-augmentation copies) always
# stay in train.txt; only real captured frames are eligible for val/test.
_SYNTHETIC_PREFIX = "kahvi"


@dataclass(frozen=True)
class PromoteSummary:
    train: int
    val: int
    test: int
    negatives: int
    skipped_missing: int

    def __str__(self) -> str:
        return (
            f"{self.train} train / {self.val} val / {self.test} test, "
            f"{self.negatives} negatives"
            + (f", {self.skipped_missing} skipped (image missing)" if self.skipped_missing else "")
        )


def dest_name_for(rel: str) -> str:
    """``2026-08-31/161913_556.jpg`` -> ``cap_20260831_161913_556.jpg``.

    Stable and collision-free, so re-running ``promote`` overwrites rather than
    duplicating.
    """
    p = PurePosixPath(rel)
    day = p.parent.name.replace("-", "")
    return f"cap_{day}_{p.stem}.jpg" if day else f"cap_{p.stem}.jpg"


def _split_bucket(rel: str, *, seed: int, val_frac: float, test_frac: float) -> str:
    digest = hashlib.sha1(f"{seed}:{rel}".encode()).hexdigest()
    frac = int(digest[:15], 16) / float(1 << 60)  # deterministic in [0, 1)
    if frac < test_frac:
        return "test"
    if frac < test_frac + val_frac:
        return "val"
    return "train"


def _existing_synthetic(images_dir: Path) -> list[str]:
    if not images_dir.is_dir():
        return []
    return sorted(
        f"./images/{p.name}"
        for p in images_dir.iterdir()
        if p.is_file() and p.name.startswith(_SYNTHETIC_PREFIX)
    )


def _ensure_test_in_data_yaml(dataset_dir: Path) -> None:
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return
    text = data_yaml.read_text()
    if "\ntest:" in text or text.startswith("test:"):
        return
    lines = text.splitlines()
    out = []
    for line in lines:
        out.append(line)
        if line.startswith("val:"):
            out.append("test: test.txt")
    data_yaml.write_text("\n".join(out) + "\n")


def promote(
    store: Path | None = None,
    captures_dir: Path | None = None,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    negatives: bool = True,
    dry_run: bool = False,
) -> PromoteSummary:
    """Build ``dataset/{images,labels}`` + split manifests from the label store.

    - Rows whose image is missing on disk are skipped.
    - ``boxes == []`` rows are negatives: image copied, empty label file. Set
      ``negatives=False`` to drop them.
    - Split is a deterministic hash of ``rel``; synthetic ``kahvi*`` frames stay
      in train, real frames are the only val/test candidates.
    - Idempotent: re-running with more labels regenerates cleanly.
    """
    from coffeecam import annotations
    from coffeecam.annotate import write_example

    if store is None:
        store = annotations.DEFAULT_STORE
    if captures_dir is None:
        captures_dir = Path("captures")
    captures_dir = Path(captures_dir)
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    rows = annotations.load(store)

    manifests: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    n_negatives = skipped = 0

    for rel in sorted(rows):
        ann = rows[rel]
        src = captures_dir / rel
        if not src.exists():
            skipped += 1
            continue
        is_negative = not ann.boxes
        if is_negative and not negatives:
            continue

        name = dest_name_for(rel)
        bucket = _split_bucket(rel, seed=seed, val_frac=val_frac, test_frac=test_frac)
        manifests[bucket].append(f"./images/{name}")
        if is_negative:
            n_negatives += 1
        if not dry_run:
            write_example(src, list(ann.boxes), images_dir=images_dir, labels_dir=labels_dir,
                          dest_name=name)

    train = sorted(set(_existing_synthetic(images_dir)) | set(manifests["train"]))
    val = sorted(manifests["val"])
    test = sorted(manifests["test"])

    if not dry_run:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "train.txt").write_text("\n".join(train) + ("\n" if train else ""))
        (dataset_dir / "val.txt").write_text("\n".join(val) + ("\n" if val else ""))
        (dataset_dir / "test.txt").write_text("\n".join(test) + ("\n" if test else ""))
        _ensure_test_in_data_yaml(dataset_dir)

    return PromoteSummary(
        train=len(train), val=len(val), test=len(test),
        negatives=n_negatives, skipped_missing=skipped,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="coffeecam dataset tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("promote", help="annotations.jsonl -> dataset/ + split manifests")
    p.add_argument("--store", type=Path, default=None)
    p.add_argument("--captures-dir", type=Path, default=None)
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-negatives", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = promote(
        store=args.store,
        captures_dir=args.captures_dir,
        dataset_dir=args.dataset_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        negatives=not args.no_negatives,
        dry_run=args.dry_run,
    )
    print(("[dry-run] " if args.dry_run else "") + str(summary))


if __name__ == "__main__":
    main()
