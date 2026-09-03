"""Build ``fullness_dataset/{train,val,test}/{class}/*.jpg`` from the label stores.

Reads ``captures/fullness.jsonl`` (fill level per frame) + ``captures/annotations.jsonl``
(the GT ``coffee_pot`` box) and writes :func:`coffeecam.fullness_crop.prepare_crop`
outputs into an ImageFolder tree ready for ``yolo classify``.

The split is the **same** deterministic hash-of-``rel`` as
``coffeecam.dataset.promote`` (`_split_bucket`, seed 0, 15/15), so a frame lands
in the same split for the fullness task as for the detector — a brew event's
near-duplicate heartbeat frames can't straddle train/test across the two tasks.

Every split keeps its **natural class prevalence**. Oversampling the minority
classes to ~1:1:1 is `fullness_train`'s job, and only on the *train* tree.

Idempotent: the output tree is wiped and rebuilt each run (it's a derived
artifact), so changing ``--merge`` doesn't leave stale class dirs behind.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from coffeecam import annotations, fullness_labels
from coffeecam.dataset import _split_bucket, dest_name_for
from coffeecam.fullness_crop import CROP_SIZE, prepare_crop

DEFAULT_OUT = Path("fullness_dataset")
SPLITS = ("train", "val", "test")

# Ways to collapse the 6 raw labels (LEVELS + "absent") into a coarser class set.
# `full` is chronically sparse (a carafe is only briefly full), so `none` will
# usually under-serve it — pick `coarse` or `binary` if the confusion matrix
# shows `full`/`high` are unlearnable.
MERGES: dict[str, dict[str, str]] = {
    "none": {lvl: lvl for lvl in fullness_labels.LABELS},
    "coarse": {
        "empty": "empty", "low": "some", "half": "some",
        "high": "lots", "full": "lots", "absent": "absent",
    },
    "binary": {
        "empty": "empty", "low": "has_coffee", "half": "has_coffee",
        "high": "has_coffee", "full": "has_coffee", "absent": "absent",
    },
}


@dataclass
class BuildSummary:
    counts: dict[str, dict[str, int]]  # split -> class -> n
    merge: str
    skipped_watched: int = 0
    skipped_no_box: int = 0
    skipped_missing: int = 0
    classes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(n for split in self.counts.values() for n in split.values())

    def __str__(self) -> str:
        rows = [f"merge={self.merge}  classes={self.classes}  total={self.total}"]
        for split in SPLITS:
            per = self.counts.get(split, {})
            body = "  ".join(f"{c} {per.get(c, 0)}" for c in self.classes)
            rows.append(f"  {split:5} {sum(per.values()):4}   {body}")
        rows.append(
            f"  skipped: watched={self.skipped_watched} "
            f"no_box={self.skipped_no_box} missing={self.skipped_missing}"
        )
        return "\n".join(rows)


def remap(level: str, merge: str) -> str:
    try:
        return MERGES[merge][level]
    except KeyError as exc:
        raise ValueError(f"unknown level {level!r} or merge {merge!r}") from exc


def _safe_to_wipe(out_dir: Path) -> bool:
    """True if ``out_dir`` is absent, empty, or clearly a previous build (only
    ``train``/``val``/``test`` children)."""
    if not out_dir.exists():
        return True
    children = {p.name for p in out_dir.iterdir()}
    return children <= set(SPLITS)


def build(
    *,
    fullness_store: Path | None = None,
    annot_store: Path | None = None,
    captures_dir: Path | None = None,
    out_dir: Path = DEFAULT_OUT,
    merge: str = "none",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    crop_size: int = CROP_SIZE,
    dry_run: bool = False,
) -> BuildSummary:
    if merge not in MERGES:
        raise ValueError(f"merge must be one of {sorted(MERGES)}, got {merge!r}")
    captures_dir = Path(captures_dir or "captures")
    fullness_store = Path(fullness_store or captures_dir / "fullness.jsonl")
    annot_store = Path(annot_store or captures_dir / "annotations.jsonl")
    out_dir = Path(out_dir)

    classes = sorted(set(MERGES[merge].values()))
    labels = fullness_labels.load(fullness_store)
    anns = annotations.load(annot_store)

    if not dry_run:
        if not _safe_to_wipe(out_dir):
            raise RuntimeError(
                f"{out_dir} has unexpected contents; refusing to wipe. "
                "Move it aside or point --out elsewhere."
            )
        if out_dir.exists():
            shutil.rmtree(out_dir)
        for split in SPLITS:
            for cls in classes:
                (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    summary = BuildSummary(
        counts={s: {c: 0 for c in classes} for s in SPLITS},
        merge=merge,
        classes=classes,
    )

    for rel in sorted(labels):
        lab = labels[rel]
        if lab.skip or not lab.level:
            summary.skipped_watched += 1
            continue
        ann = anns.get(rel)
        if ann is None or not ann.boxes or ann.skip:
            summary.skipped_no_box += 1
            continue
        src = captures_dir / rel
        if not src.exists():
            summary.skipped_missing += 1
            continue

        cls = remap(lab.level, merge)
        bucket = _split_bucket(rel, seed=seed, val_frac=val_frac, test_frac=test_frac)
        summary.counts[bucket][cls] += 1
        if dry_run:
            continue

        dest = out_dir / bucket / cls / dest_name_for(rel)
        with Image.open(src) as im:
            frame = im.convert("RGB")
        prepare_crop(frame, tuple(ann.boxes[0]), size=crop_size).save(
            dest, "JPEG", quality=92
        )

    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--captures-dir", type=Path, default=Path("captures"))
    ap.add_argument("--merge", choices=sorted(MERGES), default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    args = ap.parse_args(argv)

    summary = build(
        captures_dir=args.captures_dir,
        out_dir=args.out,
        merge=args.merge,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(summary)


if __name__ == "__main__":
    main()
