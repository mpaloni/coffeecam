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
import hashlib
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from coffeecam import annotations, fullness_labels
from coffeecam.dataset import _split_bucket, dest_name_for
from coffeecam.fullness_crop import CROP_SIZE, prepare_crop

DEFAULT_OUT = Path("fullness_dataset")
SPLITS = ("train", "val", "test")

# --balance cap: at most this many crops per source train frame (1 plain + the
# rest box-jittered). Keeps a rare class from being blown up 20x off a handful
# of frames.
MAX_VARIANTS = 8


def _rng_for(rel: str, seed: int) -> random.Random:
    h = hashlib.sha1(f"{seed}:{rel}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def jitter_box(
    box: tuple[float, float, float, float],
    rng: random.Random,
    *,
    scale: tuple[float, float] = (0.85, 1.20),
    translate: int = 12,
) -> tuple[float, float, float, float]:
    """Perturb a GT box to mimic the live detector's sloppiness: scale about the
    centre by a random factor, then shift each side independently by ±translate
    px. `prepare_crop` re-letterboxes, so the result stays a valid crop."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = rng.uniform(*scale)
    hw, hh = (x2 - x1) / 2 * s, (y2 - y1) / 2 * s
    t = lambda: rng.uniform(-translate, translate)  # noqa: E731
    return (cx - hw + t(), cy - hh + t(), cx + hw + t(), cy + hh + t())

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
    balance: bool = False,
    dry_run: bool = False,
) -> BuildSummary:
    """Build the tree. With ``balance`` the *train* split is oversampled toward
    ~1:1:1 by emitting up to :data:`MAX_VARIANTS` box-jittered crops per source
    frame for the rarer classes (val/test stay at one plain crop each)."""
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

    # Pass 1: resolve every usable frame to (rel, box, cls, bucket).
    items: list[tuple[str, tuple, str, str]] = []
    for rel in sorted(labels):
        lab = labels[rel]
        if lab.skip or not lab.level:
            summary.skipped_watched += 1
            continue
        ann = anns.get(rel)
        if ann is None or not ann.boxes or ann.skip:
            summary.skipped_no_box += 1
            continue
        if not (captures_dir / rel).exists():
            summary.skipped_missing += 1
            continue
        cls = remap(lab.level, merge)
        bucket = _split_bucket(rel, seed=seed, val_frac=val_frac, test_frac=test_frac)
        items.append((rel, tuple(ann.boxes[0]), cls, bucket))

    # Per-class variant count for the train split (1 unless balancing).
    train_per_class = {c: 0 for c in classes}
    for _, _, cls, bucket in items:
        if bucket == "train":
            train_per_class[cls] += 1
    target = max(train_per_class.values(), default=0)

    def n_variants(cls: str, bucket: str) -> int:
        if bucket != "train" or not balance or not train_per_class[cls]:
            return 1
        return max(1, min(MAX_VARIANTS, round(target / train_per_class[cls])))

    # Pass 2: emit crops.
    for rel, box, cls, bucket in items:
        k = n_variants(cls, bucket)
        summary.counts[bucket][cls] += k
        if dry_run:
            continue
        with Image.open(captures_dir / rel) as im:
            frame = im.convert("RGB")
        stem = dest_name_for(rel)[:-4]  # drop ".jpg"
        rng = _rng_for(rel, seed)
        for v in range(k):
            crop_box = box if v == 0 else jitter_box(box, rng)
            name = f"{stem}.jpg" if v == 0 else f"{stem}_j{v}.jpg"
            prepare_crop(frame, tuple(crop_box), size=crop_size).save(
                out_dir / bucket / cls / name, "JPEG", quality=92
            )

    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--captures-dir", type=Path, default=Path("captures"))
    ap.add_argument("--merge", choices=sorted(MERGES), default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--balance", action="store_true",
        help="oversample the train split toward 1:1:1 with box-jittered crops",
    )
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    args = ap.parse_args(argv)

    summary = build(
        captures_dir=args.captures_dir,
        out_dir=args.out,
        balance=args.balance,
        merge=args.merge,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(summary)


if __name__ == "__main__":
    main()
