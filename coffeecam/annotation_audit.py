"""Audit ``captures/annotations.jsonl`` for labels attached to the wrong frame.

The ``/annotate`` browser labeler had a stale positional-index bug (see
``docs/annotation-audit-handoff.md``): the image on the canvas could be several
frames ahead of the ``rel`` the drawn box was saved against. This tool estimates
the blast radius.

    .venv/bin/python -m coffeecam.annotation_audit --weights models/best-trackB-sess.pt

Layer 1 (always): for every box-positive row, run the detector on that row's
frame and compare the saved box to the detector's best box by IoU. Rows split
into OK / LIKELY_MISMATCH / INCONCLUSIVE and land in ``annotation_audit.csv``,
worst first. Explicit-negative rows where the detector is confident a pot *is*
present are flagged NEGATIVE_HAS_POT.

Layer 2 (``--contact``): montage JPGs for the flagged rows — frame + saved box
(red) + detector box (green) + ``rel`` caption — for a human to page through.

Layer 3 (``--fullness``): crop every ``fullness.jsonl`` frame by its GT box and
montage the crops grouped by assigned level, to eyeball crops that don't match
their bin.

The triage is only as good as the checkpoint. ``trackB-v1`` (~0.16 session
mAP50-95) yields many "no detection" rows; it catches gross misses (pot absent,
box in empty space, hand in frame), not a box that is 40 px off. Prefer
``models/best-trackB-sess.pt`` (0.487) or a newer run via ``--weights``.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from coffeecam import annotations
from coffeecam.detect import DEFAULT_CONF
from coffeecam.summary import _FONT, detect_on_frame

DEFAULT_STORE = Path("captures/annotations.jsonl")
DEFAULT_CAPTURES_DIR = Path("captures")
DEFAULT_FULLNESS_STORE = Path("captures/fullness.jsonl")
DEFAULT_CSV = Path("annotation_audit.csv")
DEFAULT_CONTACT_DIR = Path("audit_contact")

# IoU thresholds for the Layer-1 buckets.
IOU_OK = 0.5
IOU_MISMATCH = 0.2
# A detection this confident on a frame whose saved box it doesn't match is
# taken as real evidence the box belongs elsewhere (rather than a model miss).
MISMATCH_MIN_CONF = 0.25
# Confidence at which a detection on an *explicit-negative* frame is suspicious.
NEGATIVE_POT_CONF = 0.5

Box = tuple[float, float, float, float]

_RED = (255, 48, 48)
_GREEN = (40, 210, 90)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def classify(best_iou: float, det_conf: float | None) -> str:
    """Bucket one box-positive row from its best IoU and the detector confidence."""
    if det_conf is None:
        return "INCONCLUSIVE"  # model found nothing — can't judge
    if best_iou >= IOU_OK:
        return "OK"
    if best_iou < IOU_MISMATCH and det_conf >= MISMATCH_MIN_CONF:
        return "LIKELY_MISMATCH"
    return "INCONCLUSIVE"


@dataclass(frozen=True)
class RowResult:
    rel: str
    kind: str  # "box" | "negative"
    saved_boxes: list[Box]
    det_box: Box | None
    det_conf: float | None
    best_iou: float
    bucket: str
    note: str = ""

    def csv_row(self) -> dict:
        return {
            "rel": self.rel,
            "bucket": self.bucket,
            "kind": self.kind,
            "best_iou": f"{self.best_iou:.3f}",
            "det_conf": "" if self.det_conf is None else f"{self.det_conf:.3f}",
            "saved_boxes": ";".join(
                ",".join(str(int(round(v))) for v in b) for b in self.saved_boxes
            ),
            "det_box": ""
            if self.det_box is None
            else ",".join(str(int(round(v))) for v in self.det_box),
            "labeled_at": self.note,
        }


# Worst first: mismatches, then inconclusive, then negatives-with-pot, then OK.
_BUCKET_ORDER = {
    "LIKELY_MISMATCH": 0,
    "NEGATIVE_HAS_POT": 1,
    "INCONCLUSIVE": 2,
    "OK": 3,
}


def audit_boxes(
    anns: dict[str, annotations.Annotation],
    captures_dir: Path,
    model,
    *,
    conf: float = DEFAULT_CONF,
    limit: int | None = None,
) -> list[RowResult]:
    """Run Layer-1 triage over every non-skip row in ``anns``.

    ``model`` is a loaded YOLO model (or any object ``detect_on_frame`` accepts).
    Rows whose frame file is missing are skipped silently.
    """
    captures_dir = Path(captures_dir)
    results: list[RowResult] = []
    rels = [r for r in sorted(anns) if not anns[r].skip]
    if limit is not None:
        rels = rels[:limit]

    for rel in rels:
        ann = anns[rel]
        path = captures_dir / rel
        if not path.exists():
            continue
        with Image.open(path) as im:
            frame = im.convert("RGB")
        det, _strong = detect_on_frame(frame, model, conf=conf)
        det_box: Box | None = tuple(float(v) for v in det.bbox) if det else None
        det_conf = float(det.confidence) if det else None

        saved = [tuple(float(v) for v in b) for b in ann.boxes]
        if not saved:  # explicit negative
            bucket = (
                "NEGATIVE_HAS_POT"
                if det_conf is not None and det_conf >= NEGATIVE_POT_CONF
                else "OK"
            )
            results.append(
                RowResult(rel, "negative", [], det_box, det_conf, 0.0, bucket, ann.labeled_at)
            )
            continue

        best_iou = max((iou(b, det_box) for b in saved), default=0.0) if det_box else 0.0
        bucket = classify(best_iou, det_conf)
        results.append(
            RowResult(rel, "box", saved, det_box, det_conf, best_iou, bucket, ann.labeled_at)
        )

    results.sort(key=lambda r: (_BUCKET_ORDER.get(r.bucket, 9), -(r.det_conf or 0.0), r.rel))
    return results


def write_csv(results: list[RowResult], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rel", "bucket", "kind", "best_iou", "det_conf", "saved_boxes", "det_box", "labeled_at"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow(r.csv_row())


def summarize(results: list[RowResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.bucket] = counts.get(r.bucket, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Layer 2 — visual contact sheets
# --------------------------------------------------------------------------- #
def _draw_audit_frame(result: RowResult, captures_dir: Path) -> Image.Image | None:
    path = Path(captures_dir) / result.rel
    if not path.exists():
        return None
    im = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(im)
    for b in result.saved_boxes:
        draw.rectangle([int(v) for v in b], outline=_RED, width=3)
    if result.det_box is not None:
        draw.rectangle([int(v) for v in result.det_box], outline=_GREEN, width=2)
    cap = f"{result.rel}  iou={result.best_iou:.2f}"
    if result.det_conf is not None:
        cap += f" c={result.det_conf:.2f}"
    _, _, tw, th = draw.textbbox((0, 0), cap, font=_FONT)
    draw.rectangle((0, 0, tw + 6, th + 4), fill=(0, 0, 0))
    draw.text((3, 2), cap, fill=(255, 255, 255), font=_FONT)
    return im


def build_contact_sheets(
    results: list[RowResult],
    captures_dir: Path,
    out_dir: Path,
    *,
    buckets: tuple[str, ...] = ("LIKELY_MISMATCH", "INCONCLUSIVE", "NEGATIVE_HAS_POT"),
    per_page: int = 20,
    cols: int = 5,
) -> list[Path]:
    """Montage the flagged rows, ``per_page`` per JPG, red=saved green=detector."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flagged = [r for r in results if r.bucket in buckets]
    pages: list[Path] = []
    for pageno, start in enumerate(range(0, len(flagged), per_page), 1):
        tiles = []
        for r in flagged[start : start + per_page]:
            im = _draw_audit_frame(r, captures_dir)
            if im is not None:
                tiles.append(im)
        if not tiles:
            continue
        tw = max(t.width for t in tiles)
        thh = max(t.height for t in tiles)
        rows = math.ceil(len(tiles) / cols)
        sheet = Image.new("RGB", (cols * tw, rows * thh), (20, 20, 20))
        for idx, t in enumerate(tiles):
            x = (idx % cols) * tw
            y = (idx // cols) * thh
            sheet.paste(t, (x, y))
        page = out_dir / f"page_{pageno:02d}.jpg"
        sheet.save(page, "JPEG", quality=85)
        pages.append(page)
    return pages


# --------------------------------------------------------------------------- #
# Layer 3 — fullness crop montage
# --------------------------------------------------------------------------- #
def build_fullness_montage(
    captures_dir: Path,
    ann_store: Path,
    fullness_store: Path,
    out_dir: Path,
    *,
    per_row: int = 12,
    crop: int = 96,
) -> list[Path]:
    """One montage per fill level: each labelled frame cropped to its GT box."""
    from coffeecam import fullness_labels

    anns = annotations.load(ann_store)
    labels = fullness_labels.load(fullness_store)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_level: dict[str, list[str]] = {}
    for rel, lab in sorted(labels.items()):
        if lab.skip or not lab.level:
            continue
        by_level.setdefault(lab.level, []).append(rel)

    pages: list[Path] = []
    for level, rels in sorted(by_level.items()):
        crops = []
        for rel in rels:
            ann = anns.get(rel)
            path = Path(captures_dir) / rel
            if ann is None or not ann.boxes or not path.exists():
                continue
            with Image.open(path) as im:
                c = im.convert("RGB").crop([int(v) for v in ann.boxes[0]])
            crops.append(c.resize((crop, crop)))
        if not crops:
            continue
        rows = math.ceil(len(crops) / per_row)
        sheet = Image.new("RGB", (per_row * crop, rows * crop), (20, 20, 20))
        for idx, c in enumerate(crops):
            sheet.paste(c, ((idx % per_row) * crop, (idx // per_row) * crop))
        page = out_dir / f"fullness_{level}.jpg"
        sheet.save(page, "JPEG", quality=85)
        pages.append(page)
    return pages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--captures-dir", type=Path, default=DEFAULT_CAPTURES_DIR)
    ap.add_argument("--weights", type=Path, default=None,
                    help="detector weights (default: models/CHECKPOINT). "
                         "Prefer models/best-trackB-sess.pt.")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--limit", type=int, default=None, help="audit only the first N rows")
    ap.add_argument("--contact", action="store_true", help="also write Layer-2 contact sheets")
    ap.add_argument("--contact-dir", type=Path, default=DEFAULT_CONTACT_DIR)
    ap.add_argument("--fullness", action="store_true", help="also write Layer-3 fullness montage")
    ap.add_argument("--fullness-store", type=Path, default=DEFAULT_FULLNESS_STORE)
    args = ap.parse_args(argv)

    from coffeecam.detect import load_model

    model = load_model(args.weights)
    anns = annotations.load(args.store)
    results = audit_boxes(
        anns, args.captures_dir, model, conf=args.conf, limit=args.limit
    )
    write_csv(results, args.csv)

    counts = summarize(results)
    total = len(results)
    print(f"audited {total} rows -> {args.csv}")
    for bucket in ("LIKELY_MISMATCH", "NEGATIVE_HAS_POT", "INCONCLUSIVE", "OK"):
        if bucket in counts:
            pct = 100 * counts[bucket] / total if total else 0
            print(f"  {bucket:16s} {counts[bucket]:4d}  ({pct:.0f}%)")

    if args.contact:
        pages = build_contact_sheets(results, args.captures_dir, args.contact_dir)
        print(f"contact sheets: {len(pages)} page(s) in {args.contact_dir}/")

    if args.fullness:
        pages = build_fullness_montage(
            args.captures_dir, args.store, args.fullness_store, args.contact_dir
        )
        print(f"fullness montage: {len(pages)} level sheet(s) in {args.contact_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
