"""Translate (shift) a labeled image in x/y to synthesize more training data.

Takes an already-labeled image plus a pixel shift (dx, dy), moves the image
content by that amount, updates the bounding box to match, and writes three
artifacts:

  * ``dataset/images/<stem>_shift_x{dx}_y{dy}.png``    -- the shifted raw image
  * ``dataset/labels/<stem>_shift_x{dx}_y{dy}.txt``    -- the updated YOLO label
  * ``dataset/previews/<stem>_shift_x{dx}_y{dy}_preview.png`` -- shifted image
    with the updated box drawn, so it can be eyeballed

Every shift is recorded in ``dataset/augmentations.json`` so the set of
augmentations for the dataset is remembered and can be regenerated from scratch
with ``--replay``.

Convention: ``dx > 0`` moves content right, ``dy > 0`` moves content down. The
border area exposed by the shift is filled per ``--fill`` (edge/reflect/wrap/
black).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from coffeecam.dataset import (
    bbox_pixel_to_yolo,
    bbox_yolo_to_pixel,
    clamp_bbox,
    read_label,
    write_label,
)

FILL_MODES = ("edge", "reflect", "wrap", "black")
DEFAULT_MANIFEST = Path("dataset/augmentations.json")


def shift_bbox(
    x1: float, y1: float, x2: float, y2: float, dx: int, dy: int, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Translate a pixel-space bbox by (dx, dy) and clamp it to the frame."""
    return clamp_bbox(x1 + dx, y1 + dy, x2 + dx, y2 + dy, img_w, img_h)


def shift_image(image: Image.Image, dx: int, dy: int, fill: str = "edge") -> Image.Image:
    """Return a copy of ``image`` with its content shifted by (dx, dy) pixels."""
    if fill not in FILL_MODES:
        raise ValueError(f"fill must be one of {FILL_MODES}, got {fill!r}")

    arr = np.asarray(image)
    h, w = arr.shape[:2]

    if fill == "wrap":
        axes = (0, 1) if arr.ndim >= 2 else (0,)
        out = np.roll(arr, shift=(dy, dx), axis=axes)
        return Image.fromarray(out, mode=image.mode)

    top, bottom = max(dy, 0), max(-dy, 0)
    left, right = max(dx, 0), max(-dx, 0)
    pad_width = [(top, bottom), (left, right)] + [(0, 0)] * (arr.ndim - 2)
    if fill == "black":
        padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
    else:
        padded = np.pad(arr, pad_width, mode=fill)  # "edge" | "reflect"

    # The visible h*w window sits at (bottom, right) inside the padded array.
    out = padded[bottom : bottom + h, right : right + w]
    return Image.fromarray(np.ascontiguousarray(out), mode=image.mode)


def _out_stem(src_stem: str, dx: int, dy: int) -> str:
    return f"{src_stem}_shift_x{dx}_y{dy}"


def augment_and_save(
    image_path: Path,
    dx: int,
    dy: int,
    fill: str = "edge",
    label_path: Path | None = None,
    images_dir: Path = Path("dataset/images"),
    labels_dir: Path = Path("dataset/labels"),
    previews_dir: Path = Path("dataset/previews"),
    manifest_path: Path | None = DEFAULT_MANIFEST,
) -> dict:
    """Shift one labeled image, write image/label/preview, and record the shift.

    Returns the manifest record for the generated sample.
    """
    image_path = Path(image_path)
    label_path = Path(label_path) if label_path else labels_dir / f"{image_path.stem}.txt"

    image = Image.open(image_path)
    img_w, img_h = image.size

    boxes = read_label(label_path)
    if not boxes:
        raise ValueError(f"No boxes in label {label_path}")
    if len(boxes) > 1:
        raise ValueError(f"{label_path} has {len(boxes)} boxes; shift augmentation expects exactly 1")
    class_id, cx, cy, w, h = boxes[0]
    src_px = bbox_yolo_to_pixel(cx, cy, w, h, img_w, img_h)

    new_px = shift_bbox(*src_px, dx, dy, img_w, img_h)
    if new_px[2] - new_px[0] < 1 or new_px[3] - new_px[1] < 1:
        raise ValueError(
            f"Shift (dx={dx}, dy={dy}) pushes the box out of frame "
            f"(src px {tuple(map(round, src_px))} -> {tuple(map(round, new_px))})"
        )
    new_cx, new_cy, new_w, new_h = bbox_pixel_to_yolo(*new_px, img_w, img_h)

    stem = _out_stem(image_path.stem, dx, dy)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    shifted = shift_image(image, dx, dy, fill=fill)
    image_out = images_dir / f"{stem}.png"
    shifted.save(image_out)

    label_out = labels_dir / f"{stem}.txt"
    write_label(label_out, class_id, new_cx, new_cy, new_w, new_h)

    preview = shifted.convert("RGB")
    ImageDraw.Draw(preview).rectangle(new_px, outline="red", width=4)
    preview_out = previews_dir / f"{stem}_preview.png"
    preview.save(preview_out)

    record = {
        "source": str(image_path),
        "source_label": str(label_path),
        "dx": dx,
        "dy": dy,
        "fill": fill,
        "img_w": img_w,
        "img_h": img_h,
        "class_id": class_id,
        "src_bbox_px": [round(v, 3) for v in src_px],
        "new_bbox_px": [round(v, 3) for v in new_px],
        "src_bbox_yolo": [round(v, 6) for v in (cx, cy, w, h)],
        "new_bbox_yolo": [round(v, 6) for v in (new_cx, new_cy, new_w, new_h)],
        "image": str(image_out),
        "label": str(label_out),
        "preview": str(preview_out),
    }

    if manifest_path is not None:
        _record_shift(Path(manifest_path), record)

    return record


def _record_shift(manifest_path: Path, record: dict) -> None:
    """Upsert a shift record into the manifest, keyed by output image path."""
    entries: list[dict] = []
    if manifest_path.exists():
        entries = json.loads(manifest_path.read_text())
    entries = [e for e in entries if e.get("image") != record["image"]]
    entries.append(record)
    entries.sort(key=lambda e: e["image"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, indent=2) + "\n")


def replay(manifest_path: Path = DEFAULT_MANIFEST) -> list[dict]:
    """Regenerate every image/label/preview listed in the manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    entries = json.loads(manifest_path.read_text())
    out = []
    for e in entries:
        out.append(
            augment_and_save(
                Path(e["source"]),
                e["dx"],
                e["dy"],
                fill=e.get("fill", "edge"),
                label_path=Path(e["source_label"]) if e.get("source_label") else None,
                images_dir=Path(e["image"]).parent,
                labels_dir=Path(e["label"]).parent,
                previews_dir=Path(e["preview"]).parent,
                manifest_path=manifest_path,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, nargs="?", help="e.g. dataset/images/kahvi.png")
    parser.add_argument("--dx", type=int, help="pixels to shift right (negative = left)")
    parser.add_argument("--dy", type=int, help="pixels to shift down (negative = up)")
    parser.add_argument("--fill", choices=FILL_MODES, default="edge", help="how to fill exposed border (default: edge)")
    parser.add_argument("--label", type=Path, default=None, help="source label (default: dataset/labels/<stem>.txt)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replay", action="store_true", help="regenerate all artifacts from the manifest and exit")
    args = parser.parse_args()

    if args.replay:
        records = replay(args.manifest)
        print(f"Replayed {len(records)} augmentation(s) from {args.manifest}")
        return

    if args.image is None or args.dx is None or args.dy is None:
        parser.error("image, --dx and --dy are required unless --replay is given")

    record = augment_and_save(
        args.image,
        args.dx,
        args.dy,
        fill=args.fill,
        label_path=args.label,
        manifest_path=args.manifest,
    )
    print(f"Shift (dx={args.dx}, dy={args.dy}, fill={args.fill})")
    print(f"  bbox px : {tuple(round(v) for v in record['src_bbox_px'])} -> {tuple(round(v) for v in record['new_bbox_px'])}")
    print(f"  image   : {record['image']}")
    print(f"  label   : {record['label']}")
    print(f"  preview : {record['preview']}")
    print(f"  manifest: {args.manifest}")


if __name__ == "__main__":
    main()
