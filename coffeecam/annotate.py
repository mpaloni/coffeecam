"""Manual bounding-box annotation.

No labeling GUI (CVAT/labelImg) is set up for this project, so this takes a
pixel-space bbox on the command line, writes the YOLO-format label, and saves
a preview image with the box drawn so it can be checked visually.

:func:`write_example` is the shared "put one labeled frame into ``dataset/``"
primitive used by both this CLI and ``dataset.promote``: it copies the image
into ``dataset/images/`` and writes a (multi-box) YOLO label into
``dataset/labels/``. An empty box list writes an empty label file — the YOLO
convention for a negative.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from coffeecam.dataset import bbox_pixel_to_yolo

CLASS_NAMES = ["coffee_pot"]

Bbox = tuple[float, float, float, float]

DEFAULT_IMAGES_DIR = Path("dataset/images")
DEFAULT_LABELS_DIR = Path("dataset/labels")
DEFAULT_PREVIEWS_DIR = Path("dataset/previews")


def write_label_lines(
    label_path: Path,
    boxes: list[Bbox],
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> None:
    """Write a YOLO label file: one ``class cx cy w h`` line per box.

    ``boxes`` are pixel-space ``(x1, y1, x2, y2)``. An empty list writes an
    empty file (YOLO negative).
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for bbox in boxes:
        cx, cy, w, h = bbox_pixel_to_yolo(*bbox, img_w, img_h)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_example(
    image_path: Path,
    boxes: list[Bbox],
    *,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    class_id: int = 0,
    dest_name: str | None = None,
) -> tuple[Path, Path]:
    """Copy ``image_path`` into ``images_dir`` and write its YOLO label.

    ``dest_name`` overrides the output filename (stem shared by image and
    label); its suffix, if any, sets the image extension, else the source
    extension is kept. Returns ``(image_dest, label_path)``.
    """
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        img_w, img_h = image.size

    if dest_name:
        stem = Path(dest_name).stem
        suffix = Path(dest_name).suffix or image_path.suffix
    else:
        stem = image_path.stem
        suffix = image_path.suffix

    images_dir.mkdir(parents=True, exist_ok=True)
    image_dest = images_dir / f"{stem}{suffix}"
    shutil.copy2(image_path, image_dest)

    label_path = labels_dir / f"{stem}.txt"
    write_label_lines(label_path, boxes, img_w, img_h, class_id=class_id)
    return image_dest, label_path


def annotate_image(
    image_path: Path,
    bbox: Bbox,
    class_id: int = 0,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    previews_dir: Path = DEFAULT_PREVIEWS_DIR,
) -> tuple[Path, Path]:
    """Label a single box for an image already in ``dataset/images/``.

    Writes the YOLO label plus a preview PNG with the box drawn. Used by the
    CLI; ``dataset.promote`` uses :func:`write_example` instead (no preview).
    """
    image = Image.open(image_path)
    img_w, img_h = image.size

    label_path = labels_dir / f"{image_path.stem}.txt"
    write_label_lines(label_path, [bbox], img_w, img_h, class_id=class_id)

    preview = image.convert("RGB").copy()
    ImageDraw.Draw(preview).rectangle(bbox, outline="red", width=4)
    previews_dir.mkdir(parents=True, exist_ok=True)
    preview_path = previews_dir / f"{image_path.stem}_preview.png"
    preview.save(preview_path)

    return label_path, preview_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Label a coffee_pot bounding box for one image.")
    parser.add_argument("image", type=Path, help="e.g. dataset/images/kahvi.png")
    parser.add_argument("x1", type=float)
    parser.add_argument("y1", type=float)
    parser.add_argument("x2", type=float)
    parser.add_argument("y2", type=float)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--previews-dir", type=Path, default=DEFAULT_PREVIEWS_DIR)
    args = parser.parse_args()

    label_path, preview_path = annotate_image(
        args.image,
        (args.x1, args.y1, args.x2, args.y2),
        class_id=args.class_id,
        labels_dir=args.labels_dir,
        previews_dir=args.previews_dir,
    )
    print(f"Wrote label: {label_path}")
    print(f"Wrote preview: {preview_path}")


if __name__ == "__main__":
    main()
