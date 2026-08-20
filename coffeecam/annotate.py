"""Manual bounding-box annotation.

No labeling GUI (CVAT/labelImg) is set up for this project, so this takes a
pixel-space bbox on the command line, writes the YOLO-format label, and saves
a preview image with the box drawn so it can be checked visually.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from coffeecam.dataset import bbox_pixel_to_yolo, write_label

CLASS_NAMES = ["coffee_pot"]


def annotate_image(
    image_path: Path,
    bbox: tuple[float, float, float, float],
    class_id: int = 0,
    labels_dir: Path = Path("dataset/labels"),
    previews_dir: Path = Path("dataset/previews"),
) -> tuple[Path, Path]:
    image = Image.open(image_path)
    img_w, img_h = image.size

    cx, cy, w, h = bbox_pixel_to_yolo(*bbox, img_w, img_h)
    label_path = labels_dir / f"{image_path.stem}.txt"
    write_label(label_path, class_id, cx, cy, w, h)

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
    parser.add_argument("--labels-dir", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--previews-dir", type=Path, default=Path("dataset/previews"))
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
