"""YOLO-format label I/O and pixel/normalized bbox conversion helpers."""

from __future__ import annotations

from pathlib import Path


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
