"""Make a live frame look like the detector's training data.

The detector trained on `dataset/images/kahvi.png` (512x456, aspect ~1.123) plus
shift augmentations, several black-filled. Empirically what matters to the model
is the **frame aspect ratio and the black border**, not the browser-screenshot
look: padding the live crop out to the training aspect with a thin black border
(no downscaling) lifts live detection confidence by ~40% (≈0.29 -> ≈0.41 on a
test frame). Downscaling the crop into a small letterboxed box — mimicking the
screenshot literally — makes it *worse*, because it shrinks the pot.

This is a bridge until there are enough real frames to retrain on. `map_bbox_back`
converts a detection in padded space back to live-crop pixels so the crop/overlay
shown to the user are of the real frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

TRAIN_ASPECT = 512 / 456  # kahvi.png
TOP_STRIP_H = 20
TOP_STRIP_RGB = (232, 232, 232)
PAD_RGB = (0, 0, 0)


@dataclass(frozen=True)
class PadTransform:
    """How the live crop sits inside the padded canvas — enough to invert a bbox."""

    offset_x: int
    offset_y: int
    src_w: int
    src_h: int


def match_training_frame(crop: Image.Image, *, top_strip: bool = True) -> tuple[Image.Image, PadTransform]:
    """Pad `crop` with a black border out to the training aspect ratio. No resize."""
    src = crop.convert("RGB")
    src_w, src_h = src.size

    if src_w / src_h < TRAIN_ASPECT:
        canvas_w = round(src_h * TRAIN_ASPECT)
        canvas_h = src_h
    else:
        canvas_w = src_w
        canvas_h = round(src_w / TRAIN_ASPECT)

    strip = TOP_STRIP_H if top_strip else 0
    canvas_h += strip

    canvas = Image.new("RGB", (canvas_w, canvas_h), PAD_RGB)
    if strip:
        canvas.paste(Image.new("RGB", (canvas_w, strip), TOP_STRIP_RGB), (0, 0))

    offset_x = (canvas_w - src_w) // 2
    offset_y = strip + (canvas_h - strip - src_h) // 2
    canvas.paste(src, (offset_x, offset_y))

    return canvas, PadTransform(offset_x, offset_y, src_w, src_h)


def map_bbox_back(bbox: tuple[int, int, int, int], t: PadTransform) -> tuple[int, int, int, int]:
    """Padded-space (x1, y1, x2, y2) -> live-crop pixels, clamped to the crop."""
    x1, y1, x2, y2 = bbox
    rx1, rx2 = x1 - t.offset_x, x2 - t.offset_x
    ry1, ry2 = y1 - t.offset_y, y2 - t.offset_y
    if rx2 < rx1:
        rx1, rx2 = rx2, rx1
    if ry2 < ry1:
        ry1, ry2 = ry2, ry1
    return (
        max(0, min(rx1, t.src_w)),
        max(0, min(ry1, t.src_h)),
        max(0, min(rx2, t.src_w)),
        max(0, min(ry2, t.src_h)),
    )
