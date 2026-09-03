"""One crop transform, used identically at train and inference time.

The fullness classifier is only as stable as its input. The detector box jitters
(`w` 83±10, `h` 92±8, `x1` 0–307 across real frames); a raw `frame.crop(box)`
therefore feeds the model a differently-framed carafe every frame, and the fill
line lands in a different place each time. `prepare_crop` removes that variance:
expand whatever box we have to a fixed aspect ratio, clamp to the frame,
letterbox-pad to a square (never stretch — a squashed carafe changes the fill
geometry), and resize to `CROP_SIZE`.

`box` is in `frame` pixels: the GT box in training, the detector box at
inference, and `DEFAULT_POT_BOX` when the detector found nothing (the camera is
fixed and the pot lives in the same ~80×90 px window, so a static crop still
classifies rather than falling back to "unknown"). It never returns None.
"""

from __future__ import annotations

from PIL import Image

# Median of all GT boxes in captures/annotations.jsonl. Static fallback crop when
# the detector returns nothing.
DEFAULT_POT_BOX: tuple[int, int, int, int] = (291, 113, 373, 205)

# Square N×N fed to the classifier.
CROP_SIZE = 96

# Target aspect (w / h) the box is expanded to before letterboxing — the median
# GT box shape (83 × 92).
_ASPECT = 83.0 / 92.0


def _expand_to_aspect(
    box: tuple[float, float, float, float], aspect: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def prepare_crop(
    frame: Image.Image,
    box: tuple[int, int, int, int] | None,
    *,
    size: int = CROP_SIZE,
) -> Image.Image:
    """Expand `box` to a fixed aspect, clamp to `frame`, letterbox to square,
    resize to `size`. `box` None => `DEFAULT_POT_BOX`. Never returns None."""
    if box is None:
        box = DEFAULT_POT_BOX

    frame = frame.convert("RGB")
    fw, fh = frame.size

    ex1, ey1, ex2, ey2 = _expand_to_aspect(tuple(map(float, box)), _ASPECT)

    # Clamp to frame bounds. Clamping can dent the aspect ratio at a frame edge;
    # the letterbox step below absorbs that without stretching.
    cx1 = int(round(max(0.0, min(ex1, fw - 1))))
    cy1 = int(round(max(0.0, min(ey1, fh - 1))))
    cx2 = int(round(max(cx1 + 1, min(ex2, fw))))
    cy2 = int(round(max(cy1 + 1, min(ey2, fh))))

    region = frame.crop((cx1, cy1, cx2, cy2))
    rw, rh = region.size

    side = max(rw, rh)
    square = Image.new("RGB", (side, side), (0, 0, 0))
    square.paste(region, ((side - rw) // 2, (side - rh) // 2))

    return square.resize((size, size), Image.BILINEAR)
