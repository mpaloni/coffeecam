"""Synthesize more training data from an already-labeled image.

Three cheap, label-preserving transforms, composable in one pass:

  * **shift**    -- translate content by ``(dx, dy)`` pixels; the bounding box
    moves with it (``dx > 0`` right, ``dy > 0`` down).
  * **rotate**   -- rotate content ``--angle`` degrees counter-clockwise about
    the frame centre; the box is replaced by the axis-aligned box of its four
    rotated corners (so it grows a little on the diagonal).
  * **occlude**  -- paint one or more opaque rectangles over the frame to mimic
    a hand / mug / passer-by partly hiding the pot. The label is left alone --
    the pot is still there, just partly covered -- so occlusions that cover the
    whole box are rejected.

For a shifted/rotated image it writes three artifacts:

  * ``dataset/images/<stem>.png``             -- the transformed raw image
  * ``dataset/labels/<stem>.txt``             -- the updated YOLO label
  * ``dataset/previews/<stem>_preview.png``   -- image with the new box drawn

where ``<stem>`` encodes the transform, e.g.
``kahvi_shift_x40_y25`` or ``kahvi_shift_x0_y0_rot-12_occ2``.

Every generated sample is recorded in ``dataset/augmentations.json`` so the set
of augmentations is remembered and can be rebuilt from scratch with
``--replay``. ``generate`` adds a whole seeded batch of random shift+rotate+
occlude combos in one go (also recorded, so ``--replay`` still reproduces them). ``generate_balanced`` / ``--generate-from
... --balanced`` instead lays down a fixed per-frame recipe (N pure shifts, N
rotations, N occlusions sweeping a coverage range) so the dataset's shift /
rotate / occlude balance is a knob rather than a random draw.

The border area exposed by a shift or rotation is filled per ``--fill``
(edge/reflect/wrap/black).
"""

from __future__ import annotations

import argparse
import json
import math
import random
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
OCCLUDE_FILLS = ("black", "gray", "white", "mean")
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


def _rotate_point(
    x: float, y: float, angle_deg: float, cx: float, cy: float
) -> tuple[float, float]:
    """Rotate (x, y) counter-clockwise by ``angle_deg`` about (cx, cy).

    Image coordinates are y-down, so a visually counter-clockwise rotation is a
    clockwise rotation in maths convention -- hence the sign layout below.
    """
    rad = math.radians(angle_deg)
    cos, sin = math.cos(rad), math.sin(rad)
    ox, oy = x - cx, y - cy
    return cx + ox * cos + oy * sin, cy - ox * sin + oy * cos


def rotate_bbox(
    x1: float, y1: float, x2: float, y2: float, angle_deg: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Axis-aligned bbox of the four corners of (x1,y1,x2,y2) after rotation."""
    cx, cy = img_w / 2, img_h / 2
    corners = [
        _rotate_point(x, y, angle_deg, cx, cy)
        for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return clamp_bbox(min(xs), min(ys), max(xs), max(ys), img_w, img_h)


def rotate_image(image: Image.Image, angle_deg: float, fill: str = "edge") -> Image.Image:
    """Rotate ``image`` counter-clockwise by ``angle_deg``, keeping its size.

    The wedge exposed at each corner is filled per ``fill``: ``black`` uses a
    constant, the others rotate a padded copy (np.pad ``edge``/``reflect``/
    ``wrap``) and centre-crop back, so the wedge samples real-ish content.
    """
    if fill not in FILL_MODES:
        raise ValueError(f"fill must be one of {FILL_MODES}, got {fill!r}")
    if angle_deg % 360 == 0:
        return image.copy()

    if fill == "black":
        return image.rotate(angle_deg, resample=Image.BILINEAR, expand=False, fillcolor=0)

    arr = np.asarray(image)
    h, w = arr.shape[:2]
    pad = max(h, w)
    mode = {"edge": "edge", "reflect": "reflect", "wrap": "wrap"}[fill]
    pad_width = [(pad, pad), (pad, pad)] + [(0, 0)] * (arr.ndim - 2)
    padded = Image.fromarray(np.pad(arr, pad_width, mode=mode), mode=image.mode)
    rotated = padded.rotate(angle_deg, resample=Image.BILINEAR, expand=False)
    out = np.asarray(rotated)[pad : pad + h, pad : pad + w]
    return Image.fromarray(np.ascontiguousarray(out), mode=image.mode)


def _occlusion_fill_value(image: Image.Image, fill: str) -> int | tuple:
    bands = len(image.getbands())
    if fill == "mean":
        arr = np.asarray(image).reshape(-1, bands) if bands > 1 else np.asarray(image).reshape(-1, 1)
        mean = tuple(int(v) for v in arr.mean(axis=0).round())
        return mean if bands > 1 else mean[0]
    level = {"black": 0, "gray": 128, "white": 255}[fill]
    return (level,) * bands if bands > 1 else level


def apply_occlusions(image: Image.Image, patches: list[dict]) -> Image.Image:
    """Paint each ``{x1,y1,x2,y2,fill}`` patch (pixel coords) onto a copy."""
    out = image.convert(image.mode).copy()
    draw = ImageDraw.Draw(out)
    for p in patches:
        fill = p.get("fill", "gray")
        if fill not in OCCLUDE_FILLS:
            raise ValueError(f"occlusion fill must be one of {OCCLUDE_FILLS}, got {fill!r}")
        draw.rectangle(
            [p["x1"], p["y1"], p["x2"], p["y2"]],
            fill=_occlusion_fill_value(out, fill),
        )
    return out


def _covered_fraction(box_px: tuple[float, float, float, float], patches: list[dict]) -> float:
    """Largest single-patch fraction of ``box_px`` that is covered."""
    bx1, by1, bx2, by2 = box_px
    box_area = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    if box_area <= 0:
        return 1.0
    worst = 0.0
    for p in patches:
        ix1, iy1 = max(bx1, p["x1"]), max(by1, p["y1"])
        ix2, iy2 = min(bx2, p["x2"]), min(by2, p["y2"])
        inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
        worst = max(worst, inter / box_area)
    return worst


def _out_stem(src_stem: str, dx: int, dy: int, angle: float, n_occ: int) -> str:
    stem = f"{src_stem}_shift_x{dx}_y{dy}"
    if angle:
        stem += f"_rot{angle:g}"
    if n_occ:
        stem += f"_occ{n_occ}"
    return stem


def augment_and_save(
    image_path: Path,
    dx: int = 0,
    dy: int = 0,
    fill: str = "edge",
    angle: float = 0.0,
    occlusions: list[dict] | None = None,
    label_path: Path | None = None,
    images_dir: Path = Path("dataset/images"),
    labels_dir: Path = Path("dataset/labels"),
    previews_dir: Path = Path("dataset/previews"),
    manifest_path: Path | None = DEFAULT_MANIFEST,
) -> dict:
    """Shift + rotate + occlude one labeled image and record the result.

    Returns the manifest record for the generated sample.
    """
    image_path = Path(image_path)
    label_path = Path(label_path) if label_path else labels_dir / f"{image_path.stem}.txt"
    occlusions = list(occlusions or [])

    image = Image.open(image_path)
    img_w, img_h = image.size

    boxes = read_label(label_path)
    if not boxes:
        raise ValueError(f"No boxes in label {label_path}")
    if len(boxes) > 1:
        raise ValueError(f"{label_path} has {len(boxes)} boxes; augmentation expects exactly 1")
    class_id, cx, cy, w, h = boxes[0]
    src_px = bbox_yolo_to_pixel(cx, cy, w, h, img_w, img_h)

    new_px = shift_bbox(*src_px, dx, dy, img_w, img_h)
    if angle:
        new_px = rotate_bbox(*new_px, angle, img_w, img_h)
    if new_px[2] - new_px[0] < 1 or new_px[3] - new_px[1] < 1:
        raise ValueError(
            f"Transform (dx={dx}, dy={dy}, angle={angle}) pushes the box out of frame "
            f"(src px {tuple(map(round, src_px))} -> {tuple(map(round, new_px))})"
        )
    if occlusions and _covered_fraction(new_px, occlusions) >= 0.9:
        raise ValueError("occlusion(s) cover >=90% of the box; the pot would be invisible")
    new_cx, new_cy, new_w, new_h = bbox_pixel_to_yolo(*new_px, img_w, img_h)

    stem = _out_stem(image_path.stem, dx, dy, angle, len(occlusions))
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    out_img = shift_image(image, dx, dy, fill=fill)
    if angle:
        out_img = rotate_image(out_img, angle, fill=fill)
    if occlusions:
        out_img = apply_occlusions(out_img, occlusions)
    image_out = images_dir / f"{stem}.png"
    out_img.save(image_out)

    label_out = labels_dir / f"{stem}.txt"
    write_label(label_out, class_id, new_cx, new_cy, new_w, new_h)

    preview = out_img.convert("RGB")
    ImageDraw.Draw(preview).rectangle(new_px, outline="red", width=4)
    preview_out = previews_dir / f"{stem}_preview.png"
    preview.save(preview_out)

    record = {
        "source": str(image_path),
        "source_label": str(label_path),
        "dx": dx,
        "dy": dy,
        "fill": fill,
        "angle": angle,
        "occlusions": occlusions,
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
                e.get("dx", 0),
                e.get("dy", 0),
                fill=e.get("fill", "edge"),
                angle=e.get("angle", 0.0),
                occlusions=e.get("occlusions") or [],
                label_path=Path(e["source_label"]) if e.get("source_label") else None,
                images_dir=Path(e["image"]).parent,
                labels_dir=Path(e["label"]).parent,
                previews_dir=Path(e["preview"]).parent,
                manifest_path=manifest_path,
            )
        )
    return out


def occlusions_for_coverage(
    rng: random.Random,
    box_px: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    coverage: float,
) -> list[dict]:
    """One slab covering ~``coverage`` of the box area, on a random side.

    Unlike :func:`_random_occlusions` this always returns **exactly one** patch
    (the caller guarantees "at least one obstruction per image") and its size is
    driven by an explicit target fraction so a batch can sweep a range of
    "percentage blocked". The slab spans the full box on its minor axis, so the
    covered area fraction is ~= its extent fraction on the major axis.
    """
    bx1, by1, bx2, by2 = box_px
    bw, bh = bx2 - bx1, by2 - by1
    cover = max(0.05, min(coverage, 0.85))
    if rng.random() < 0.5:  # vertical slab: covers the left or right `cover` of the box
        span = cover * bw
        x1 = bx1 if rng.random() < 0.5 else bx2 - span
        patch = {"x1": x1, "y1": by1 - bh * 0.15, "x2": x1 + span, "y2": by2 + bh * 0.15}
    else:  # horizontal slab: covers the top or bottom `cover` of the box
        span = cover * bh
        y1 = by1 if rng.random() < 0.5 else by2 - span
        patch = {"x1": bx1 - bw * 0.15, "y1": y1, "x2": bx2 + bw * 0.15, "y2": y1 + span}
    patch = {k: round(max(0.0, min(v, img_w if "x" in k else img_h)), 1) for k, v in patch.items()}
    patch["fill"] = rng.choice(OCCLUDE_FILLS)
    return [patch]


def _random_occlusions(
    rng: random.Random, box_px: tuple[float, float, float, float], img_w: int, img_h: int
) -> list[dict]:
    """0-2 rectangles that clip an edge of the box without burying it."""
    bx1, by1, bx2, by2 = box_px
    bw, bh = bx2 - bx1, by2 - by1
    patches = []
    for _ in range(rng.choice((0, 1, 1, 2))):
        # a slab covering one side of the box, 25-55% of its extent
        if rng.random() < 0.5:  # vertical slab (covers left or right)
            cover = rng.uniform(0.25, 0.55) * bw
            x1 = bx1 - bw * 0.1 if rng.random() < 0.5 else bx2 - cover
            patch = {"x1": x1, "y1": by1 - bh * 0.2, "x2": x1 + cover, "y2": by2 + bh * 0.2}
        else:  # horizontal slab (covers top or bottom)
            cover = rng.uniform(0.25, 0.55) * bh
            y1 = by1 - bh * 0.1 if rng.random() < 0.5 else by2 - cover
            patch = {"x1": bx1 - bw * 0.2, "y1": y1, "x2": bx2 + bw * 0.2, "y2": y1 + cover}
        patch = {k: round(max(0.0, min(v, img_w if "x" in k else img_h)), 1) for k, v in patch.items()}
        patch["fill"] = rng.choice(OCCLUDE_FILLS)
        patches.append(patch)
    return patches


def generate(
    source: Path,
    count: int,
    *,
    seed: int = 0,
    max_shift: int = 120,
    max_angle: float = 15.0,
    fill: str = "edge",
    label_path: Path | None = None,
    images_dir: Path = Path("dataset/images"),
    labels_dir: Path = Path("dataset/labels"),
    previews_dir: Path = Path("dataset/previews"),
    manifest_path: Path | None = DEFAULT_MANIFEST,
) -> list[dict]:
    """Seeded batch of random shift+rotate+occlude combos from ``source``.

    Deterministic for a given ``(source, count, seed)``. Combos that push the
    box out of frame or bury it are skipped, so you may get slightly fewer than
    ``count`` samples; the shortfall is reported by the CLI.
    """
    source = Path(source)
    label_path = Path(label_path) if label_path else labels_dir / f"{source.stem}.txt"
    rng = random.Random(f"{source}:{count}:{seed}")

    with Image.open(source) as im:
        img_w, img_h = im.size
    boxes = read_label(label_path)
    if len(boxes) != 1:
        raise ValueError(f"{label_path} must have exactly 1 box, has {len(boxes)}")
    _, cx, cy, w, h = boxes[0]
    box_px = bbox_yolo_to_pixel(cx, cy, w, h, img_w, img_h)

    out: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < count and attempts < count * 20:
        attempts += 1
        dx = rng.randint(-max_shift, max_shift)
        dy = rng.randint(-max_shift, max_shift)
        angle = round(rng.uniform(-max_angle, max_angle), 1)
        shifted = shift_bbox(*box_px, dx, dy, img_w, img_h)
        occ = _random_occlusions(rng, shifted, img_w, img_h)
        stem = _out_stem(source.stem, dx, dy, angle, len(occ))
        if stem in seen:
            continue
        seen.add(stem)
        try:
            out.append(
                augment_and_save(
                    source, dx, dy, fill=fill, angle=angle, occlusions=occ,
                    label_path=label_path, images_dir=images_dir, labels_dir=labels_dir,
                    previews_dir=previews_dir, manifest_path=manifest_path,
                )
            )
        except ValueError:
            continue
    return out


def generate_balanced(
    source: Path,
    *,
    n_shift: int = 3,
    n_rotate: int = 3,
    n_occlude: int = 1,
    seed: int = 0,
    max_shift: int = 80,
    rotate_max_shift: int = 40,
    min_angle: float = 3.0,
    max_angle: float = 12.0,
    occ_min_cover: float = 0.1,
    occ_max_cover: float = 0.6,
    fill: str = "edge",
    label_path: Path | None = None,
    images_dir: Path = Path("dataset/images"),
    labels_dir: Path = Path("dataset/labels"),
    previews_dir: Path = Path("dataset/previews"),
    manifest_path: Path | None = DEFAULT_MANIFEST,
) -> list[dict]:
    """Emit a fixed *recipe* of augmentations from one frame, by category.

    Rather than ``generate``'s uniform "every sample is a random
    shift+rotate+occlude combo", this produces three disjoint blocks so the
    dataset balance is controllable:

      * ``n_shift``   -- pure translations (no rotation, no occlusion). Together
        with the untouched originals these are meant to be the bulk of train.
      * ``n_rotate``  -- a small shift plus a genuine rotation
        (``|angle| >= min_angle``), no occlusion.
      * ``n_occlude`` -- a moderate shift, sometimes rotated, always with
        **exactly one** occluding slab whose covered fraction is swept linearly
        across ``[occ_min_cover, occ_max_cover]``. Kept a minority on purpose.

    Deterministic for a given ``(source, seed)`` and each block's counts.
    """
    source = Path(source)
    label_path = Path(label_path) if label_path else labels_dir / f"{source.stem}.txt"
    rng = random.Random(f"{source}:balanced:{seed}")

    with Image.open(source) as im:
        img_w, img_h = im.size
    boxes = read_label(label_path)
    if len(boxes) != 1:
        raise ValueError(f"{label_path} must have exactly 1 box, has {len(boxes)}")
    _, cx, cy, w, h = boxes[0]
    box_px = bbox_yolo_to_pixel(cx, cy, w, h, img_w, img_h)

    plans: list[tuple[int, int, float, list[dict]]] = []  # (dx, dy, angle, occlusions)
    for _ in range(max(0, n_shift)):
        plans.append((rng.randint(-max_shift, max_shift), rng.randint(-max_shift, max_shift), 0.0, []))
    for _ in range(max(0, n_rotate)):
        mag = rng.uniform(min_angle, max_angle)
        angle = round(mag if rng.random() < 0.5 else -mag, 1)
        plans.append(
            (
                rng.randint(-rotate_max_shift, rotate_max_shift),
                rng.randint(-rotate_max_shift, rotate_max_shift),
                angle,
                [],
            )
        )
    n_occ = max(0, n_occlude)
    for i in range(n_occ):
        # Spread "percentage blocked" across [occ_min_cover, occ_max_cover]: an
        # even sweep when several occlusions come off one frame, a seeded random
        # draw when there's only one (so the spread still shows up frame-to-frame).
        if n_occ == 1:
            coverage = rng.uniform(occ_min_cover, occ_max_cover)
        else:
            frac = i / (n_occ - 1)
            coverage = occ_min_cover + frac * (occ_max_cover - occ_min_cover)
        dx = rng.randint(-rotate_max_shift, rotate_max_shift)
        dy = rng.randint(-rotate_max_shift, rotate_max_shift)
        angle = 0.0
        if rng.random() < 0.5:
            mag = rng.uniform(min_angle, max_angle)
            angle = round(mag if rng.random() < 0.5 else -mag, 1)
        shifted = shift_bbox(*box_px, dx, dy, img_w, img_h)
        occ = occlusions_for_coverage(rng, shifted, img_w, img_h, coverage)
        plans.append((dx, dy, angle, occ))

    out: list[dict] = []
    seen: set[str] = set()
    for dx, dy, angle, occ in plans:
        stem = _out_stem(source.stem, dx, dy, angle, len(occ))
        if stem in seen:
            continue
        seen.add(stem)
        try:
            out.append(
                augment_and_save(
                    source, dx, dy, fill=fill, angle=angle, occlusions=occ,
                    label_path=label_path, images_dir=images_dir, labels_dir=labels_dir,
                    previews_dir=previews_dir, manifest_path=manifest_path,
                )
            )
        except ValueError:
            continue
    return out


def generate_from_list(
    train_list: Path,
    per: int = 4,
    *,
    dataset_dir: Path = Path("dataset"),
    seed: int = 0,
    max_shift: int = 80,
    max_angle: float = 12.0,
    fill: str = "edge",
    balanced: bool = False,
    n_shift: int = 3,
    n_rotate: int = 3,
    n_occlude: int = 1,
    occ_min_cover: float = 0.1,
    occ_max_cover: float = 0.6,
    manifest_path: Path | None = DEFAULT_MANIFEST,
) -> dict:
    """Augment every positive single-box frame listed in ``train_list``.

    ``train_list`` is a YOLO image-list file (``./images/<name>`` per line,
    relative to ``dataset_dir``) -- normally ``dataset/train.txt`` straight after
    ``dataset promote``, so only real *training* frames are touched and the
    copies stay pinned to train (leakage-safe). Negatives (empty label) and
    multi-box frames are skipped. Deterministic: each source gets its own RNG
    stream keyed by its name, so re-running is idempotent and adding frames does
    not disturb existing ones.

    With ``balanced=True`` each source is expanded by :func:`generate_balanced`
    (``n_shift`` pure shifts + ``n_rotate`` rotations + ``n_occlude`` occlusions
    sweeping ``[occ_min_cover, occ_max_cover]``) instead of ``per`` uniform
    random combos.

    Returns ``{"sources": n, "generated": m, "skipped": k}``.
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    previews_dir = dataset_dir / "previews"

    entries = [
        line.strip() for line in Path(train_list).read_text().splitlines() if line.strip()
    ]
    sources = skipped = generated = 0
    for entry in entries:
        name = entry.split("/")[-1]
        if "_shift_x" in name or name.startswith("kahvi"):
            continue  # already-synthetic; don't augment augmentations
        src = images_dir / name
        label = labels_dir / f"{src.stem}.txt"
        if not src.exists() or not label.exists():
            skipped += 1
            continue
        boxes = read_label(label)
        if len(boxes) != 1:  # negative or multi-box
            skipped += 1
            continue
        sources += 1
        if balanced:
            records = generate_balanced(
                src, n_shift=n_shift, n_rotate=n_rotate, n_occlude=n_occlude,
                seed=seed, max_shift=max_shift, max_angle=max_angle,
                occ_min_cover=occ_min_cover, occ_max_cover=occ_max_cover, fill=fill,
                label_path=label, images_dir=images_dir, labels_dir=labels_dir,
                previews_dir=previews_dir, manifest_path=manifest_path,
            )
        else:
            records = generate(
                src, per, seed=seed, max_shift=max_shift, max_angle=max_angle, fill=fill,
                label_path=label, images_dir=images_dir, labels_dir=labels_dir,
                previews_dir=previews_dir, manifest_path=manifest_path,
            )
        generated += len(records)
    return {"sources": sources, "generated": generated, "skipped": skipped}


def _parse_occlude(spec: str) -> dict:
    parts = spec.split(",")
    if len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError("--occlude wants x1,y1,x2,y2[,fill]")
    x1, y1, x2, y2 = (float(v) for v in parts[:4])
    fill = parts[4] if len(parts) == 5 else "gray"
    if fill not in OCCLUDE_FILLS:
        raise argparse.ArgumentTypeError(f"occlude fill must be one of {OCCLUDE_FILLS}")
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "fill": fill}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, nargs="?", help="e.g. dataset/images/kahvi.png")
    parser.add_argument("--dx", type=int, default=0, help="pixels to shift right (negative = left)")
    parser.add_argument("--dy", type=int, default=0, help="pixels to shift down (negative = up)")
    parser.add_argument("--angle", type=float, default=0.0, help="degrees to rotate counter-clockwise")
    parser.add_argument(
        "--occlude", type=_parse_occlude, action="append", default=None, metavar="x1,y1,x2,y2[,fill]",
        help=f"opaque patch in pixels; repeatable. fill in {OCCLUDE_FILLS} (default gray)",
    )
    parser.add_argument("--fill", choices=FILL_MODES, default="edge", help="how to fill exposed border (default: edge)")
    parser.add_argument("--label", type=Path, default=None, help="source label (default: dataset/labels/<stem>.txt)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replay", action="store_true", help="regenerate all artifacts from the manifest and exit")
    parser.add_argument(
        "--generate", type=int, metavar="COUNT", default=None,
        help="generate COUNT random shift+rotate+occlude combos (seeded) and exit; "
        "image defaults to dataset/images/kahvi.png",
    )
    parser.add_argument(
        "--generate-from", type=Path, metavar="TRAIN_LIST", default=None,
        help="augment every positive single-box frame listed in TRAIN_LIST "
        "(e.g. dataset/train.txt, run after `dataset promote`) and exit",
    )
    parser.add_argument("--per", type=int, default=4, help="--generate-from: combos per source frame (non-balanced)")
    parser.add_argument(
        "--balanced", action="store_true",
        help="--generate-from: emit a fixed recipe per frame (n-shift pure shifts + "
        "n-rotate rotations + n-occlude occlusions) instead of --per random combos",
    )
    parser.add_argument("--n-shift", type=int, default=3, help="--balanced: pure-shift copies per frame")
    parser.add_argument("--n-rotate", type=int, default=3, help="--balanced: shift+rotate copies per frame")
    parser.add_argument("--n-occlude", type=int, default=1, help="--balanced: occluded copies per frame (>=1 patch each)")
    parser.add_argument("--occ-min-cover", type=float, default=0.1, help="--balanced: min box fraction an occlusion covers")
    parser.add_argument("--occ-max-cover", type=float, default=0.6, help="--balanced: max box fraction an occlusion covers")
    parser.add_argument("--seed", type=int, default=0, help="--generate/--generate-from RNG seed")
    parser.add_argument(
        "--max-shift", type=int, default=None,
        help="max |dx|,|dy| in pixels (default 120 for --generate, 80 for --generate-from)",
    )
    parser.add_argument(
        "--max-angle", type=float, default=None,
        help="max |rotation| in degrees (default 15 for --generate, 12 for --generate-from)",
    )

    args = parser.parse_args()

    if args.generate_from is not None:
        kw = {"seed": args.seed, "fill": args.fill, "manifest_path": args.manifest}
        if args.max_shift is not None:
            kw["max_shift"] = args.max_shift
        if args.max_angle is not None:
            kw["max_angle"] = args.max_angle
        if args.balanced:
            kw.update(
                balanced=True, n_shift=args.n_shift, n_rotate=args.n_rotate,
                n_occlude=args.n_occlude, occ_min_cover=args.occ_min_cover,
                occ_max_cover=args.occ_max_cover,
            )
        stats = generate_from_list(args.generate_from, args.per, **kw)
        recipe = (
            f"recipe {args.n_shift} shift / {args.n_rotate} rot / {args.n_occlude} occ"
            if args.balanced
            else f"x{args.per}"
        )
        print(
            f"Augmented {stats['sources']} frame(s) {recipe} -> {stats['generated']} sample(s), "
            f"{stats['skipped']} skipped (negative / multi-box / missing) -> {args.manifest}"
        )
        return

    if args.generate is not None:
        source = args.image or Path("dataset/images/kahvi.png")
        records = generate(
            source, args.generate, seed=args.seed,
            max_shift=args.max_shift if args.max_shift is not None else 120,
            max_angle=args.max_angle if args.max_angle is not None else 15.0,
            fill=args.fill, label_path=args.label, manifest_path=args.manifest,
        )
        print(f"Generated {len(records)} augmentation(s) (requested {args.generate}) -> {args.manifest}")
        if len(records) < args.generate:
            print(f"  {args.generate - len(records)} combo(s) skipped (box left frame or buried)")
        return

    if args.replay:
        records = replay(args.manifest)
        print(f"Replayed {len(records)} augmentation(s) from {args.manifest}")
        return

    if args.image is None or (args.dx == 0 and args.dy == 0 and args.angle == 0.0 and not args.occlude):
        parser.error("give an image and at least one of --dx/--dy/--angle/--occlude, or use 'generate' / --replay")

    record = augment_and_save(
        args.image,
        args.dx,
        args.dy,
        fill=args.fill,
        angle=args.angle,
        occlusions=args.occlude or [],
        label_path=args.label,
        manifest_path=args.manifest,
    )
    print(f"shift (dx={args.dx}, dy={args.dy}), angle={args.angle}, occlusions={len(args.occlude or [])}, fill={args.fill}")
    print(f"  bbox px : {tuple(round(v) for v in record['src_bbox_px'])} -> {tuple(round(v) for v in record['new_bbox_px'])}")
    print(f"  image   : {record['image']}")
    print(f"  label   : {record['label']}")
    print(f"  preview : {record['preview']}")
    print(f"  manifest: {args.manifest}")


if __name__ == "__main__":
    main()
