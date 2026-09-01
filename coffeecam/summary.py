"""Stitch every captured frame so far into one animated GIF timelapse.

The capture loop (`coffeecam.capture`) drops privacy-cropped frames into
``captures/YYYY-MM-DD/HHMMSS_fff.jpg``. This walks those day folders in
chronological order, optionally runs the current detector on each frame the same
way `coffeecam.pipeline` does (normalize -> detect -> map the box back to frame
pixels) and draws the result rectangle, stamps each frame with its timestamp, and
writes an animated GIF.

    .venv/bin/python -m coffeecam.summary                       # annotated, -> summary.gif
    .venv/bin/python -m coffeecam.summary --no-annotate --ms 80
    .venv/bin/python -m coffeecam.summary --max-frames 120 --scale 0.75

No ffmpeg needed; Pillow writes the GIF directly. The Flask server serves the
same thing at ``/summary`` (see `coffeecam.server`).
"""

from __future__ import annotations

import argparse
import io
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from coffeecam.detect import Detection, detect_pot
from coffeecam.normalize import map_bbox_back, match_training_frame
from coffeecam.pipeline import DEFAULT_CONF

DEFAULT_CAPTURES_DIR = Path("captures")
DEFAULT_MAX_FRAMES = 240
DEFAULT_DURATION_MS = 120
# Probe well below the pipeline's confidence floor so we can still show the box
# the model is *least unsure* about even when nothing clears the bar.
PROBE_CONF = 0.01

_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FONT = ImageFont.load_default(size=13)

_BOX_STRONG = (255, 64, 64)
_BOX_WEAK = (255, 165, 40)


class SummaryEmpty(RuntimeError):
    """Raised when there are no captured frames to summarise yet."""


@dataclass(frozen=True)
class FrameRef:
    path: Path
    day: str
    ts_label: str  # "YYYY-MM-DD HH:MM:SS"
    _key: tuple


def _hms(stem: str) -> str:
    digits = stem.split("_", 1)[0]
    if len(digits) >= 6 and digits[:6].isdigit():
        return f"{digits[0:2]}:{digits[2:4]}:{digits[4:6]}"
    return ""


def collect_frames(captures_dir: Path = DEFAULT_CAPTURES_DIR) -> list[FrameRef]:
    """Every ``captures/<day>/*.jpg`` in chronological order (excludes pipeline/)."""
    captures_dir = Path(captures_dir)
    if not captures_dir.is_dir():
        return []
    refs: list[FrameRef] = []
    for day_dir in sorted(p for p in captures_dir.iterdir() if p.is_dir() and _DAY_RE.match(p.name)):
        for jpg in sorted(day_dir.glob("*.jpg")):
            hms = _hms(jpg.stem)
            label = f"{day_dir.name} {hms}" if hms else f"{day_dir.name} {jpg.stem}"
            refs.append(FrameRef(jpg, day_dir.name, label, (day_dir.name, jpg.name)))
    refs.sort(key=lambda r: r._key)
    return refs


def _sample(seq: list[FrameRef], max_n: int) -> list[FrameRef]:
    """Evenly spaced subset that always keeps the first and last frame."""
    if max_n is None or max_n <= 0 or len(seq) <= max_n:
        return list(seq)
    if max_n == 1:
        return [seq[-1]]
    step = (len(seq) - 1) / (max_n - 1)
    idx = sorted({int(round(i * step)) for i in range(max_n)})
    return [seq[i] for i in idx]


def detect_on_frame(frame: Image.Image, model, *, conf: float = DEFAULT_CONF) -> tuple[Detection | None, bool]:
    """Best coffee_pot box for an already privacy-cropped frame, in frame pixels.

    Mirrors `coffeecam.pipeline`: pad to the training aspect, detect, map the box
    back. Returns ``(detection, is_strong)`` where ``is_strong`` is whether its
    confidence clears ``conf``; the box is still returned (for display) when it
    doesn't. ``(None, False)`` when the model finds nothing at all.
    """
    normed, pad = match_training_frame(frame)
    best = detect_pot(normed, model, conf=PROBE_CONF)
    if best is None:
        return None, False
    x1, y1, x2, y2 = map_bbox_back(best.bbox, pad)
    if not (x2 > x1 and y2 > y1):
        return None, False
    return Detection(x1, y1, x2, y2, best.confidence), best.confidence >= conf


def _label(draw: ImageDraw.ImageDraw, xy, text, *, fg, bg, anchor) -> None:
    l, t, r, b = draw.textbbox(xy, text, font=_FONT, anchor=anchor)
    draw.rectangle((l - 2, t - 1, r + 2, b + 1), fill=bg)
    draw.text(xy, text, fill=fg, font=_FONT, anchor=anchor)


def _render_frame(ref: FrameRef, *, model, conf: float, scale: float, index: int, n: int) -> tuple[Image.Image, str]:
    im = Image.open(ref.path).convert("RGB")

    det: Detection | None = None
    strong = False
    if model is not None:
        det, strong = detect_on_frame(im, model, conf=conf)

    if scale != 1.0:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))))
        if det is not None:
            det = Detection(*(int(v * scale) for v in det.bbox), det.confidence)

    draw = ImageDraw.Draw(im)
    kind = "no_box"
    if det is not None and strong:
        draw.rectangle(det.bbox, outline=_BOX_STRONG, width=3)
        _label(draw, (im.width - 4, 4), f"conf {det.confidence:.2f}",
               fg=(255, 255, 255), bg=(200, 32, 32), anchor="ra")
        kind = "strong"
    elif det is not None:
        draw.rectangle(det.bbox, outline=_BOX_WEAK, width=1)
        _label(draw, (im.width - 4, 4), f"best ~{det.confidence:.2f}",
               fg=(20, 20, 20), bg=_BOX_WEAK, anchor="ra")
        kind = "weak_only"
    elif model is not None:
        _label(draw, (im.width - 4, 4), "no box", fg=(210, 210, 210), bg=(40, 40, 40), anchor="ra")

    _label(draw, (4, 4), ref.ts_label, fg=(255, 255, 255), bg=(0, 0, 0), anchor="la")
    _label(draw, (im.width - 4, im.height - 4), f"{index}/{n}",
           fg=(185, 185, 185), bg=(0, 0, 0), anchor="rd")
    return im, kind


def build_summary_gif(
    *,
    captures_dir: Path = DEFAULT_CAPTURES_DIR,
    out: Path | None = None,
    model=None,
    conf: float = DEFAULT_CONF,
    max_frames: int = DEFAULT_MAX_FRAMES,
    duration_ms: int = DEFAULT_DURATION_MS,
    scale: float = 1.0,
) -> tuple[bytes, dict]:
    """Build the timelapse GIF. Pass ``model`` to annotate each frame with the
    detector's result box; leave it ``None`` for a plain timelapse."""
    t0 = time.perf_counter()
    refs = collect_frames(captures_dir)
    if not refs:
        raise SummaryEmpty(f"no capture frames under {captures_dir}/")

    picked = _sample(refs, max_frames)
    counts = {"strong": 0, "weak_only": 0, "no_box": 0}
    images: list[Image.Image] = []
    for i, ref in enumerate(picked, 1):
        im, kind = _render_frame(ref, model=model, conf=conf, scale=scale, index=i, n=len(picked))
        counts[kind] += 1
        images.append(im)

    buf = io.BytesIO()
    images[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    gif = buf.getvalue()

    meta = {
        "source_frames": len(refs),
        "rendered_frames": len(picked),
        "annotated": model is not None,
        "conf_floor": round(conf, 3),
        "span": [picked[0].ts_label, picked[-1].ts_label],
        "duration_ms": duration_ms,
        "scale": scale,
        "bytes": len(gif),
        "build_secs": round(time.perf_counter() - t0, 1),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    if model is not None:
        meta["detections"] = counts

    if out is not None:
        Path(out).write_bytes(gif)
    return gif, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("summary.gif"))
    ap.add_argument("--captures-dir", type=Path, default=DEFAULT_CAPTURES_DIR)
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--ms", type=int, default=DEFAULT_DURATION_MS, help="milliseconds per frame")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence floor for a 'strong' box")
    ap.add_argument("--no-annotate", action="store_true", help="plain timelapse, don't run the detector")
    args = ap.parse_args(argv)

    model = None
    if not args.no_annotate:
        from coffeecam.detect import load_model

        model = load_model()

    try:
        _, meta = build_summary_gif(
            captures_dir=args.captures_dir,
            out=args.out,
            model=model,
            conf=args.conf,
            max_frames=args.max_frames,
            duration_ms=args.ms,
            scale=args.scale,
        )
    except SummaryEmpty as exc:
        ap.error(str(exc))

    print(
        f"wrote {args.out}  ({meta['bytes'] / 1024:.0f} KB, "
        f"{meta['rendered_frames']}/{meta['source_frames']} frames, {meta['build_secs']}s)"
    )
    if meta.get("detections"):
        print("detections:", meta["detections"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
