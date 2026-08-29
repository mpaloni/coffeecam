"""Stitch the shift-augmentation previews into one animated GIF.

Reads ``dataset/augmentations.json`` (falling back to a glob of the preview
PNGs), orders the frames so the box sweeps around the frame — by angle of the
(dx, dy) shift, then by magnitude — stamps each with its dx/dy, and writes an
animated GIF. Handy for eyeballing the whole augmented set at once.

    .venv/bin/python -m coffeecam.preview_anim [--out dataset/shift_previews.gif] [--ms 350]

No ffmpeg needed; Pillow writes the GIF directly.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw

DEFAULT_MANIFEST = Path("dataset/augmentations.json")
DEFAULT_OUT = Path("dataset/shift_previews.gif")
_STEM_RE = re.compile(r"_shift_x(-?\d+)_y(-?\d+)_preview$")


def _frames_from_manifest(manifest_path: Path) -> list[tuple[int, int, Path]]:
    entries = json.loads(manifest_path.read_text())
    out = []
    for e in entries:
        p = Path(e["preview"])
        if p.exists():
            out.append((int(e["dx"]), int(e["dy"]), p))
    return out


def _frames_from_glob(previews_dir: Path) -> list[tuple[int, int, Path]]:
    out = []
    for p in previews_dir.glob("*_shift_x*_y*_preview.png"):
        m = _STEM_RE.search(p.stem)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), p))
    return out


def collect_frames(
    manifest_path: Path = DEFAULT_MANIFEST, previews_dir: Path = Path("dataset/previews")
) -> list[tuple[int, int, Path]]:
    frames = _frames_from_manifest(manifest_path) if manifest_path.exists() else []
    if not frames:
        frames = _frames_from_glob(previews_dir)
    # sweep: by angle of the shift vector, then by distance from origin
    frames.sort(key=lambda f: (math.atan2(f[1], f[0]), math.hypot(f[0], f[1])))
    return frames


def build_anim(
    out_path: Path = DEFAULT_OUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    previews_dir: Path = Path("dataset/previews"),
    ms_per_frame: int = 350,
    label: bool = True,
) -> Path:
    frames = collect_frames(manifest_path, previews_dir)
    if not frames:
        raise FileNotFoundError("No shift previews found (manifest empty and none on disk).")

    images = []
    for dx, dy, path in frames:
        im = Image.open(path).convert("RGB")
        if label:
            draw = ImageDraw.Draw(im)
            text = f"dx={dx:+d}  dy={dy:+d}"
            draw.rectangle((6, 6, 6 + 9 * len(text), 22), fill="black")
            draw.text((10, 8), text, fill="white")
        images.append(im)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=ms_per_frame,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--previews-dir", type=Path, default=Path("dataset/previews"))
    parser.add_argument("--ms", type=int, default=350, help="milliseconds per frame")
    parser.add_argument("--no-label", action="store_true", help="don't stamp dx/dy on each frame")
    args = parser.parse_args()

    out = build_anim(
        args.out,
        manifest_path=args.manifest,
        previews_dir=args.previews_dir,
        ms_per_frame=args.ms,
        label=not args.no_label,
    )
    n = len(collect_frames(args.manifest, args.previews_dir))
    print(f"Wrote {out} ({n} frames, {args.ms}ms each)")


if __name__ == "__main__":
    main()
