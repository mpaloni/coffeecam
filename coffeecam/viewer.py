"""Scrubbable version of the capture timelapse: annotated frames + a manifest.

`coffeecam.summary` stitches every captured frame into one animated GIF, which
plays but can't be paused or seeked. This builds the same per-frame annotation
(same normalize -> detect -> map-box-back path as `coffeecam.pipeline`) but keeps
the frames *separate* — a list of JPEG bytes plus a manifest describing each one
(source file path, timestamp, what the detector saw) — so a tiny HTML page can
offer play/pause, a frame slider, and a live readout of the current frame's path.

    .venv/bin/python -m coffeecam.viewer --out-dir viewer_frames/
    .venv/bin/python -m coffeecam.viewer --no-annotate --max-frames 120 --scale 1

The Flask server serves this at ``/viewer`` (see `coffeecam.server`); the frames
go out as ``/viewer/frame/<i>.jpg`` and the manifest as ``/viewer/manifest.json``.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from coffeecam.pipeline import DEFAULT_CONF
from coffeecam.summary import (
    DEFAULT_CAPTURES_DIR,
    SummaryEmpty,
    _render_frame,
    _sample,
    collect_frames,
)

DEFAULT_MAX_FRAMES = 160
DEFAULT_JPEG_QUALITY = 85


@dataclass
class ViewerBuild:
    frames: list[bytes]  # annotated JPEG per rendered frame, in playback order
    manifest: list[dict]  # one entry per frame: i, path, rel, ts, kind
    meta: dict

    def frame(self, i: int) -> bytes:
        """1-indexed frame lookup (matches the manifest's ``i`` and the GIF labels)."""
        if not 1 <= i <= len(self.frames):
            raise IndexError(i)
        return self.frames[i - 1]


def _encode_jpeg(img, quality: int) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def build_viewer(
    *,
    captures_dir: Path = DEFAULT_CAPTURES_DIR,
    model=None,
    conf: float = DEFAULT_CONF,
    max_frames: int = DEFAULT_MAX_FRAMES,
    scale: float = 1.0,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> ViewerBuild:
    """Render the sampled capture frames individually. Pass ``model`` to draw the
    detector's box on each (as `coffeecam.summary` does); leave it ``None`` for a
    plain timelapse. Raises `SummaryEmpty` when there are no captures yet."""
    t0 = time.perf_counter()
    refs = collect_frames(captures_dir)
    if not refs:
        raise SummaryEmpty(f"no capture frames under {captures_dir}/")

    picked = _sample(refs, max_frames)
    counts = {"strong": 0, "weak_only": 0, "no_box": 0}
    frames: list[bytes] = []
    manifest: list[dict] = []
    for i, ref in enumerate(picked, 1):
        im, kind = _render_frame(ref, model=model, conf=conf, scale=scale, index=i, n=len(picked))
        counts[kind] += 1
        frames.append(_encode_jpeg(im, jpeg_quality))
        manifest.append(
            {
                "i": i,
                "path": str(ref.path),
                "rel": f"{ref.day}/{ref.path.name}",
                "ts": ref.ts_label,
                "kind": kind,
            }
        )

    meta = {
        "source_frames": len(refs),
        "rendered_frames": len(picked),
        "annotated": model is not None,
        "conf_floor": round(conf, 3),
        "scale": scale,
        "span": [picked[0].ts_label, picked[-1].ts_label],
        "bytes": sum(len(f) for f in frames),
        "build_secs": round(time.perf_counter() - t0, 1),
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    if model is not None:
        meta["detections"] = counts

    return ViewerBuild(frames=frames, manifest=manifest, meta=meta)


def dump(build: ViewerBuild, out_dir: Path) -> Path:
    """Write ``frame_0001.jpg`` … and ``manifest.json`` into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry, data in zip(build.manifest, build.frames):
        (out_dir / f"frame_{entry['i']:04d}.jpg").write_bytes(data)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"frames": build.manifest, "meta": build.meta}, indent=2))
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("viewer_frames"))
    ap.add_argument("--captures-dir", type=Path, default=DEFAULT_CAPTURES_DIR)
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence floor for a 'strong' box")
    ap.add_argument("--quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality 1-95")
    ap.add_argument("--no-annotate", action="store_true", help="plain timelapse, don't run the detector")
    args = ap.parse_args(argv)

    model = None
    if not args.no_annotate:
        from coffeecam.detect import load_model

        model = load_model()

    try:
        build = build_viewer(
            captures_dir=args.captures_dir,
            model=model,
            conf=args.conf,
            max_frames=args.max_frames,
            scale=args.scale,
            jpeg_quality=args.quality,
        )
    except SummaryEmpty as exc:
        ap.error(str(exc))

    manifest_path = dump(build, args.out_dir)
    m = build.meta
    print(
        f"wrote {len(build.frames)} frames + {manifest_path.name} to {args.out_dir}/  "
        f"({m['bytes'] / 1024:.0f} KB, {m['rendered_frames']}/{m['source_frames']} frames, {m['build_secs']}s)"
    )
    if m.get("detections"):
        print("detections:", m["detections"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
