"""Old-vs-new detector comparison: the same frames, every model's result box
drawn in its own panel side by side in one animated GIF, plus a per-model stats
diff (detection-kind counts, and on the labelled dataset splits the mAP /
precision / recall from ultralytics' own validator).

    .venv/bin/python -m coffeecam.compare \
        --model nomosaic-2=runs/detect/runs/train-nomosaic-2 \
        --model realdata-v1=runs/detect/runs/train-realdata-v1 \
        --model balanced-v1=models/best-balanced-v1.pt \
        --set test --out compare-oldvsnew.gif --stats-json compare-oldvsnew.json

``--set`` is any `summary.FRAMESETS` value (captures | train | val | test).
``--no-map`` skips the (slow) ``.val()`` pass; it is skipped automatically for
``--set captures`` since those frames have no labels.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw

from coffeecam.pipeline import DEFAULT_CONF
from coffeecam.summary import (
    DEFAULT_DATASET_DIR,
    FRAMESETS,
    SummaryEmpty,
    _BOX_STRONG,
    _BOX_WEAK,
    _FONT,
    _label,
    _sample,
    detect_on_frame,
    resolve_frames,
)
from coffeecam.detect import Detection

DEFAULT_DURATION_MS = 350
DEFAULT_MAX_FRAMES = 120
_PANEL_GAP = 4
_CAPTION_H = 18
_BG = (17, 17, 17)


def resolve_model_weights(spec: str) -> Path:
    """A ``name=path`` value's path half -> the ``.pt`` file. ``path`` may be the
    ``.pt`` itself or a run dir (``.../weights/best.pt`` is appended)."""
    p = Path(spec)
    if p.suffix == ".pt":
        return p
    return p / "weights" / "best.pt"


def parse_model_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--model must be NAME=PATH, got {raw!r}")
    name, _, spec = raw.partition("=")
    name = name.strip()
    weights = resolve_model_weights(spec.strip())
    if not weights.is_file():
        raise argparse.ArgumentTypeError(f"{name}: no weights at {weights}")
    return name, weights


def _panel(base: Image.Image, det: Detection | None, strong: bool, name: str) -> Image.Image:
    """One model's annotated copy of a frame, with a caption bar under it."""
    im = base.copy()
    draw = ImageDraw.Draw(im)
    if det is not None and strong:
        draw.rectangle(det.bbox, outline=_BOX_STRONG, width=3)
        tag, fg, bg = f"conf {det.confidence:.2f}", (255, 255, 255), (200, 32, 32)
    elif det is not None:
        draw.rectangle(det.bbox, outline=_BOX_WEAK, width=1)
        tag, fg, bg = f"best ~{det.confidence:.2f}", (20, 20, 20), _BOX_WEAK
    else:
        tag, fg, bg = "no box", (210, 210, 210), (40, 40, 40)
    _label(draw, (im.width - 4, 4), tag, fg=fg, bg=bg, anchor="ra")

    out = Image.new("RGB", (im.width, im.height + _CAPTION_H), _BG)
    out.paste(im, (0, 0))
    cap = ImageDraw.Draw(out)
    cap.text((4, im.height + _CAPTION_H // 2), name, fill=(235, 235, 235), font=_FONT, anchor="lm")
    return out


def build_comparison_gif(
    refs,
    named_models: dict,
    *,
    conf: float = DEFAULT_CONF,
    scale: float = 1.0,
    max_frames: int = DEFAULT_MAX_FRAMES,
    duration_ms: int = DEFAULT_DURATION_MS,
) -> tuple[bytes, dict]:
    """Stitch ``[panel(model_a) | panel(model_b) | ...]`` per frame into a GIF.

    ``named_models`` maps a short label to a loaded model (anything
    `summary.detect_on_frame` accepts). Returns ``(gif_bytes, stats)`` where
    ``stats['counts'][name]`` is ``{strong, weak_only, no_box}``.
    """
    if not refs:
        raise SummaryEmpty("no frames to compare")
    if not named_models:
        raise ValueError("need at least one model")

    t0 = time.perf_counter()
    picked = _sample(list(refs), max_frames)
    counts = {name: {"strong": 0, "weak_only": 0, "no_box": 0} for name in named_models}
    frames: list[Image.Image] = []

    for i, ref in enumerate(picked, 1):
        full = Image.open(ref.path).convert("RGB")
        dets: dict = {}
        for name, model in named_models.items():
            det, strong = detect_on_frame(full, model, conf=conf)
            dets[name] = (det, strong)
            kind = "strong" if (det is not None and strong) else "weak_only" if det is not None else "no_box"
            counts[name][kind] += 1

        base = full
        if scale != 1.0:
            base = full.resize((max(1, round(full.width * scale)), max(1, round(full.height * scale))))

        panels = []
        for name, (det, strong) in dets.items():
            d = det
            if d is not None and scale != 1.0:
                d = Detection(*(int(v * scale) for v in det.bbox), det.confidence)
            panels.append(_panel(base, d, strong, name))

        pw, ph = panels[0].width, panels[0].height
        strip = Image.new("RGB", (pw * len(panels) + _PANEL_GAP * (len(panels) - 1), ph + _CAPTION_H), _BG)
        for j, panel in enumerate(panels):
            strip.paste(panel, (j * (pw + _PANEL_GAP), _CAPTION_H))
        head = ImageDraw.Draw(strip)
        _label(head, (4, 2), f"{ref.ts_label}   {i}/{len(picked)}",
               fg=(255, 255, 255), bg=(0, 0, 0), anchor="la")
        frames.append(strip)

    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=True, disposal=2,
    )
    gif = buf.getvalue()
    stats = {
        "models": list(named_models),
        "source_frames": len(refs),
        "rendered_frames": len(picked),
        "conf_floor": round(conf, 3),
        "counts": counts,
        "bytes": len(gif),
        "build_secs": round(time.perf_counter() - t0, 1),
    }
    return gif, stats


def score_models(
    named_weights: dict[str, Path],
    *,
    split: str,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> dict[str, dict]:
    """Each model's ``.val()`` on ``dataset/<split>`` -> mAP / P / R.

    Uses each checkpoint's own train-time imgsz (ultralytics reads it back), so
    the numbers reflect how the model is actually used.
    """
    from ultralytics import YOLO

    data_yaml = Path(dataset_dir) / "data.yaml"
    out: dict[str, dict] = {}
    for name, weights in named_weights.items():
        res = YOLO(str(weights)).val(
            data=str(data_yaml), split=split, verbose=False, plots=False,
            project="runs/detect/runs", name=f"compare-val-{name}", exist_ok=True,
        )
        b = res.box
        out[name] = {
            "map50": round(float(b.map50), 4),
            "map50_95": round(float(b.map), 4),
            "precision": round(float(b.mp), 4),
            "recall": round(float(b.mr), 4),
        }
    return out


def _fmt_table(stats: dict, scores: dict | None) -> str:
    rows = ["model            strong  weak  none" + ("   mAP50  mAP50-95   P      R" if scores else "")]
    for name in stats["models"]:
        c = stats["counts"][name]
        line = f"{name:<15}  {c['strong']:>5}  {c['weak_only']:>4}  {c['no_box']:>4}"
        if scores and name in scores:
            s = scores[name]
            line += f"   {s['map50']:.3f}   {s['map50_95']:.3f}   {s['precision']:.3f}  {s['recall']:.3f}"
        rows.append(line)
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", dest="models", action="append", type=parse_model_arg, required=True,
                    metavar="NAME=PATH", help="repeatable; PATH is a .pt or a run dir")
    ap.add_argument("--set", dest="frameset", choices=FRAMESETS, default="test")
    ap.add_argument("--out", type=Path, default=Path("compare.gif"))
    ap.add_argument("--stats-json", type=Path, default=None)
    ap.add_argument("--captures-dir", type=Path, default=Path("captures"))
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--ms", type=int, default=DEFAULT_DURATION_MS)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--no-map", action="store_true", help="skip the .val() mAP pass")
    args = ap.parse_args(argv)

    named_weights = dict(args.models)
    from coffeecam.detect import load_model

    named_models = {name: load_model(w) for name, w in named_weights.items()}

    try:
        refs = resolve_frames(args.frameset, captures_dir=args.captures_dir, dataset_dir=args.dataset_dir)
        gif, stats = build_comparison_gif(
            refs, named_models, conf=args.conf, scale=args.scale,
            max_frames=args.max_frames, duration_ms=args.ms,
        )
    except SummaryEmpty as exc:
        ap.error(str(exc))

    scores = None
    if not args.no_map and args.frameset in ("train", "val", "test"):
        scores = score_models(named_weights, split=args.frameset, dataset_dir=args.dataset_dir)

    args.out.write_bytes(gif)
    stats = {"set": args.frameset, **stats}
    if scores:
        stats["scores"] = scores
    if args.stats_json:
        args.stats_json.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"wrote {args.out}  ({stats['bytes'] / 1024:.0f} KB, "
          f"{stats['rendered_frames']}/{stats['source_frames']} frames, {stats['build_secs']}s)")
    print(f"set={args.frameset}\n")
    print(_fmt_table(stats, scores))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
