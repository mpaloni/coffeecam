"""fullness-v1 vs the brightness heuristic over the held-out test split.

`build_test_gif()` renders one frame per test crop — the crop, true/pred header,
`ModelFullness`'s per-class probabilities as a legend, and `BrightnessFullness`'s
guess dimmed below — and returns `(gif_bytes, scoreboard)`. Served at
`/fullness/compare.gif` / `.json`; also runnable:

    .venv/bin/python -m coffeecam.fullness_compare  --out scratchpad/fullness-v1-test.gif
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from coffeecam.fullness import BrightnessFullness, ModelFullness, resolve_fullness_weights
from coffeecam.summary import _FONT

DEFAULT_DATASET_DIR = Path("fullness_dataset")
# legend order, emptiest -> fullest; `absent` off the scale but shown
LEGEND = ("absent", "empty", "some", "lots")
_WALK = ("empty", "some", "lots", "absent")  # frame order
_ORD = {"empty": 0.0, "some": 0.5, "lots": 1.0}
_B2C = {"empty": "empty", "low": "some", "half": "some", "high": "lots", "full": "lots"}
_BAR = {"absent": (150, 150, 150), "empty": (90, 150, 220),
        "some": (230, 170, 60), "lots": (90, 200, 120)}
_CROP_PX, _PANEL_W, _PAD, _ROW_H = 256, 260, 12, 30


class NoTestData(RuntimeError):
    """`fullness_dataset/test/` is missing or empty (gitignored — run
    `python -m coffeecam.fullness_dataset --merge coarse --balance`)."""


def _load_items(dataset_dir: Path):
    test = Path(dataset_dir) / "test"
    items = []
    for c in _WALK:
        for p in sorted((test / c).glob("*.jpg")):
            with Image.open(p) as im:
                items.append((c, p.name, im.convert("RGB").copy()))
    if not items:
        raise NoTestData(NoTestData.__doc__)
    return items


def _rank(a):
    a = np.asarray(a, float)
    o = a.argsort()
    r = np.empty_like(o, float)
    r[o] = np.arange(len(a))
    return r


def _stats(items, preds, scores):
    idx = {c: i for i, c in enumerate(("empty", "some", "lots", "absent"))}
    cm = np.zeros((4, 4), int)
    for (t, _, _), p in zip(items, preds):
        cm[idx[t]][idx[p]] += 1
    recs = [cm[i, i] / cm[i].sum() if cm[i].sum() else float("nan") for i in range(4)]
    xs = [(_ORD[t], s) for (t, _, _), s in zip(items, scores) if t in _ORD and s is not None]
    a = np.array(xs) if xs else np.zeros((1, 2))
    hc = [1, 2]
    return {
        "raw_accuracy": round(float(np.trace(cm) / cm.sum()), 3),
        "balanced_accuracy": round(float(np.nanmean(recs)), 3),
        "balanced_accuracy_fill_only": round(float(np.nanmean(recs[:3])), 3),
        "recall": {c: (None if np.isnan(recs[idx[c]]) else round(recs[idx[c]], 3))
                   for c in ("empty", "some", "lots", "absent")},
        "has_coffee_recall": round(float(cm[np.ix_(hc, hc)].sum() / cm[hc].sum()), 3),
        "fill_score_mae": round(float(np.abs(a[:, 0] - a[:, 1]).mean()), 3),
        "fill_score_spearman": round(float(np.corrcoef(_rank(a[:, 0]), _rank(a[:, 1]))[0, 1]), 3),
        "confusion": cm.tolist(),
        "confusion_labels": ["empty", "some", "lots", "absent"],
    }


def _render(true_c, name, crop, mr, br, i, n):
    canvas = Image.new("RGB", (_CROP_PX + _PANEL_W, _CROP_PX + 22), (17, 17, 17))
    canvas.paste(crop.resize((_CROP_PX, _CROP_PX), Image.NEAREST), (0, 0))
    d = ImageDraw.Draw(canvas)
    x0 = _CROP_PX + _PAD
    ok = mr.level == true_c
    d.text((x0, 8), f"true  {true_c}", font=_FONT, fill=(235, 235, 235))
    d.text((x0, 8 + _ROW_H // 2 - 3),
           f"pred  {mr.level}  ({'ok' if ok else 'X'})  score {mr.score}",
           font=_FONT, fill=(90, 200, 120) if ok else (230, 90, 90))
    top = 8 + _ROW_H
    for k, c in enumerate(LEGEND):
        y = top + k * _ROW_H
        p = mr.detail["probs"].get(c, 0.0)
        d.text((x0, y), f"{c:>6}", font=_FONT, fill=(180, 180, 180))
        bx, full = x0 + 52, _PANEL_W - 52 - _PAD - 44
        d.rectangle((bx, y + 1, bx + full, y + 12), outline=(70, 70, 70))
        d.rectangle((bx, y + 1, bx + int(full * p), y + 12), fill=_BAR[c])
        d.text((bx + full + 4, y), f"{p * 100:4.1f}%", font=_FONT, fill=(200, 200, 200))
    by = top + len(LEGEND) * _ROW_H + 4
    d.text((x0, by), f"brightness: {_B2C[br.level]}  (score {br.score})",
           font=_FONT, fill=(120, 120, 120))
    d.text((6, _CROP_PX + 5), f"{name}   {i + 1}/{n}", font=_FONT, fill=(150, 150, 150))
    return canvas


def build_test_gif(*, weights: Path | None = None,
                   dataset_dir: Path = DEFAULT_DATASET_DIR,
                   duration_ms: int = 900) -> tuple[bytes, dict]:
    w = weights or resolve_fullness_weights()
    if w is None:
        raise NoTestData("no fullness weights — models/FULLNESS_CHECKPOINT unset")
    items = _load_items(dataset_dir)
    model, bright = ModelFullness(w), BrightnessFullness()
    mp = [model.estimate(im) for _, _, im in items]
    bp = [bright.estimate(im) for _, _, im in items]
    frames = [_render(t, n, im, mr, br, i, len(items))
              for i, ((t, n, im), mr, br) in enumerate(zip(items, mp, bp))]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    scoreboard = {
        "n": len(items),
        "fullness_v1": _stats(items, [r.level for r in mp], [r.score for r in mp]),
        "brightness_heuristic": _stats(items, [_B2C[r.level] for r in bp], [r.score for r in bp]),
    }
    return buf.getvalue(), scoreboard


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("scratchpad/fullness-v1-test.gif"))
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = ap.parse_args(argv)
    gif, sb = build_test_gif(dataset_dir=args.dataset_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(gif)
    import json
    print(json.dumps(sb, indent=2))
    print(f"\nwrote {args.out}  ({sb['n']} frames)")


if __name__ == "__main__":
    main()
