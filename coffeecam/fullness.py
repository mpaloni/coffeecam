"""Fullness estimation for the detected pot crop.

`ModelFullness` is the real thing: a `yolov8n-cls` head over the 96 px
`prepare_crop` output, trained by `coffeecam.fullness_train` and pointed at by
`models/FULLNESS_CHECKPOINT`. `default_estimator()` returns it when those weights
resolve, else `NullFullness`.

`BrightnessFullness` is the retired placeholder — a glass carafe full of coffee
is dark, an empty one bright, so mean luminance of the lower-centre was a rough
proxy. Uncalibrated (`method` said so); kept only for reference / offline compare.
All three share `.estimate(crop) -> FullnessResult`, so the pipeline is agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

LEVELS = ("empty", "low", "half", "high", "full")

# Maps any class a fullness model might emit to a 0..1 fill scalar. Covers the
# raw 5-level scale and the `coarse` merge (empty/some/lots) from
# `fullness_dataset`. `absent` is off the scale -> score None.
_FILL_SCALAR = {
    "empty": 0.0, "low": 0.25, "some": 0.33, "half": 0.5,
    "high": 0.75, "lots": 0.83, "full": 1.0,
}

FULLNESS_CHECKPOINT_FILE = Path("models/FULLNESS_CHECKPOINT")


@dataclass(frozen=True)
class FullnessResult:
    level: str  # one of LEVELS, or "unknown"
    score: float | None  # 0.0 empty .. 1.0 full, or None
    method: str
    detail: dict = field(default_factory=dict)


class FullnessEstimator(Protocol):
    def estimate(self, crop: Image.Image | None) -> FullnessResult: ...


class NullFullness:
    """No estimate — the honest default until a classifier exists."""

    def estimate(self, crop: Image.Image | None) -> FullnessResult:
        return FullnessResult(level="unknown", score=None, method="none")


@dataclass
class BrightnessFullness:
    """Darker lower-centre of the crop => fuller. Uncalibrated proxy, not a model."""

    # Region of the crop to sample (fractions): the lower-centre, where liquid pools.
    x0: float = 0.25
    x1: float = 0.75
    y0: float = 0.45
    y1: float = 0.95
    # Luminance (0-255) mapping to the score: at/below dark_lum => full, at/above
    # bright_lum => empty, linear between.
    dark_lum: float = 60.0
    bright_lum: float = 170.0

    def estimate(self, crop: Image.Image | None) -> FullnessResult:
        if crop is None:
            return FullnessResult(level="unknown", score=None, method="brightness-heuristic")

        g = np.asarray(crop.convert("L"), dtype=np.float32)
        h, w = g.shape
        region = g[
            int(h * self.y0) : max(int(h * self.y1), int(h * self.y0) + 1),
            int(w * self.x0) : max(int(w * self.x1), int(w * self.x0) + 1),
        ]
        mean_lum = float(region.mean()) if region.size else float(g.mean())

        span = max(self.bright_lum - self.dark_lum, 1e-6)
        score = float(np.clip((self.bright_lum - mean_lum) / span, 0.0, 1.0))
        level = LEVELS[min(int(score * len(LEVELS)), len(LEVELS) - 1)]

        return FullnessResult(
            level=level,
            score=round(score, 3),
            method="brightness-heuristic",
            detail={"mean_luminance": round(mean_lum, 1), "uncalibrated": True},
        )


def resolve_fullness_weights(explicit: Path | None = None) -> Path | None:
    """`models/FULLNESS_CHECKPOINT` -> `<run>/weights/best.pt`, or ``None`` when
    no pointer / file exists (a fresh clone has no `runs/`). Unlike the
    detector's `resolve_weights`, absence is not an error — the pipeline falls
    back to `NullFullness`."""
    if explicit is not None:
        return explicit if Path(explicit).exists() else None
    if not FULLNESS_CHECKPOINT_FILE.exists():
        return None
    run = FULLNESS_CHECKPOINT_FILE.read_text().strip()
    if not run:
        return None
    weights = Path(run)
    if weights.suffix != ".pt":
        weights = weights / "weights" / "best.pt"
    return weights if weights.exists() else None


@dataclass
class ModelFullness:
    """`yolov8n-cls` over the `prepare_crop` output. `.estimate()` returns the
    argmax class as `level`, a 0..1 fill scalar as `score` (probability-weighted
    over the fill classes; ``None`` when the model calls `absent`), and the full
    prob vector in `detail`."""

    weights: Path
    _model: object = field(default=None, repr=False, compare=False)

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy: keep import optional

            self._model = YOLO(str(self.weights))
        return self._model

    def estimate(self, crop: Image.Image | None) -> FullnessResult:
        if crop is None:
            return FullnessResult(level="unknown", score=None, method="yolov8n-cls")

        res = self._load().predict(crop, verbose=False)[0]
        names = res.names
        probs = {names[i]: round(float(p), 4) for i, p in enumerate(res.probs.data.tolist())}
        level = names[int(res.probs.top1)]

        scale_mass = sum(probs[c] for c in probs if c in _FILL_SCALAR)
        if scale_mass > 0:
            score = round(
                sum(probs[c] * _FILL_SCALAR[c] for c in probs if c in _FILL_SCALAR)
                / scale_mass, 3,
            )
        else:  # model is confident it's `absent` / off-scale
            score = None

        return FullnessResult(
            level=level, score=score, method="yolov8n-cls", detail={"probs": probs},
        )


def default_estimator(weights: Path | None = None) -> FullnessEstimator:
    """`ModelFullness` when weights resolve, else `NullFullness`."""
    resolved = resolve_fullness_weights(weights)
    return ModelFullness(resolved) if resolved is not None else NullFullness()
