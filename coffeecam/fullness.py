"""Fullness estimation for the detected pot crop.

There is no trained classifier yet (no labelled fill-level data). `BrightnessFullness`
is a deliberate placeholder: a glass carafe full of coffee is dark, an empty one is
bright (you see the white holder / background through it), so mean luminance of the
lower-centre of the crop is a rough proxy. It is uncalibrated — `method` says so in
the JSON. Swap in a `ModelFullness` with the same `.estimate()` once crops are
labelled; nothing else in the pipeline changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image

LEVELS = ("empty", "low", "half", "high", "full")


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
