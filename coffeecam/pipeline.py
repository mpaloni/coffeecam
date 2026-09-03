"""acquire -> detect -> crop -> classify, with every stage's image kept.

`run_pipeline` takes a raw camera snapshot and returns a `PipelineResult` holding
each intermediate image, per-stage timings, and a list of non-fatal errors (a
stage that fails is recorded and the pipeline continues degraded rather than
raising). The Flask server renders these; the classify stage is a placeholder
(`fullness.BrightnessFullness`) until a real model exists.

The raw, un-privacy-cropped frame is never stored on the result — `apply_transform`
(rotate 180 + privacy crop) runs first and only its output propagates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from PIL import Image, ImageDraw

from coffeecam.capture import apply_transform
from coffeecam.detect import Detection, detect_pot
from coffeecam.fullness import BrightnessFullness, FullnessEstimator, FullnessResult
from coffeecam.fullness_crop import DEFAULT_POT_BOX, prepare_crop
from coffeecam.normalize import map_bbox_back, match_training_frame

# Live frames detect ~0.29 bare / ~0.41 normalized; 0.15 clears the normalized
# path with margin without inviting noise.
DEFAULT_CONF = 0.15


@dataclass
class PipelineResult:
    ts: datetime
    frame: Image.Image  # privacy-cropped (rotate 180 + crop) — the "privacy filtered version"
    normalized: Image.Image | None  # black-padded to training aspect — what the detector sees
    bounded: Image.Image  # `frame` with the detection box drawn (copy of `frame` if none)
    crop: Image.Image | None  # prepare_crop() output — the classifier input; None only if transform failed
    detection: Detection | None  # bbox in `frame` coordinates
    fullness: FullnessResult
    timings_ms: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def run_pipeline(
    snapshot: Image.Image,
    *,
    model,
    estimator: FullnessEstimator | None = None,
    normalize: bool = True,
    conf: float = DEFAULT_CONF,
) -> PipelineResult:
    estimator = estimator or BrightnessFullness()
    timings: dict[str, float] = {}
    errors: list[str] = []

    def _timed(name, fn, default=None):
        t0 = time.perf_counter()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — degrade, don't crash the server loop
            errors.append(f"{name}: {exc}")
            return default
        finally:
            timings[name] = round((time.perf_counter() - t0) * 1000, 1)

    frame = _timed("transform", lambda: apply_transform(snapshot))
    if frame is None:  # transform failed — nothing downstream is meaningful
        return PipelineResult(
            ts=datetime.now(),
            frame=snapshot.convert("RGB"),
            normalized=None,
            bounded=snapshot.convert("RGB"),
            crop=None,
            detection=None,
            fullness=estimator.estimate(None),
            timings_ms=timings,
            errors=errors,
        )

    if normalize:
        normed_pair = _timed("normalize", lambda: match_training_frame(frame))
        normalized, pad = normed_pair if normed_pair else (None, None)
    else:
        normalized, pad = None, None

    detect_src = normalized if normalized is not None else frame
    raw_det = _timed("detect", lambda: detect_pot(detect_src, model, conf=conf))

    detection: Detection | None = None
    if raw_det is not None:
        if pad is not None:
            x1, y1, x2, y2 = map_bbox_back(raw_det.bbox, pad)
        else:
            x1, y1, x2, y2 = raw_det.bbox
        if x2 > x1 and y2 > y1:
            detection = Detection(x1, y1, x2, y2, raw_det.confidence)

    # `crop` is the model input: prepare_crop(box) with the detector box, or the
    # static DEFAULT_POT_BOX when the detector found nothing (camera is fixed, so
    # a static crop still classifies rather than degrading to "unknown"). Same
    # transform at train and inference time — see coffeecam/fullness_crop.py.
    crop_box = detection.bbox if detection is not None else DEFAULT_POT_BOX
    crop = _timed("crop", lambda: prepare_crop(frame, crop_box))

    bounded = frame.copy()
    if detection is not None:
        ImageDraw.Draw(bounded).rectangle(detection.bbox, outline="red", width=3)

    fullness = _timed(
        "classify", lambda: estimator.estimate(crop), default=FullnessResult("unknown", None, "error")
    )

    return PipelineResult(
        ts=datetime.now(),
        frame=frame,
        normalized=normalized,
        bounded=bounded,
        crop=crop,
        detection=detection,
        fullness=fullness,
        timings_ms=timings,
        errors=errors,
    )
