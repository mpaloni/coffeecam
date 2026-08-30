import numpy as np
import pytest
from PIL import Image

from coffeecam.fullness import BrightnessFullness, NullFullness
from coffeecam.normalize import TRAIN_ASPECT, map_bbox_back, match_training_frame
from coffeecam.pipeline import run_pipeline


# --------------------------------------------------------------------------- #
# fake YOLO model
# --------------------------------------------------------------------------- #
class _Scalar(float):
    """float that also supports `x[0]` — mimics a 1-elem tensor for both
    `float(box.conf)` and `box.conf[0]` in detect_pot."""

    def __getitem__(self, _i):
        return self


class _Box:
    def __init__(self, xyxy, conf):
        self.xyxy = np.array([xyxy], dtype=float)
        self.conf = _Scalar(conf)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = []

    def predict(self, source=None, conf=0.25, imgsz=320, verbose=False):
        self.calls.append({"conf": conf, "imgsz": imgsz})
        return [_Result([b for b in self._boxes if float(b.conf[0]) >= conf])]


def _snapshot(w=1280, h=720):
    return Image.new("RGB", (w, h), (90, 90, 90))


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #
def test_match_training_frame_hits_training_aspect():
    canvas, t = match_training_frame(Image.new("RGB", (424, 353)), top_strip=False)
    assert abs(canvas.width / canvas.height - TRAIN_ASPECT) < 0.02
    assert canvas.size >= (424, 353)  # never downscales
    assert (t.offset_x, t.offset_y) == ((canvas.width - 424) // 2, (canvas.height - 353) // 2)


def test_map_bbox_back_round_trips_interior_box():
    _, t = match_training_frame(Image.new("RGB", (400, 300)), top_strip=True)
    orig = (50, 40, 210, 260)
    canvas_box = (orig[0] + t.offset_x, orig[1] + t.offset_y, orig[2] + t.offset_x, orig[3] + t.offset_y)
    assert map_bbox_back(canvas_box, t) == orig


def test_map_bbox_back_clamps_into_the_border():
    _, t = match_training_frame(Image.new("RGB", (400, 300)))
    # a box entirely in the black bar above the pasted crop -> zero-height after clamp
    x1, y1, x2, y2 = map_bbox_back((0, 0, 5, 3), t)
    assert (y1, y2) == (0, 0) and 0 <= x1 <= x2 <= t.src_w


# --------------------------------------------------------------------------- #
# fullness heuristic
# --------------------------------------------------------------------------- #
def test_brightness_fullness_dark_is_full_bright_is_empty():
    dark = BrightnessFullness().estimate(Image.new("RGB", (60, 80), (15, 15, 15)))
    bright = BrightnessFullness().estimate(Image.new("RGB", (60, 80), (210, 210, 210)))
    assert dark.level == "full" and dark.score > 0.9
    assert bright.level == "empty" and bright.score == 0.0
    assert dark.method == "brightness-heuristic" and dark.detail["uncalibrated"] is True


def test_fullness_none_crop():
    assert BrightnessFullness().estimate(None).score is None
    assert NullFullness().estimate(Image.new("RGB", (4, 4))).method == "none"


# --------------------------------------------------------------------------- #
# run_pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_happy_path_maps_bbox_into_frame():
    model = FakeModel([_Box([120, 90, 260, 300], 0.7)])
    r = run_pipeline(_snapshot(), model=model, normalize=True)

    assert r.errors == []
    assert set(r.timings_ms) == {"transform", "normalize", "detect", "crop", "classify"}
    assert r.normalized is not None and r.normalized.size[0] > r.frame.size[0] - 1
    assert model.calls[0]["imgsz"] == 320  # trained size forwarded

    d = r.detection
    assert d is not None
    assert 0 <= d.x1 < d.x2 <= r.frame.width
    assert 0 <= d.y1 < d.y2 <= r.frame.height
    assert r.crop.size == (d.x2 - d.x1, d.y2 - d.y1)
    assert r.bounded.size == r.frame.size
    assert r.fullness.method == "brightness-heuristic"


def test_pipeline_without_normalize_uses_frame_coords_directly():
    model = FakeModel([_Box([10, 20, 80, 120], 0.9)])
    r = run_pipeline(_snapshot(), model=model, normalize=False)
    assert r.normalized is None
    assert r.detection.bbox == (10, 20, 80, 120)


def test_pipeline_no_detection_degrades_cleanly():
    r = run_pipeline(_snapshot(), model=FakeModel([_Box([0, 0, 10, 10], 0.01)]), conf=0.5)
    assert r.detection is None
    assert r.crop is None
    assert r.bounded.size == r.frame.size
    assert r.fullness.level == "unknown"
    assert r.errors == []


def test_pipeline_records_stage_error_without_raising(monkeypatch):
    def boom(_img):
        raise RuntimeError("bad transform")

    monkeypatch.setattr("coffeecam.pipeline.apply_transform", boom)
    r = run_pipeline(_snapshot(), model=FakeModel([]))
    assert any("transform: bad transform" in e for e in r.errors)
    assert r.detection is None
