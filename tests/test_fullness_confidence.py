from datetime import datetime

from PIL import Image

from coffeecam import fullness_confidence, state_history
from coffeecam.detect import Detection
from coffeecam.fullness import FullnessResult
from coffeecam.pipeline import PipelineResult


def _result(level, p, ts, *, det=True):
    frame = Image.new("RGB", (120, 100), (80, 80, 80))
    probs = None if p is None else {level: p, "_other": round(1 - p, 4)}
    return PipelineResult(
        ts=ts, frame=frame, normalized=None, bounded=frame.copy(),
        crop=frame.crop((10, 10, 50, 50)),
        detection=Detection(10, 10, 50, 50, 0.8) if det else None,
        fullness=FullnessResult(level, 0.5, "yolov8n-cls", {} if probs is None else {"probs": probs}),
        timings_ms={}, errors=[],
    )


def test_row_carries_top1_prob_and_frame(tmp_path):
    ts = datetime(2026, 9, 3, 8, 0, 0)
    row = state_history.append_row(
        tmp_path, _result("lots", 0.91, ts), frame_rel="2026-09-03/080000_000.jpg"
    )
    assert row["p"] == 0.91
    assert row["frame"] == "2026-09-03/080000_000.jpg"
    # NullFullness-style result (no probs) -> p is None
    row2 = state_history.append_row(tmp_path, _result("empty", None, ts.replace(minute=1)))
    assert row2["p"] is None
    assert "frame" not in row2


def test_summarise_reports_distribution_and_worst(tmp_path):
    ts = datetime(2026, 9, 3, 8, 0, 0)
    # p is the top-1 (argmax) probability, so keep it >= 0.5
    for i, (lvl, p) in enumerate(
        [("empty", 0.99), ("empty", 0.97), ("lots", 0.55), ("some", 0.52), ("lots", 0.93)]
    ):
        state_history.append_row(
            tmp_path, _result(lvl, p, ts.replace(minute=i), det=(i != 3)),
            frame_rel=f"2026-09-03/08{i:02d}00_000.jpg",
        )
    rows = fullness_confidence._load(tmp_path, ["2026-09-03"])
    text = fullness_confidence.summarise(rows, low=0.6, limit=3)
    assert "with classifier p 5" in text
    assert "below 0.60: 2" in text
    # worst frame (p=0.41) listed first, and its detector miss shown as "-"
    worst = text.split("lowest-confidence frames")[1].splitlines()
    assert "0.520" in worst[2] and "080300" in worst[2]
    assert "1 misses" in text


def test_load_across_days(tmp_path):
    for day, minute in [("2026-09-03", 0), ("2026-09-04", 0)]:
        state_history.append_row(
            tmp_path, _result("empty", 0.9, datetime.fromisoformat(f"{day}T08:0{minute}:00")),
            frame_rel=f"{day}/080000_000.jpg",
        )
    assert len(fullness_confidence._load(tmp_path, None)) == 2
    assert len(fullness_confidence._load(tmp_path, ["2026-09-03"])) == 1
