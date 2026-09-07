from datetime import datetime

from PIL import Image

from coffeecam import state_history
from coffeecam.detect import Detection
from coffeecam.fullness import FullnessResult
from coffeecam.pipeline import PipelineResult


def _result(level, *, ts, det=True):
    frame = Image.new("RGB", (120, 100), (80, 80, 80))
    return PipelineResult(
        ts=ts,
        frame=frame,
        normalized=None,
        bounded=frame.copy(),
        crop=frame.crop((10, 10, 50, 50)),
        detection=Detection(10, 10, 50, 50, 0.4) if det else None,
        fullness=FullnessResult(level, 0.5, "model:test", {}),
        timings_ms={"detect": 5.0},
        errors=[],
    )


def test_append_and_load_roundtrip(tmp_path):
    ts = datetime(2026, 9, 7, 8, 0, 0)
    state_history.append_row(tmp_path, _result("empty", ts=ts))
    state_history.append_row(tmp_path, _result("low", ts=ts.replace(minute=5), det=False))

    assert state_history.available_dates(tmp_path) == ["2026-09-07"]
    rows = state_history.load_rows(tmp_path)
    assert [r["level"] for r in rows] == ["empty", "low"]
    assert rows[0]["bbox"] == [10, 10, 50, 50]
    assert rows[1]["bbox"] is None
    assert rows[0]["method"] == "model:test"


def test_load_rows_limit_and_missing_date(tmp_path):
    ts = datetime(2026, 9, 7, 8, 0, 0)
    for i in range(4):
        state_history.append_row(tmp_path, _result("full", ts=ts.replace(minute=i)))
    assert len(state_history.load_rows(tmp_path, limit=2)) == 2
    assert state_history.load_rows(tmp_path, date="2000-01-01") == []


def test_rows_split_by_day(tmp_path):
    state_history.append_row(tmp_path, _result("empty", ts=datetime(2026, 9, 7, 23, 59)))
    state_history.append_row(tmp_path, _result("low", ts=datetime(2026, 9, 8, 0, 1)))
    assert state_history.available_dates(tmp_path) == ["2026-09-07", "2026-09-08"]
    assert len(state_history.load_rows(tmp_path, date="2026-09-07")) == 1


def test_runs_from_rows_collapses_and_times():
    rows = [
        {"ts": "2026-09-07T08:00:00", "level": "empty"},
        {"ts": "2026-09-07T08:00:10", "level": "empty"},
        {"ts": "2026-09-07T08:05:10", "level": "full", "artifact": "2026-09-07/080510_frame.jpg"},
        {"ts": "2026-09-07T08:06:10", "level": "full"},
    ]
    runs = state_history.runs_from_rows(rows)
    assert [r["level"] for r in runs] == ["empty", "full"]
    assert runs[0]["frames"] == 2
    assert runs[0]["duration_s"] == 10.0
    assert runs[0]["artifact"] is None
    assert runs[1]["artifact"] == "2026-09-07/080510_frame.jpg"
    assert runs[1]["duration_s"] == 60.0


def test_load_rows_skips_malformed_lines(tmp_path):
    ts = datetime(2026, 9, 7, 8, 0, 0)
    state_history.append_row(tmp_path, _result("empty", ts=ts))
    path = state_history.path_for(tmp_path, ts)
    with path.open("a") as fh:
        fh.write("not json\n")
    assert [r["level"] for r in state_history.load_rows(tmp_path)] == ["empty"]


def test_load_span_recent_and_stride(tmp_path):
    for day, n in [("2026-09-05", 3), ("2026-09-06", 4), ("2026-09-07", 5)]:
        base = datetime.fromisoformat(f"{day}T08:00:00")
        for i in range(n):
            state_history.append_row(tmp_path, _result("full", ts=base.replace(minute=i)))

    assert state_history.recent_dates(tmp_path, 2) == ["2026-09-06", "2026-09-07"]

    dates, rows = state_history.load_span(tmp_path, days=2)
    assert dates == ["2026-09-06", "2026-09-07"]
    assert len(rows) == 9 and all("_date" in r for r in rows)
    assert [r["_date"] for r in rows[:4]] == ["2026-09-06"] * 4

    _, capped = state_history.load_span(tmp_path, days=3, max_points=4)
    assert 4 <= len(capped) <= 6  # strided down, transitions (none here) aside
