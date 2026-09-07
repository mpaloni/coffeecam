import json
from datetime import datetime

import pytest
from PIL import Image

from coffeecam import backfill_history, state_history
from coffeecam.fullness import FullnessResult


class _StepEstimator:
    """empty -> lots -> empty over successive frames, so the backfill produces
    exactly two transitions."""

    def __init__(self):
        self._seq = ["empty", "empty", "lots", "lots", "empty"]
        self._i = 0

    def estimate(self, _crop):
        level = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        score = 1.0 if level == "lots" else 0.0
        return FullnessResult(level, score, "stub", {})


@pytest.fixture
def day_dir(tmp_path):
    day = "2026-09-07"
    d = tmp_path / day
    d.mkdir()
    rows = []
    for i, hhmmss in enumerate(("040001", "041501", "043001", "044501", "060001")):
        fn = f"{hhmmss}_000.jpg"
        Image.new("RGB", (424, 353), (80, 80, 80)).save(d / fn, "JPEG")
        rows.append({"t": f"{day}T{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:]}",
                     "kept": True, "file": f"{day}/{fn}"})
    rows.insert(2, {"t": f"{day}T04:20:00", "kept": False})  # a dropped frame
    (d / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path, day


def test_kept_frames_reads_index_in_order(day_dir):
    captures_dir, day = day_dir
    frames = backfill_history._kept_frames(captures_dir, day)
    assert [ts.strftime("%H%M%S") for _, ts in frames] == \
        ["040001", "041501", "043001", "044501", "060001"]


def test_backfill_writes_rows_and_transition_artifacts(day_dir, monkeypatch):
    captures_dir, day = day_dir
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(captures_dir))

    out = backfill_history.backfill(
        day, captures_dir=captures_dir, model=object(), estimator=_StepEstimator()
    )
    assert out["rows"] == 5
    assert out["transitions"] == 2
    assert out["levels"] == {"empty": 3, "lots": 2}

    runs = state_history.runs_from_rows(state_history.load_rows(captures_dir, date=day))
    assert [r["level"] for r in runs] == ["empty", "lots", "empty"]
    assert runs[0]["artifact"] is None
    # each transition dropped a frame artifact, resolvable via the pipeline dir
    for r in runs[1:]:
        assert (captures_dir / "pipeline" / r["artifact"]).is_file()


def test_backfill_refuses_existing_log_without_force(day_dir, monkeypatch):
    captures_dir, day = day_dir
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(captures_dir))
    kw = dict(captures_dir=captures_dir, model=object(), estimator=_StepEstimator())
    backfill_history.backfill(day, **kw)
    with pytest.raises(SystemExit):
        backfill_history.backfill(day, **kw)
    # --force clears and rebuilds
    out = backfill_history.backfill(day, force=True, **kw)
    assert out["rows"] == 5
