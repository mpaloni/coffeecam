import io
import threading

import numpy as np
import pytest
from PIL import Image

from coffeecam import server
from coffeecam.summary import (
    SummaryEmpty,
    build_summary_gif,
    collect_frames,
    detect_on_frame,
    _sample,
)


# --------------------------------------------------------------------------- #
# fake YOLO model (same shape as tests/test_pipeline.py)
# --------------------------------------------------------------------------- #
class _Scalar(float):
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
    """Returns one fixed box (or none) for any input, filtered by conf like YOLO."""

    def __init__(self, box: _Box | None):
        self._box = box

    def predict(self, source=None, conf=0.25, imgsz=320, verbose=False):
        boxes = [self._box] if self._box is not None and float(self._box.conf[0]) >= conf else []
        return [_Result(boxes)]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write_frame(path, color=(90, 90, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (424, 353), color).save(path, "JPEG")


@pytest.fixture
def captures(tmp_path):
    _write_frame(tmp_path / "2026-08-31" / "070008_549.jpg")
    _write_frame(tmp_path / "2026-08-31" / "180042_126.jpg")
    _write_frame(tmp_path / "2026-09-01" / "070510_644.jpg", color=(30, 30, 30))
    _write_frame(tmp_path / "2026-09-01" / "101558_558.jpg", color=(30, 30, 30))
    # must be ignored: harvested pipeline output lives under captures/pipeline/
    _write_frame(tmp_path / "pipeline" / "2026-08-30" / "120000_frame.jpg")
    return tmp_path


# --------------------------------------------------------------------------- #
# collect_frames / _sample
# --------------------------------------------------------------------------- #
def test_collect_frames_orders_chronologically_and_skips_pipeline(captures):
    refs = collect_frames(captures)
    assert [r.path.name for r in refs] == [
        "070008_549.jpg",
        "180042_126.jpg",
        "070510_644.jpg",
        "101558_558.jpg",
    ]
    assert refs[0].ts_label == "2026-08-31 07:00:08"
    assert refs[-1].ts_label == "2026-09-01 10:15:58"


def test_collect_frames_missing_dir_is_empty(tmp_path):
    assert collect_frames(tmp_path / "nope") == []


def test_sample_keeps_endpoints_and_count():
    seq = list(range(100))
    picked = _sample(seq, 10)
    assert len(picked) == 10
    assert picked[0] == 0 and picked[-1] == 99
    assert _sample(seq, 500) == seq
    assert _sample(seq, 1) == [99]


# --------------------------------------------------------------------------- #
# detect_on_frame
# --------------------------------------------------------------------------- #
def test_detect_on_frame_strong_and_weak():
    frame = Image.new("RGB", (424, 353), (90, 90, 90))

    det, strong = detect_on_frame(frame, FakeModel(_Box([120, 90, 260, 300], 0.7)), conf=0.15)
    assert det is not None and strong
    assert 0 <= det.x1 < det.x2 <= frame.width
    assert 0 <= det.y1 < det.y2 <= frame.height

    det, strong = detect_on_frame(frame, FakeModel(_Box([120, 90, 260, 300], 0.05)), conf=0.15)
    assert det is not None and not strong  # box still returned for display

    assert detect_on_frame(frame, FakeModel(None), conf=0.15) == (None, False)


# --------------------------------------------------------------------------- #
# build_summary_gif
# --------------------------------------------------------------------------- #
def _n_frames(gif: bytes) -> int:
    im = Image.open(io.BytesIO(gif))
    assert im.format == "GIF"
    return getattr(im, "n_frames", 1)


def test_build_plain_timelapse_no_model(captures, tmp_path):
    out = tmp_path / "summary.gif"
    gif, meta = build_summary_gif(captures_dir=captures, out=out, model=None)
    assert out.read_bytes() == gif
    assert _n_frames(gif) == 4
    assert meta["annotated"] is False
    assert "detections" not in meta
    assert meta["source_frames"] == 4 and meta["rendered_frames"] == 4
    assert meta["span"] == ["2026-08-31 07:00:08", "2026-09-01 10:15:58"]


def test_build_annotated_counts_strong_boxes(captures):
    gif, meta = build_summary_gif(
        captures_dir=captures, model=FakeModel(_Box([100, 80, 240, 300], 0.9)), conf=0.15
    )
    assert meta["annotated"] is True
    assert meta["detections"] == {"strong": 4, "weak_only": 0, "no_box": 0}
    assert _n_frames(gif) == 4


def test_build_annotated_weak_only_when_below_floor(captures):
    _, meta = build_summary_gif(
        captures_dir=captures, model=FakeModel(_Box([100, 80, 240, 300], 0.06)), conf=0.15
    )
    assert meta["detections"] == {"strong": 0, "weak_only": 4, "no_box": 0}


def test_build_respects_max_frames_and_ms(captures):
    gif, meta = build_summary_gif(captures_dir=captures, model=None, max_frames=2, duration_ms=40)
    assert meta["rendered_frames"] == 2
    assert meta["duration_ms"] == 40
    assert _n_frames(gif) == 2


def test_build_empty_raises(tmp_path):
    with pytest.raises(SummaryEmpty):
        build_summary_gif(captures_dir=tmp_path, model=None)


# --------------------------------------------------------------------------- #
# server routes
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_server_state():
    server._latest = None
    server._latest_jpeg = {}
    server._last_fetch_error = None
    server._worker_started = False
    server._model = None
    server._summary_cache = None
    server._viewer_cache = None
    server._annot_lock = threading.Lock()
    yield


@pytest.fixture
def client():
    return server.create_app(start_worker=False).test_client()


def test_summary_route_serves_gif(client, captures, monkeypatch):
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(captures))

    resp = client.get("/summary?annotate=0")
    assert resp.status_code == 200
    assert resp.mimetype == "image/gif"
    assert _n_frames(resp.data) == 4

    meta = client.get("/summary.json?annotate=0").get_json()
    assert meta["rendered_frames"] == 4
    assert meta["model_ready"] is False


def test_summary_route_cache_and_rebuild(client, captures, monkeypatch):
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(captures))

    client.get("/summary?annotate=0&frames=3")
    assert server._summary_cache is not None
    first_built = server._summary_cache["at"]

    client.get("/summary?annotate=0&frames=3")  # served from cache
    assert server._summary_cache["at"] == first_built

    client.get("/summary?annotate=0&frames=3&rebuild=1")  # forced
    assert server._summary_cache["at"] > first_built


def test_summary_route_404_when_no_captures(client, tmp_path, monkeypatch):
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(tmp_path / "empty"))
    assert client.get("/summary").status_code == 404
    assert client.get("/summary.json").status_code == 404
