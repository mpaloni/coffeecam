import io

import numpy as np
import pytest
from PIL import Image

from coffeecam import server
from coffeecam.summary import SummaryEmpty
from coffeecam.viewer import build_viewer, dump


# --------------------------------------------------------------------------- #
# fake YOLO model (same shape as tests/test_pipeline.py / test_summary.py)
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
    _write_frame(tmp_path / "pipeline" / "2026-08-30" / "120000_frame.jpg")  # must be ignored
    return tmp_path


def _is_jpeg(data: bytes) -> bool:
    im = Image.open(io.BytesIO(data))
    return im.format == "JPEG"


# --------------------------------------------------------------------------- #
# build_viewer
# --------------------------------------------------------------------------- #
def test_build_viewer_frames_and_manifest_line_up(captures):
    build = build_viewer(captures_dir=captures, model=None)

    assert len(build.frames) == 4
    assert len(build.manifest) == 4
    assert all(_is_jpeg(f) for f in build.frames)

    assert [e["i"] for e in build.manifest] == [1, 2, 3, 4]
    assert [e["rel"] for e in build.manifest] == [
        "2026-08-31/070008_549.jpg",
        "2026-08-31/180042_126.jpg",
        "2026-09-01/070510_644.jpg",
        "2026-09-01/101558_558.jpg",
    ]
    assert build.manifest[0]["path"].endswith("2026-08-31/070008_549.jpg")
    assert build.manifest[0]["ts"] == "2026-08-31 07:00:08"

    m = build.meta
    assert m["annotated"] is False
    assert "detections" not in m
    assert m["source_frames"] == 4 and m["rendered_frames"] == 4
    assert m["span"] == ["2026-08-31 07:00:08", "2026-09-01 10:15:58"]
    assert m["bytes"] == sum(len(f) for f in build.frames)


def test_build_viewer_annotates_and_counts_kinds(captures):
    build = build_viewer(
        captures_dir=captures, model=FakeModel(_Box([100, 80, 240, 300], 0.9)), conf=0.15
    )
    assert build.meta["annotated"] is True
    assert build.meta["detections"] == {"strong": 4, "weak_only": 0, "no_box": 0}
    assert [e["kind"] for e in build.manifest] == ["strong"] * 4


def test_build_viewer_weak_only_below_floor(captures):
    build = build_viewer(
        captures_dir=captures, model=FakeModel(_Box([100, 80, 240, 300], 0.06)), conf=0.15
    )
    assert build.meta["detections"] == {"strong": 0, "weak_only": 4, "no_box": 0}
    assert {e["kind"] for e in build.manifest} == {"weak_only"}


def test_build_viewer_respects_max_frames(captures):
    build = build_viewer(captures_dir=captures, model=None, max_frames=2)
    assert build.meta["rendered_frames"] == 2
    assert len(build.frames) == 2
    assert [e["i"] for e in build.manifest] == [1, 2]


def test_frame_lookup_is_one_indexed(captures):
    build = build_viewer(captures_dir=captures, model=None)
    assert build.frame(1) == build.frames[0]
    assert build.frame(4) == build.frames[3]
    for bad in (0, 5, -1):
        with pytest.raises(IndexError):
            build.frame(bad)


def test_build_viewer_empty_raises(tmp_path):
    with pytest.raises(SummaryEmpty):
        build_viewer(captures_dir=tmp_path, model=None)


def test_dump_writes_frames_and_manifest(captures, tmp_path):
    build = build_viewer(captures_dir=captures, model=None)
    out = tmp_path / "viewer_out"
    manifest_path = dump(build, out)

    assert manifest_path == out / "manifest.json"
    assert sorted(p.name for p in out.glob("frame_*.jpg")) == [
        "frame_0001.jpg",
        "frame_0002.jpg",
        "frame_0003.jpg",
        "frame_0004.jpg",
    ]
    import json

    doc = json.loads(manifest_path.read_text())
    assert len(doc["frames"]) == 4
    assert doc["meta"]["rendered_frames"] == 4
    assert (out / "frame_0001.jpg").read_bytes() == build.frames[0]


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
    server._compare_cache = None
    server._compare_models.clear()
    yield


@pytest.fixture
def client():
    return server.create_app(start_worker=False).test_client()


def test_viewer_page_served(client):
    resp = client.get("/viewer")
    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    assert b"coffeecam viewer" in resp.data
    assert b"/viewer/manifest.json" in resp.data


def test_viewer_manifest_and_frames(client, captures, monkeypatch):
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(captures))

    doc = client.get("/viewer/manifest.json?annotate=0").get_json()
    assert [e["i"] for e in doc["frames"]] == [1, 2, 3, 4]
    assert doc["frames"][0]["rel"] == "2026-08-31/070008_549.jpg"
    assert doc["meta"]["rendered_frames"] == 4
    assert doc["meta"]["model_ready"] is False

    r1 = client.get("/viewer/frame/1.jpg?annotate=0")
    assert r1.status_code == 200
    assert r1.mimetype == "image/jpeg"
    assert _is_jpeg(r1.data)

    assert client.get("/viewer/frame/4.jpg?annotate=0").status_code == 200
    assert client.get("/viewer/frame/5.jpg?annotate=0").status_code == 404
    assert client.get("/viewer/frame/0.jpg?annotate=0").status_code == 404


def test_viewer_frame_fetch_never_forces_rebuild(client, captures, monkeypatch):
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(captures))

    client.get("/viewer/manifest.json?annotate=0&frames=3")
    built_at = server._viewer_cache["at"]

    # even with rebuild=1 in the query, a per-frame fetch serves from cache
    client.get("/viewer/frame/2.jpg?annotate=0&frames=3&rebuild=1")
    assert server._viewer_cache["at"] == built_at

    # the manifest route does honour rebuild=1
    client.get("/viewer/manifest.json?annotate=0&frames=3&rebuild=1")
    assert server._viewer_cache["at"] > built_at


def test_viewer_cache_reused_across_calls(client, captures, monkeypatch):
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(captures))

    client.get("/viewer/manifest.json?annotate=0&frames=3")
    first = server._viewer_cache["at"]
    client.get("/viewer/frame/1.jpg?annotate=0&frames=3")
    client.get("/viewer/manifest.json?annotate=0&frames=3")
    assert server._viewer_cache["at"] == first


def test_viewer_404_when_no_captures(client, tmp_path, monkeypatch):
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(tmp_path / "empty"))
    assert client.get("/viewer/manifest.json").status_code == 404
    assert client.get("/viewer/frame/1.jpg").status_code == 404
    assert client.get("/viewer").status_code == 200  # the page itself always renders
