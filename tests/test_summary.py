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
    collect_split_frames,
    detect_on_frame,
    resolve_frames,
    _render_frame,
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
# frame sets: dataset splits
# --------------------------------------------------------------------------- #
@pytest.fixture
def dataset(tmp_path):
    """A minimal promoted dataset: 3 test frames (2 with labels), 1 train frame."""
    ds = tmp_path / "dataset"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir()
    for name in ("cap_20260831_131814_558", "cap_20260901_090000_001", "cap_20260901_091500"):
        _write_frame(ds / "images" / f"{name}.jpg")
    (ds / "labels" / "cap_20260831_131814_558.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (ds / "labels" / "cap_20260901_090000_001.txt").write_text("0 0.4 0.4 0.1 0.1\n")
    _write_frame(ds / "images" / "cap_20260701_120000_shift_x10_y5.jpg")
    (ds / "test.txt").write_text(
        "./images/cap_20260831_131814_558.jpg\n"
        "./images/cap_20260901_090000_001.jpg\n"
        "./images/cap_20260901_091500.jpg\n"
    )
    (ds / "train.txt").write_text("./images/cap_20260701_120000_shift_x10_y5.jpg\n")
    return ds


def test_collect_split_frames_resolves_images_labels_and_ts(dataset):
    refs = collect_split_frames("test", dataset)
    assert [r.path.name for r in refs] == [
        "cap_20260831_131814_558.jpg",
        "cap_20260901_090000_001.jpg",
        "cap_20260901_091500.jpg",
    ]
    assert refs[0].label_path is not None and refs[0].label_path.name == "cap_20260831_131814_558.txt"
    assert refs[2].label_path is None  # no label file on disk
    assert refs[0].ts_label == "test  2026-08-31 13:18:14"
    assert refs[0].day == "test"


def test_collect_split_frames_missing_listing_is_empty(dataset):
    assert collect_split_frames("val", dataset) == []


def test_resolve_frames_dispatch_and_unknown(captures, dataset):
    assert len(resolve_frames("captures", captures_dir=captures)) == 4
    assert len(resolve_frames("test", dataset_dir=dataset)) == 3
    with pytest.raises(ValueError):
        resolve_frames("bogus")


def test_build_summary_gif_over_a_split(dataset):
    gif, meta = build_summary_gif(frameset="test", dataset_dir=dataset, model=None)
    assert meta["frameset"] == "test"
    assert meta["source_frames"] == 3
    assert _n_frames(gif) == 3


def test_build_summary_gif_split_missing_raises(dataset):
    with pytest.raises(SummaryEmpty):
        build_summary_gif(frameset="val", dataset_dir=dataset, model=None)


def test_gt_boxes_pixel_converts_yolo_label(dataset):
    from coffeecam.summary import _gt_boxes_pixel

    lbl = dataset / "labels" / "cap_20260831_131814_558.txt"  # "0 0.5 0.5 0.2 0.2"
    assert _gt_boxes_pixel(lbl, 424, 353) == [(170, 141, 254, 212)]


def test_build_summary_gif_gt_mode_draws_label_boxes(dataset):
    from coffeecam.summary import _BOX_GT

    _, meta = build_summary_gif(frameset="test", dataset_dir=dataset, model=None, gt=True)
    # 2 of the 3 test frames have a label file on disk
    assert meta["ground_truth_frames"] == 2

    def has_color(im, rgb):
        return any(c == rgb for _n, c in im.getcolors(maxcolors=1 << 24))

    refs = collect_split_frames("test", dataset)
    labelled, _ = _render_frame(refs[0], model=None, conf=0.25, scale=1.0, index=1, n=1, gt=True)
    unlabelled, _ = _render_frame(refs[2], model=None, conf=0.25, scale=1.0, index=1, n=1, gt=True)
    assert has_color(labelled, _BOX_GT)
    assert not has_color(unlabelled, _BOX_GT)


def test_gt_mode_is_a_noop_for_captures(captures):
    _, meta = build_summary_gif(captures_dir=captures, frameset="captures", model=None, gt=True)
    assert meta["ground_truth_frames"] == 0


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


def test_summary_route_set_param_selects_split(client, dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("COFFEECAM_DATASET_DIR", str(dataset))
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(tmp_path / "no-captures"))

    meta = client.get("/summary.json?annotate=0&set=test").get_json()
    assert meta["frameset"] == "test"
    assert meta["source_frames"] == 3
    assert meta["rendered_frames"] == 3

    # unknown set falls back to captures -> 404 here (captures dir is empty)
    assert client.get("/summary.json?annotate=0&set=bogus").status_code == 404

    # the split is part of the cache key: a different frames= rebuilds cleanly
    meta2 = client.get("/summary.json?annotate=0&set=test&frames=2").get_json()
    assert meta2["frameset"] == "test" and meta2["rendered_frames"] == 2
