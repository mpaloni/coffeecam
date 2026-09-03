import threading
from datetime import datetime

import pytest
from PIL import Image, ImageDraw

from coffeecam import server
from coffeecam.detect import Detection
from coffeecam.fullness import FullnessResult
from coffeecam.pipeline import PipelineResult


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
    server._fullness_lock = threading.Lock()
    yield


@pytest.fixture
def client():
    return server.create_app(start_worker=False).test_client()


@pytest.fixture
def captures_env(tmp_path, monkeypatch):
    """A captures/ dir with two day-folders, wired to COFFEECAM_CAPTURES_DIR."""
    for rel in ("2026-08-31/070008_549.jpg", "2026-08-31/180042_126.jpg",
                "2026-09-01/101558_558.jpg"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (424, 353), (90, 90, 90)).save(p, "JPEG")
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(tmp_path))
    return tmp_path


def _result(*, with_crop=True):
    frame = Image.new("RGB", (200, 160), (100, 100, 100))
    bounded = frame.copy()
    ImageDraw.Draw(bounded).rectangle((20, 30, 90, 120), outline="red", width=3)
    crop = frame.crop((20, 30, 90, 120)) if with_crop else None
    return PipelineResult(
        ts=datetime.now(),
        frame=frame,
        normalized=Image.new("RGB", (200, 178), (0, 0, 0)),
        bounded=bounded,
        crop=crop,
        detection=Detection(20, 30, 90, 120, 0.42) if with_crop else None,
        fullness=FullnessResult("half", 0.5, "brightness-heuristic", {"uncalibrated": True}),
        timings_ms={"detect": 12.3},
        errors=[],
    )


def test_routes_503_before_first_result(client):
    assert client.get("/healthz").status_code == 503
    assert client.get("/pipeline.json").status_code == 503
    assert client.get("/fullness.json").status_code == 503
    assert client.get("/frame.jpg").status_code == 404
    assert client.get("/").status_code == 503


def test_routes_serve_stored_result(client):
    server._store(_result())

    for path in ("/frame.jpg", "/normalized.jpg", "/bounded.jpg", "/crop.jpg"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert resp.mimetype == "image/jpeg"
        assert len(resp.data) > 100

    pj = client.get("/pipeline.json").get_json()
    assert pj["detection"]["bbox"] == [20, 30, 90, 120]
    assert pj["detection"]["confidence"] == 0.42
    assert pj["fullness"]["level"] == "half"
    assert pj["images"]["crop"] == "/crop.jpg"
    assert pj["timings_ms"] == {"detect": 12.3}

    assert client.get("/fullness.json").get_json()["level"] == "half"
    assert client.get("/healthz").status_code == 200

    page = client.get("/")
    assert page.status_code == 200
    assert b"fullness" in page.data and b"half" in page.data


def test_crop_absent_when_no_detection(client):
    server._store(_result(with_crop=False))
    assert client.get("/crop.jpg").status_code == 404
    assert client.get("/pipeline.json").get_json()["images"]["crop"] is None
    assert client.get("/pipeline.json").get_json()["detection"] is None
    assert client.get("/healthz").status_code == 200


def test_healthz_degraded_on_fetch_error(client):
    server._store(_result())
    server._last_fetch_error = "connection refused"
    resp = client.get("/healthz")
    assert resp.status_code == 503
    assert resp.get_json()["status"] == "degraded"


# --- /annotate labeling endpoint -----------------------------------------

import numpy as np  # noqa: E402


class _Scalar(float):
    def __getitem__(self, _i):
        return self


class _Box:
    def __init__(self, xyxy, conf):
        self.xyxy = np.array([xyxy], dtype=float)
        self.conf = _Scalar(conf)


class _FakeModel:
    def __init__(self, box):
        self._box = box

    def predict(self, source=None, conf=0.25, imgsz=320, verbose=False):
        class _R:
            pass

        r = _R()
        r.boxes = [self._box] if self._box and float(self._box.conf[0]) >= conf else []
        return [r]


def test_annotate_page_is_html(client):
    resp = client.get("/annotate")
    assert resp.status_code == 200
    assert resp.mimetype == "text/html"
    body = resp.get_data(as_text=True)
    assert "<canvas id=cv>" in body
    assert "/annotate/queue.json" in body  # the page self-wires to the API
    assert "/static/" not in body  # fully inline, no external asset


def test_queue_empty_when_no_captures_dir(client, tmp_path, monkeypatch):
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(tmp_path / "nope"))
    resp = client.get("/annotate/queue.json")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["frames"] == []
    assert body["counts"] == {"total": 0, "labeled": 0, "watched": 0, "remaining": 0}
    assert client.get("/annotate/frame/0.jpg").status_code == 404


def test_queue_lists_unlabeled_then_drops_labeled(client, captures_env):
    body = client.get("/annotate/queue.json").get_json()
    assert body["counts"] == {"total": 3, "labeled": 0, "watched": 0, "remaining": 3}
    assert [f["rel"] for f in body["frames"]] == [
        "2026-08-31/070008_549.jpg",
        "2026-08-31/180042_126.jpg",
        "2026-09-01/101558_558.jpg",
    ]

    r = client.post("/annotate/label", json={"rel": "2026-08-31/180042_126.jpg",
                                             "boxes": [[10, 20, 100, 120]]})
    assert r.status_code == 200
    saved = r.get_json()
    assert saved["boxes"] == [[10, 20, 100, 120]]
    assert saved["labeled_at"]

    body = client.get("/annotate/queue.json").get_json()
    assert body["counts"] == {"total": 3, "labeled": 1, "watched": 0, "remaining": 2}
    assert "2026-08-31/180042_126.jpg" not in [f["rel"] for f in body["frames"]]

    labeled = client.get("/annotate/queue.json?filter=labeled").get_json()
    assert [f["rel"] for f in labeled["frames"]] == ["2026-08-31/180042_126.jpg"]
    assert labeled["frames"][0]["boxes"] == [[10, 20, 100, 120]]

    assert len(client.get("/annotate/queue.json?filter=all").get_json()["frames"]) == 3


def test_queue_stride_and_start(client, captures_env):
    strided = client.get("/annotate/queue.json?stride=2").get_json()["frames"]
    assert [f["rel"] for f in strided] == [
        "2026-08-31/070008_549.jpg",
        "2026-09-01/101558_558.jpg",
    ]
    started = client.get("/annotate/queue.json?start=2026-09-01").get_json()
    assert [f["rel"] for f in started["frames"]] == ["2026-09-01/101558_558.jpg"]
    assert started["counts"]["total"] == 1


def test_annotate_frame_serves_raw_bytes(client, captures_env):
    raw = (captures_env / "2026-08-31/070008_549.jpg").read_bytes()
    resp = client.get("/annotate/frame/0.jpg")
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert resp.data == raw
    assert client.get("/annotate/frame/9.jpg").status_code == 404


def test_label_negative_round_trips(client, captures_env):
    r = client.post("/annotate/label", json={"rel": "2026-09-01/101558_558.jpg", "boxes": []})
    assert r.status_code == 200
    assert r.get_json()["boxes"] == []
    labeled = client.get("/annotate/queue.json?filter=labeled").get_json()["frames"]
    assert labeled[0]["rel"] == "2026-09-01/101558_558.jpg"


def test_label_rejects_bad_box(client, captures_env):
    r = client.post("/annotate/label", json={"rel": "2026-08-31/070008_549.jpg",
                                             "boxes": [[100, 100, 50, 50]]})
    assert r.status_code == 400
    r = client.post("/annotate/label", json={"boxes": []})
    assert r.status_code == 400
    # box outside the 424x353 frame
    r = client.post("/annotate/label", json={"rel": "2026-08-31/070008_549.jpg",
                                             "boxes": [[0, 0, 999, 10]]})
    assert r.status_code == 400


def test_label_delete_puts_frame_back_in_queue(client, captures_env):
    client.post("/annotate/label", json={"rel": "2026-08-31/070008_549.jpg",
                                         "boxes": [[1, 2, 3, 4]]})
    r = client.post("/annotate/label/delete", json={"rel": "2026-08-31/070008_549.jpg"})
    assert r.get_json() == {"removed": True}
    r = client.post("/annotate/label/delete", json={"rel": "2026-08-31/070008_549.jpg"})
    assert r.get_json() == {"removed": False}
    rels = [f["rel"] for f in client.get("/annotate/queue.json").get_json()["frames"]]
    assert "2026-08-31/070008_549.jpg" in rels


def test_suggest_503_without_model_then_returns_box(client, captures_env):
    assert client.get("/annotate/suggest/0.json").status_code == 503

    server._model = _FakeModel(_Box([100, 80, 240, 300], 0.9))
    body = client.get("/annotate/suggest/0.json").get_json()
    assert body["source"] == "model"
    assert len(body["boxes"]) == 1 and len(body["boxes"][0]) == 4
    assert body["conf"] == pytest.approx(0.9, abs=0.05)

    server._model = _FakeModel(None)
    assert client.get("/annotate/suggest/0.json").get_json()["source"] == "none"


def test_skip_drops_frame_from_unlabeled_and_into_watched(client, captures_env):
    r = client.post("/annotate/skip", json={"rel": "2026-08-31/070008_549.jpg"})
    assert r.status_code == 200 and r.get_json()["skip"] is True

    body = client.get("/annotate/queue.json").get_json()
    assert body["counts"] == {"total": 3, "labeled": 0, "watched": 1, "remaining": 2}
    assert "2026-08-31/070008_549.jpg" not in [f["rel"] for f in body["frames"]]

    watched = client.get("/annotate/queue.json?filter=watched").get_json()
    assert [f["rel"] for f in watched["frames"]] == ["2026-08-31/070008_549.jpg"]
    assert watched["frames"][0]["skip"] is True

    # un-skip via the existing delete route
    client.post("/annotate/label/delete", json={"rel": "2026-08-31/070008_549.jpg"})
    assert client.get("/annotate/queue.json").get_json()["counts"]["remaining"] == 3


def test_skip_missing_rel_is_400(client, captures_env):
    assert client.post("/annotate/skip", json={}).status_code == 400


def test_skip_queue_marks_all_unlabeled(client, captures_env):
    client.post("/annotate/label", json={"rel": "2026-08-31/180042_126.jpg",
                                         "boxes": [[10, 20, 100, 120]]})
    r = client.post("/annotate/skip-queue")
    assert r.status_code == 200 and r.get_json()["skipped"] == 2  # the other two

    body = client.get("/annotate/queue.json").get_json()
    assert body["counts"] == {"total": 3, "labeled": 1, "watched": 2, "remaining": 0}
    assert body["frames"] == []
    # honours ?start= — nothing older than the cutoff is touched a second time
    assert client.post("/annotate/skip-queue").get_json()["skipped"] == 0


def test_skip_queue_respects_start(client, captures_env):
    r = client.post("/annotate/skip-queue?start=2026-09-01")
    assert r.get_json()["skipped"] == 1  # only the 2026-09-01 frame
    body = client.get("/annotate/queue.json").get_json()
    assert body["counts"]["watched"] == 1 and body["counts"]["remaining"] == 2


def test_promote_route_needs_confirm(client, captures_env, tmp_path, monkeypatch):
    monkeypatch.setenv("COFFEECAM_DATASET_DIR", str(tmp_path / "ds"))
    assert client.post("/annotate/promote").status_code == 400
    client.post("/annotate/label", json={"rel": "2026-08-31/070008_549.jpg",
                                         "boxes": [[10, 10, 80, 80]]})
    r = client.post("/annotate/promote?confirm=1")
    assert r.status_code == 200
    body = r.get_json()
    assert body["train"] + body["val"] + body["test"] >= 1
    assert "train" in body["summary"]


# --- /fullness fill-level labeling endpoint -----------------------------

@pytest.fixture
def fullness_env(captures_env):
    """captures_env + an annotations.jsonl giving two of the three frames a box."""
    from coffeecam import annotations

    store = captures_env / "annotations.jsonl"
    annotations.upsert("2026-08-31/070008_549.jpg", [[291, 113, 373, 205]], store=store)
    annotations.upsert("2026-09-01/101558_558.jpg", [[280, 110, 360, 200]], store=store)
    annotations.upsert("2026-08-31/180042_126.jpg", [], store=store)  # negative — excluded
    return captures_env


def test_fullness_page_is_html(client):
    resp = client.get("/fullness")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["Content-Type"]
    assert "/fullness/queue.json" in resp.get_data(as_text=True)


def test_fullness_queue_only_box_positive_frames(client, fullness_env):
    body = client.get("/fullness/queue.json").get_json()
    rels = [f["rel"] for f in body["frames"]]
    assert rels == ["2026-08-31/070008_549.jpg", "2026-09-01/101558_558.jpg"]
    assert body["counts"]["total"] == 2
    assert body["counts"]["labeled"] == 0
    assert body["counts"]["empty"] == 0


def test_fullness_label_moves_frame_out_of_unlabeled(client, fullness_env):
    r = client.post("/fullness/label", json={"rel": "2026-08-31/070008_549.jpg", "level": "full"})
    assert r.status_code == 200 and r.get_json()["level"] == "full"

    body = client.get("/fullness/queue.json").get_json()
    assert [f["rel"] for f in body["frames"]] == ["2026-09-01/101558_558.jpg"]
    assert body["counts"]["labeled"] == 1
    assert body["counts"]["full"] == 1

    labeled = client.get("/fullness/queue.json?filter=labeled").get_json()
    assert labeled["frames"][0]["level"] == "full"


def test_fullness_label_rejects_bad_level(client, fullness_env):
    r = client.post("/fullness/label", json={"rel": "2026-08-31/070008_549.jpg", "level": "brimming"})
    assert r.status_code == 400


def test_fullness_crop_and_frame_jpeg(client, fullness_env):
    crop = client.get("/fullness/crop/0.jpg")
    assert crop.status_code == 200 and crop.headers["Content-Type"] == "image/jpeg"
    frame = client.get("/fullness/frame/0.jpg")
    assert frame.status_code == 200 and frame.headers["Content-Type"] == "image/jpeg"
    assert len(crop.get_data()) < len(frame.get_data())  # 96px crop is smaller
    assert client.get("/fullness/crop/9.jpg").status_code == 404


def test_fullness_skip_and_delete(client, fullness_env):
    assert client.post("/fullness/skip", json={"rel": "2026-08-31/070008_549.jpg"}).status_code == 200
    body = client.get("/fullness/queue.json").get_json()
    assert body["counts"]["watched"] == 1
    assert [f["rel"] for f in body["frames"]] == ["2026-09-01/101558_558.jpg"]

    assert client.post("/fullness/label/delete",
                       json={"rel": "2026-08-31/070008_549.jpg"}).get_json()["removed"] is True
    body = client.get("/fullness/queue.json").get_json()
    assert body["counts"]["watched"] == 0 and body["counts"]["total"] == 2


def test_fullness_skip_queue(client, fullness_env):
    client.post("/fullness/label", json={"rel": "2026-08-31/070008_549.jpg", "level": "empty"})
    r = client.post("/fullness/skip-queue")
    assert r.get_json()["skipped"] == 1  # the one remaining box-positive frame
    body = client.get("/fullness/queue.json").get_json()
    assert body["frames"] == []
    assert body["counts"] == {"total": 2, "labeled": 1, "watched": 1, "remaining": 0,
                              "empty": 1, "low": 0, "half": 0, "high": 0, "full": 0}
