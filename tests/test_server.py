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
    yield


@pytest.fixture
def client():
    return server.create_app(start_worker=False).test_client()


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
