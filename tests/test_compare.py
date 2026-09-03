import argparse
import io
import json
import threading

import numpy as np
import pytest
from PIL import Image

from coffeecam import compare, server
from coffeecam.summary import collect_frames


# fake YOLO model, same shape as tests/test_summary.py
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
    def __init__(self, box):
        self._box = box

    def predict(self, source=None, conf=0.25, imgsz=320, verbose=False):
        boxes = [self._box] if self._box is not None and float(self._box.conf[0]) >= conf else []
        return [_Result(boxes)]


def _write_frame(path, color=(90, 90, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (424, 353), color).save(path, "JPEG")


@pytest.fixture
def refs(tmp_path):
    _write_frame(tmp_path / "2026-08-31" / "070008_549.jpg")
    _write_frame(tmp_path / "2026-08-31" / "180042_126.jpg")
    _write_frame(tmp_path / "2026-09-01" / "101558_558.jpg")
    return collect_frames(tmp_path)


def _n_frames(gif: bytes) -> int:
    im = Image.open(io.BytesIO(gif))
    assert im.format == "GIF"
    return getattr(im, "n_frames", 1)


# --------------------------------------------------------------------------- #
# resolve_model_weights / parse_model_arg
# --------------------------------------------------------------------------- #
def test_resolve_model_weights_pt_vs_run_dir():
    assert compare.resolve_model_weights("models/x.pt").as_posix() == "models/x.pt"
    assert compare.resolve_model_weights("runs/detect/runs/foo").as_posix() == \
        "runs/detect/runs/foo/weights/best.pt"


def test_parse_model_arg(tmp_path):
    w = tmp_path / "run" / "weights" / "best.pt"
    w.parent.mkdir(parents=True)
    w.write_bytes(b"x")
    name, path = compare.parse_model_arg(f"balanced=" + str(tmp_path / "run"))
    assert name == "balanced" and path == w

    with pytest.raises(argparse.ArgumentTypeError):
        compare.parse_model_arg("noequalssign")
    with pytest.raises(argparse.ArgumentTypeError):
        compare.parse_model_arg("missing=" + str(tmp_path / "nope"))


# --------------------------------------------------------------------------- #
# build_comparison_gif
# --------------------------------------------------------------------------- #
def test_comparison_gif_panels_counts_and_size(refs):
    models = {
        "old": FakeModel(_Box([100, 80, 240, 300], 0.05)),   # weak only (below 0.15)
        "new": FakeModel(_Box([100, 80, 240, 300], 0.90)),   # strong
    }
    gif, stats = compare.build_comparison_gif(refs, models, conf=0.15, scale=1.0)

    assert _n_frames(gif) == 3
    assert stats["models"] == ["old", "new"]
    assert stats["counts"]["old"] == {"strong": 0, "weak_only": 3, "no_box": 0}
    assert stats["counts"]["new"] == {"strong": 3, "weak_only": 0, "no_box": 0}

    # stitched width is ~2 panels wide (+ gap); taller than a bare frame (caption bars)
    frame0 = Image.open(io.BytesIO(gif))
    assert frame0.width > 424 * 2
    assert frame0.height > 353


def test_comparison_gif_respects_max_frames_and_scale(refs):
    models = {"m": FakeModel(None)}
    gif, stats = compare.build_comparison_gif(refs, models, max_frames=2, scale=0.5)
    assert stats["rendered_frames"] == 2
    assert stats["counts"]["m"] == {"strong": 0, "weak_only": 0, "no_box": 2}
    assert _n_frames(gif) == 2


def test_comparison_gif_empty_and_no_models(refs):
    with pytest.raises(compare.SummaryEmpty):
        compare.build_comparison_gif([], {"m": FakeModel(None)})
    with pytest.raises(ValueError):
        compare.build_comparison_gif(refs, {})


def test_fmt_table_with_and_without_scores():
    stats = {"models": ["a", "b"], "counts": {
        "a": {"strong": 1, "weak_only": 2, "no_box": 0},
        "b": {"strong": 3, "weak_only": 0, "no_box": 0},
    }}
    plain = compare._fmt_table(stats, None)
    assert "mAP50" not in plain and "a" in plain and "b" in plain

    scored = compare._fmt_table(stats, {
        "a": {"map50": 0.216, "map50_95": 0.065, "precision": 0.85, "recall": 0.21},
        "b": {"map50": 0.573, "map50_95": 0.479, "precision": 0.73, "recall": 0.64},
    })
    assert "mAP50" in scored and "0.573" in scored


# --------------------------------------------------------------------------- #
# /compare server route
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _reset_server_state():
    server._model = None
    server._summary_cache = None
    server._viewer_cache = None
    server._compare_cache = None
    server._compare_models.clear()
    yield


@pytest.fixture
def client():
    return server.create_app(start_worker=False).test_client()


@pytest.fixture
def compare_env(tmp_path, monkeypatch):
    for rel in ("2026-08-31/070008_549.jpg", "2026-08-31/180042_126.jpg",
                "2026-09-01/101558_558.jpg"):
        _write_frame(tmp_path / rel)
    monkeypatch.setenv("COFFEECAM_CAPTURES_DIR", str(tmp_path))
    # stub model loading: spec string -> a FakeModel, no real YOLO/checkpoint
    weak = FakeModel(_Box([100, 80, 240, 300], 0.05))
    fakes = {
        "old": weak,                                          # always weak
        "runs/detect/runs/train-nomosaic-2": weak,            # default baseline spec
        "": FakeModel(_Box([100, 80, 240, 300], 0.90)),       # live checkpoint, strong
    }
    monkeypatch.setattr(server, "_compare_model_for",
                        lambda spec: fakes.get(spec) or (_ for _ in ()).throw(FileNotFoundError(spec)))
    return tmp_path


def test_compare_route_default_pair_serves_gif(client, compare_env):
    resp = client.get("/compare?frames=3&conf=0.15")
    assert resp.status_code == 200 and resp.mimetype == "image/gif"
    assert _n_frames(resp.data) == 3

    meta = client.get("/compare.json?frames=3&conf=0.15").get_json()
    assert meta["models"] == ["nomosaic-2", "checkpoint"]
    assert meta["counts"]["nomosaic-2"] == {"strong": 0, "weak_only": 3, "no_box": 0}
    assert meta["counts"]["checkpoint"] == {"strong": 3, "weak_only": 0, "no_box": 0}
    assert meta["specs"]["checkpoint"] == "models/CHECKPOINT"


def test_compare_route_explicit_models_and_cache(client, compare_env):
    client.get("/compare?model=a=old&model=b=old&frames=2")
    assert server._compare_cache is not None
    built = server._compare_cache["at"]
    client.get("/compare?model=a=old&model=b=old&frames=2")   # cached
    assert server._compare_cache["at"] == built
    client.get("/compare?model=a=old&model=b=old&frames=2&rebuild=1")
    assert server._compare_cache["at"] > built


def test_compare_route_bad_spec_is_400(client, compare_env):
    assert client.get("/compare?model=x=nosuchrun").status_code == 400
    assert client.get("/compare.json?model=x=nosuchrun").status_code == 400
