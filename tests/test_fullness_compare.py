import io

import pytest
from PIL import Image

from coffeecam import annotations, fullness_labels
from coffeecam.fullness_compare import NoTestData, build_test_gif
from coffeecam.fullness_dataset import build


@pytest.fixture
def trained_tree(tmp_path, monkeypatch):
    """A tiny fullness_dataset/ + a stubbed ModelFullness so build_test_gif runs
    without ultralytics."""
    cap = tmp_path / "captures"
    for i, lvl in enumerate(["empty", "some_low", "some_half", "lots_high", "lots_full",
                             "absent"] * 3):
        raw = {"some_low": "low", "some_half": "half", "lots_high": "high",
               "lots_full": "full"}.get(lvl, lvl)
        rel = f"2026-09-0{i % 3 + 1}/{i:06d}.jpg"
        p = cap / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (424, 353), (60 + i, 60, 60)).save(p, "JPEG")
        annotations.upsert(rel, [[291, 113, 373, 205]], store=cap / "annotations.jsonl")
        fullness_labels.upsert(rel, raw, store=cap / "fullness.jsonl")
    ds = tmp_path / "fds"
    build(captures_dir=cap, out_dir=ds, merge="coarse")

    import coffeecam.fullness_compare as fc

    class _R:
        def __init__(s, level, score, probs):
            s.level, s.score, s.detail = level, score, {"probs": probs}

    class _StubModel:
        def __init__(s, w):
            pass

        def estimate(s, crop):
            return _R("some", 0.5, {"absent": 0.1, "empty": 0.2, "some": 0.6, "lots": 0.1})

    monkeypatch.setattr(fc, "ModelFullness", _StubModel)
    monkeypatch.setattr(fc, "resolve_fullness_weights", lambda *a, **k: tmp_path / "w.pt")
    return ds


def test_build_test_gif_returns_gif_and_scoreboard(trained_tree):
    gif, sb = build_test_gif(dataset_dir=trained_tree)
    with Image.open(io.BytesIO(gif)) as im:
        assert im.format == "GIF"
        assert getattr(im, "n_frames", 1) == sb["n"] > 0
    for key in ("fullness_v1", "brightness_heuristic"):
        s = sb[key]
        assert set(s) >= {"balanced_accuracy", "confusion", "has_coffee_recall",
                          "fill_score_spearman", "recall"}
        assert len(s["confusion"]) == 4


def test_no_test_data_raises(tmp_path, monkeypatch):
    import coffeecam.fullness_compare as fc

    monkeypatch.setattr(fc, "resolve_fullness_weights", lambda *a, **k: tmp_path / "w.pt")
    with pytest.raises(NoTestData):
        build_test_gif(dataset_dir=tmp_path / "empty")


def test_no_weights_raises(tmp_path, monkeypatch):
    import coffeecam.fullness_compare as fc

    monkeypatch.setattr(fc, "resolve_fullness_weights", lambda *a, **k: None)
    with pytest.raises(NoTestData):
        build_test_gif(dataset_dir=tmp_path)


def test_server_routes(monkeypatch):
    from coffeecam import server

    gif = b"GIF89a" + b"\0" * 20
    monkeypatch.setattr(
        "coffeecam.fullness_compare.build_test_gif",
        lambda **k: (gif, {"n": 3, "fullness_v1": {}, "brightness_heuristic": {}}),
    )
    c = server.create_app(start_worker=False).test_client()
    r = c.get("/fullness/compare.gif")
    assert r.status_code == 200 and r.mimetype == "image/gif" and r.data == gif
    assert c.get("/fullness/compare.json").get_json()["n"] == 3


def test_server_route_404_without_data(monkeypatch):
    from coffeecam import server
    from coffeecam.fullness_compare import NoTestData

    def _boom(**k):
        raise NoTestData("nope")

    monkeypatch.setattr("coffeecam.fullness_compare.build_test_gif", _boom)
    c = server.create_app(start_worker=False).test_client()
    assert c.get("/fullness/compare.gif").status_code == 404
    assert c.get("/fullness/compare.json").status_code == 404
