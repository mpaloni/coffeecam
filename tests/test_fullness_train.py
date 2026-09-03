"""fullness_train wiring — without actually running ultralytics."""

import sys
import types

import pytest
from PIL import Image

from coffeecam import annotations, fullness_labels, fullness_train


@pytest.fixture
def captures(tmp_path):
    cap = tmp_path / "captures"
    for i, lvl in enumerate(["empty", "low", "half", "high", "full", "absent"] * 3):
        rel = f"2026-09-0{i % 3 + 1}/{i:06d}.jpg"
        p = cap / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (424, 353), (70, 70, 70)).save(p, "JPEG")
        annotations.upsert(rel, [[291, 113, 373, 205]], store=cap / "annotations.jsonl")
        fullness_labels.upsert(rel, lvl, store=cap / "fullness.jsonl")
    return cap


@pytest.fixture
def fake_yolo(monkeypatch):
    """Stub ultralytics.YOLO; record the train() kwargs."""
    calls = {}

    class _Results:
        def __init__(self, save_dir):
            self.save_dir = save_dir

    class _YOLO:
        def __init__(self, weights):
            calls["weights"] = weights

        def train(self, **kw):
            calls["train"] = kw
            run = kw["_run_dir"] = __import__("pathlib").Path(kw["project"]) / kw["name"]
            (run / "weights").mkdir(parents=True, exist_ok=True)
            (run / "weights" / "best.pt").write_bytes(b"stub")
            return _Results(run)

    mod = types.ModuleType("ultralytics")
    mod.YOLO = _YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return calls


def test_builds_tree_then_trains_and_writes_checkpoint(captures, tmp_path, monkeypatch, fake_yolo):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "fds"
    ckpt = tmp_path / "models" / "FULLNESS_CHECKPOINT"
    monkeypatch.setattr(fullness_train, "FULLNESS_CHECKPOINT_FILE", ckpt)

    fullness_train.main([
        "--captures-dir", str(captures), "--data", str(out),
        "--merge", "coarse", "--epochs", "3", "--project", str(tmp_path / "runs"),
        "--name", "ft-test",
    ])

    assert (out / "train").is_dir()
    assert fake_yolo["weights"] == "yolov8n-cls.pt"
    assert fake_yolo["train"]["imgsz"] == 96
    assert fake_yolo["train"]["epochs"] == 3
    assert fake_yolo["train"]["data"] == str(out)
    assert ckpt.read_text().strip() == str(tmp_path / "runs" / "ft-test")


def test_no_build_skips_tree_construction(captures, tmp_path, monkeypatch, fake_yolo):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "fds"
    monkeypatch.setattr(fullness_train, "FULLNESS_CHECKPOINT_FILE",
                        tmp_path / "models" / "FULLNESS_CHECKPOINT")
    fullness_train.main([
        "--no-build", "--data", str(out), "--project", str(tmp_path / "runs"), "--name", "x",
    ])
    assert not out.exists()  # nothing built
    assert fake_yolo["train"]["data"] == str(out)
