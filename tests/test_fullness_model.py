import pytest
from PIL import Image

from coffeecam import fullness as F


@pytest.fixture
def ckpt(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "FULLNESS_CHECKPOINT_FILE", tmp_path / "FULLNESS_CHECKPOINT")
    return tmp_path


# --- resolve_fullness_weights ------------------------------------------------

def test_resolve_none_when_no_pointer(ckpt):
    assert F.resolve_fullness_weights() is None


def test_resolve_none_when_run_dir_missing(ckpt):
    (ckpt / "FULLNESS_CHECKPOINT").write_text("runs/classify/nope\n")
    assert F.resolve_fullness_weights() is None


def test_resolve_appends_weights_best_pt(ckpt):
    run = ckpt / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"x")
    (ckpt / "FULLNESS_CHECKPOINT").write_text(str(run) + "\n")
    got = F.resolve_fullness_weights()
    assert got == run / "weights" / "best.pt"


def test_resolve_explicit_path_checked_for_existence(tmp_path):
    w = tmp_path / "w.pt"
    assert F.resolve_fullness_weights(w) is None
    w.write_bytes(b"x")
    assert F.resolve_fullness_weights(w) == w


def test_default_estimator_falls_back_to_null(ckpt):
    assert isinstance(F.default_estimator(), F.NullFullness)


# --- ModelFullness (stub ultralytics) --------------------------------------

class _Probs:
    def __init__(self, data):
        self.data = _T(data)
        self.top1 = max(range(len(data)), key=lambda i: data[i])


class _T(list):
    def tolist(self):
        return list(self)


class _Res:
    def __init__(self, names, probs):
        self.names = names
        self.probs = _Probs(probs)


class _StubYOLO:
    names = {0: "empty", 1: "some", 2: "lots", 3: "absent"}

    def __init__(self, weights):
        self.weights = weights

    def predict(self, crop, verbose=False):
        return [_Res(self.names, self._probs)]


@pytest.fixture
def stub_yolo(monkeypatch):
    import sys, types

    mod = types.ModuleType("ultralytics")
    mod.YOLO = _StubYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    return _StubYOLO


def test_model_fullness_argmax_and_weighted_score(stub_yolo, tmp_path):
    stub_yolo._probs = [0.1, 0.7, 0.15, 0.05]  # -> "some"
    est = F.ModelFullness(tmp_path / "best.pt")
    r = est.estimate(Image.new("RGB", (96, 96)))
    assert r.level == "some"
    assert r.method == "yolov8n-cls"
    assert r.detail["probs"] == {"empty": 0.1, "some": 0.7, "lots": 0.15, "absent": 0.05}
    # probability-weighted over the on-scale classes (empty0 some.33 lots.83),
    # renormalised by their mass (0.95).
    expect = (0.1 * 0.0 + 0.7 * 0.33 + 0.15 * 0.83) / 0.95
    assert r.score == round(expect, 3)


def test_model_fullness_absent_gives_no_score(stub_yolo, tmp_path):
    stub_yolo._probs = [0.02, 0.03, 0.0, 0.95]  # -> "absent", ~no on-scale mass
    r = F.ModelFullness(tmp_path / "best.pt").estimate(Image.new("RGB", (96, 96)))
    assert r.level == "absent"
    assert r.score is not None  # small mass still on empty/some
    stub_yolo._probs = [0.0, 0.0, 0.0, 1.0]
    r2 = F.ModelFullness(tmp_path / "best.pt").estimate(Image.new("RGB", (96, 96)))
    assert r2.score is None


def test_model_fullness_none_crop(stub_yolo, tmp_path):
    r = F.ModelFullness(tmp_path / "best.pt").estimate(None)
    assert r.level == "unknown" and r.score is None and r.method == "yolov8n-cls"


def test_default_estimator_returns_model_when_weights_resolve(ckpt, stub_yolo):
    run = ckpt / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"x")
    (ckpt / "FULLNESS_CHECKPOINT").write_text(str(run) + "\n")
    assert isinstance(F.default_estimator(), F.ModelFullness)
