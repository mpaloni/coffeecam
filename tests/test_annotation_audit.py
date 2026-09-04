import csv

import numpy as np
import pytest
from PIL import Image

from coffeecam import annotations, annotation_audit
from coffeecam.annotation_audit import (
    audit_boxes,
    build_contact_sheets,
    classify,
    iou,
    summarize,
    write_csv,
)
from coffeecam.summary import detect_on_frame


# --- fake YOLO model (same shape as tests/test_summary.py) ----------------- #
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


def _frame(path, size=(424, 353), color=(90, 90, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


# --- iou ------------------------------------------------------------------- #
def test_iou_identical_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_iou_half_overlap():
    # 10x10 boxes sharing a 5x10 strip -> inter 50, union 150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


# --- classify ------------------------------------------------------------- #
def test_classify_no_detection_is_inconclusive():
    assert classify(0.0, None) == "INCONCLUSIVE"


def test_classify_good_iou_is_ok():
    assert classify(0.8, 0.3) == "OK"


def test_classify_low_iou_confident_detection_is_mismatch():
    assert classify(0.05, 0.6) == "LIKELY_MISMATCH"


def test_classify_low_iou_weak_detection_is_inconclusive():
    assert classify(0.05, 0.1) == "INCONCLUSIVE"


def test_classify_mid_iou_is_inconclusive():
    assert classify(0.35, 0.9) == "INCONCLUSIVE"


# --- audit_boxes -------------------------------------------------------- #
@pytest.fixture
def store_and_captures(tmp_path):
    caps = tmp_path / "captures"
    for rel in ("2026-08-31/090000.jpg", "2026-08-31/090500.jpg", "2026-08-31/091000.jpg"):
        _frame(caps / rel)
    store = caps / "annotations.jsonl"
    return store, caps


def test_audit_ok_when_saved_box_matches_detector(store_and_captures):
    store, caps = store_and_captures
    model = FakeModel(_Box([120, 90, 300, 240], 0.9))
    # what the detector will actually report for this frame, mapped to frame px:
    det, _ = detect_on_frame(Image.open(caps / "2026-08-31/090000.jpg").convert("RGB"), model)
    annotations.upsert("2026-08-31/090000.jpg", [list(det.bbox)], store=store)

    (res,) = audit_boxes(annotations.load(store), caps, model)
    assert res.bucket == "OK"
    assert res.best_iou == pytest.approx(1.0, abs=1e-6)


def test_audit_flags_box_far_from_confident_detection(store_and_captures):
    store, caps = store_and_captures
    model = FakeModel(_Box([120, 90, 300, 240], 0.9))
    annotations.upsert("2026-08-31/090000.jpg", [[1, 1, 20, 20]], store=store)

    (res,) = audit_boxes(annotations.load(store), caps, model)
    assert res.bucket == "LIKELY_MISMATCH"
    assert res.det_conf == pytest.approx(0.9)


def test_audit_inconclusive_when_model_finds_nothing(store_and_captures):
    store, caps = store_and_captures
    model = FakeModel(None)
    annotations.upsert("2026-08-31/090000.jpg", [[100, 80, 260, 220]], store=store)

    (res,) = audit_boxes(annotations.load(store), caps, model)
    assert res.bucket == "INCONCLUSIVE"
    assert res.det_box is None


def test_audit_negative_row_with_confident_pot_is_flagged(store_and_captures):
    store, caps = store_and_captures
    model = FakeModel(_Box([120, 90, 300, 240], 0.9))
    annotations.upsert("2026-08-31/090000.jpg", [], store=store)  # explicit negative

    (res,) = audit_boxes(annotations.load(store), caps, model)
    assert res.kind == "negative"
    assert res.bucket == "NEGATIVE_HAS_POT"


def test_audit_skips_watched_rows_and_missing_files(store_and_captures):
    store, caps = store_and_captures
    model = FakeModel(_Box([120, 90, 300, 240], 0.9))
    annotations.upsert("2026-08-31/090000.jpg", [[100, 80, 260, 220]], store=store)
    annotations.skip("2026-08-31/090500.jpg", store=store)
    annotations.upsert("2026-08-31/gone.jpg", [[10, 10, 40, 40]], store=store)

    results = audit_boxes(annotations.load(store), caps, model)
    assert [r.rel for r in results] == ["2026-08-31/090000.jpg"]


def test_audit_sorts_worst_bucket_first(store_and_captures):
    store, caps = store_and_captures
    model = FakeModel(_Box([120, 90, 300, 240], 0.9))
    det, _ = detect_on_frame(Image.open(caps / "2026-08-31/090000.jpg").convert("RGB"), model)
    annotations.upsert("2026-08-31/090000.jpg", [list(det.bbox)], store=store)  # OK
    annotations.upsert("2026-08-31/090500.jpg", [[1, 1, 12, 12]], store=store)  # mismatch

    results = audit_boxes(annotations.load(store), caps, model)
    assert [r.bucket for r in results] == ["LIKELY_MISMATCH", "OK"]


# --- csv / summary / contact sheets ------------------------------------- #
def test_write_csv_roundtrip(store_and_captures, tmp_path):
    store, caps = store_and_captures
    model = FakeModel(_Box([120, 90, 300, 240], 0.9))
    annotations.upsert("2026-08-31/090000.jpg", [[1, 1, 20, 20]], store=store)
    results = audit_boxes(annotations.load(store), caps, model)

    out = tmp_path / "audit.csv"
    write_csv(results, out)
    rows = list(csv.DictReader(out.open()))
    assert rows[0]["rel"] == "2026-08-31/090000.jpg"
    assert rows[0]["bucket"] == "LIKELY_MISMATCH"
    assert rows[0]["saved_boxes"] == "1,1,20,20"


def test_summarize_counts_buckets(store_and_captures):
    store, caps = store_and_captures
    model = FakeModel(_Box([120, 90, 300, 240], 0.9))
    annotations.upsert("2026-08-31/090000.jpg", [[1, 1, 12, 12]], store=store)
    annotations.upsert("2026-08-31/090500.jpg", [[1, 1, 12, 12]], store=store)
    counts = summarize(audit_boxes(annotations.load(store), caps, model))
    assert counts == {"LIKELY_MISMATCH": 2}


def test_build_contact_sheets_writes_pages(store_and_captures, tmp_path):
    store, caps = store_and_captures
    model = FakeModel(_Box([120, 90, 300, 240], 0.9))
    annotations.upsert("2026-08-31/090000.jpg", [[1, 1, 20, 20]], store=store)
    results = audit_boxes(annotations.load(store), caps, model)

    pages = build_contact_sheets(results, caps, tmp_path / "contact", per_page=20)
    assert len(pages) == 1
    assert pages[0].exists()
    with Image.open(pages[0]) as im:
        assert im.width > 0 and im.height > 0


def test_build_contact_sheets_empty_when_all_ok(store_and_captures, tmp_path):
    store, caps = store_and_captures
    model = FakeModel(_Box([120, 90, 300, 240], 0.9))
    det, _ = detect_on_frame(Image.open(caps / "2026-08-31/090000.jpg").convert("RGB"), model)
    annotations.upsert("2026-08-31/090000.jpg", [list(det.bbox)], store=store)
    results = audit_boxes(annotations.load(store), caps, model)
    assert build_contact_sheets(results, caps, tmp_path / "c") == []
