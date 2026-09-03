import pytest
from PIL import Image

from coffeecam import annotations
from coffeecam.dataset import (
    bbox_pixel_to_yolo,
    bbox_yolo_to_pixel,
    clamp_bbox,
    dest_name_for,
    promote,
    read_label,
    write_label,
)


def test_clamp_bbox_within_bounds():
    assert clamp_bbox(10, 20, 50, 60, 100, 100) == (10, 20, 50, 60)


def test_clamp_bbox_outside_bounds():
    assert clamp_bbox(-10, -5, 150, 200, 100, 100) == (0, 0, 100, 100)


def test_clamp_bbox_swapped_coords():
    assert clamp_bbox(50, 60, 10, 20, 100, 100) == (10, 20, 50, 60)


def test_bbox_pixel_to_yolo_and_back_roundtrip():
    img_w, img_h = 400, 300
    x1, y1, x2, y2 = 100, 50, 300, 250

    cx, cy, w, h = bbox_pixel_to_yolo(x1, y1, x2, y2, img_w, img_h)
    assert 0 <= cx <= 1 and 0 <= cy <= 1
    assert 0 <= w <= 1 and 0 <= h <= 1

    assert bbox_yolo_to_pixel(cx, cy, w, h, img_w, img_h) == (x1, y1, x2, y2)


def test_write_and_read_label_roundtrip(tmp_path):
    label_path = tmp_path / "labels" / "foo.txt"
    write_label(label_path, 0, 0.5, 0.5, 0.2, 0.4)

    assert read_label(label_path) == [(0, 0.5, 0.5, 0.2, 0.4)]


# --- promote ---------------------------------------------------------------

def _capture(captures_dir, rel, size=(424, 353)):
    path = captures_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "gray").save(path, format="JPEG")
    return path


@pytest.fixture
def promoted(tmp_path):
    captures = tmp_path / "captures"
    dataset = tmp_path / "dataset"
    store = captures / "annotations.jsonl"

    _capture(captures, "2026-08-31/161913_556.jpg")
    _capture(captures, "2026-08-31/155905_054.jpg")
    _capture(captures, "2026-09-01/090000_000.jpg")
    annotations.upsert("2026-08-31/161913_556.jpg", [[171, 88, 249, 206]], store=store)
    annotations.upsert("2026-08-31/155905_054.jpg", [], note="carafe removed", store=store)
    annotations.upsert("2026-09-01/090000_000.jpg", [[10, 10, 60, 60], [100, 100, 200, 200]],
                       store=store)
    # a labeled row whose image no longer exists on disk
    annotations.upsert("2026-08-30/000000_000.jpg", [[1, 2, 3, 4]], store=store)

    return dict(captures=captures, dataset=dataset, store=store)


def test_dest_name_for():
    assert dest_name_for("2026-08-31/161913_556.jpg") == "cap_20260831_161913_556.jpg"
    assert dest_name_for("2026-08-31/161913.jpg") == "cap_20260831_161913.jpg"


def test_promote_builds_images_labels_and_manifests(promoted):
    summary = promote(store=promoted["store"], captures_dir=promoted["captures"],
                      dataset_dir=promoted["dataset"], val_frac=0.0, test_frac=0.0)

    ds = promoted["dataset"]
    assert summary.skipped_missing == 1
    assert summary.negatives == 1
    for name in ("train.txt", "val.txt", "test.txt"):
        assert (ds / name).exists()

    # positive multi-box row -> 2 label lines, image copied
    lbl = ds / "labels" / "cap_20260901_090000_000.txt"
    assert len(read_label(lbl)) == 2
    assert (ds / "images" / "cap_20260901_090000_000.jpg").exists()

    # label content matches the conversion helper
    cx, cy, w, h = bbox_pixel_to_yolo(171, 88, 249, 206, 424, 353)
    got = read_label(ds / "labels" / "cap_20260831_161913_556.txt")[0]
    assert got == pytest.approx((0, cx, cy, w, h), abs=1e-6)


def test_promote_negative_is_empty_label_but_image_present(promoted):
    promote(store=promoted["store"], captures_dir=promoted["captures"],
            dataset_dir=promoted["dataset"], val_frac=0.0, test_frac=0.0)
    ds = promoted["dataset"]
    neg = ds / "labels" / "cap_20260831_155905_054.txt"
    assert neg.read_text() == ""
    assert (ds / "images" / "cap_20260831_155905_054.jpg").exists()


def test_promote_drops_negatives_when_disabled(promoted):
    summary = promote(store=promoted["store"], captures_dir=promoted["captures"],
                      dataset_dir=promoted["dataset"], negatives=False,
                      val_frac=0.0, test_frac=0.0)
    assert summary.negatives == 0
    assert not (promoted["dataset"] / "labels" / "cap_20260831_155905_054.txt").exists()


def test_promote_keeps_synthetic_in_train_only(promoted):
    ds = promoted["dataset"]
    (ds / "images").mkdir(parents=True)
    Image.new("RGB", (512, 456), "gray").save(ds / "images" / "kahvi.png")
    Image.new("RGB", (512, 456), "gray").save(ds / "images" / "kahvi_shift_x10_y10.png")

    promote(store=promoted["store"], captures_dir=promoted["captures"],
            dataset_dir=ds, val_frac=1.0, test_frac=0.0)

    train = (ds / "train.txt").read_text().splitlines()
    val = (ds / "val.txt").read_text().splitlines()
    assert "./images/kahvi.png" in train
    assert "./images/kahvi_shift_x10_y10.png" in train
    assert not any("kahvi" in line for line in val)
    assert any("cap_" in line for line in val)  # real frames went to val
    assert not any("cap_" in line for line in train)


def test_promote_drop_kahvi_aug_keeps_bare_kahvi_and_real_augs(promoted):
    ds = promoted["dataset"]
    (ds / "images").mkdir(parents=True)
    for n in ("kahvi.png", "kahvi_shift_x10_y10.png", "cap_20260101_120000_shift_x5_y5.png"):
        Image.new("RGB", (512, 456), "gray").save(ds / "images" / n)

    promote(store=promoted["store"], captures_dir=promoted["captures"],
            dataset_dir=ds, val_frac=1.0, test_frac=0.0, kahvi_aug=False)

    train = (ds / "train.txt").read_text().splitlines()
    assert "./images/kahvi.png" in train                       # seed frame kept
    assert "./images/kahvi_shift_x10_y10.png" not in train     # kahvi aug dropped
    assert "./images/cap_20260101_120000_shift_x5_y5.png" in train  # real-frame aug kept


def test_promote_drop_kahvi_removes_all_kahvi(promoted):
    ds = promoted["dataset"]
    (ds / "images").mkdir(parents=True)
    for n in ("kahvi.png", "kahvi_shift_x10_y10.png", "cap_20260101_120000_shift_x5_y5.png"):
        Image.new("RGB", (512, 456), "gray").save(ds / "images" / n)

    promote(store=promoted["store"], captures_dir=promoted["captures"],
            dataset_dir=ds, val_frac=1.0, test_frac=0.0, drop_kahvi=True)

    train = (ds / "train.txt").read_text().splitlines()
    assert not any("kahvi" in line for line in train)             # bare + augs gone
    assert "./images/cap_20260101_120000_shift_x5_y5.png" in train  # real-frame aug kept


def test_promote_is_idempotent(promoted):
    kw = dict(store=promoted["store"], captures_dir=promoted["captures"],
              dataset_dir=promoted["dataset"], val_frac=0.0, test_frac=0.0)
    promote(**kw)
    ds = promoted["dataset"]
    first = {n: (ds / n).read_text() for n in ("train.txt", "val.txt", "test.txt")}
    n_images = len(list((ds / "images").iterdir()))

    promote(**kw)
    second = {n: (ds / n).read_text() for n in ("train.txt", "val.txt", "test.txt")}
    assert first == second
    assert len(list((ds / "images").iterdir())) == n_images


def test_promote_dry_run_writes_nothing(promoted):
    summary = promote(store=promoted["store"], captures_dir=promoted["captures"],
                      dataset_dir=promoted["dataset"], dry_run=True)
    assert summary.train >= 0
    assert not promoted["dataset"].exists()


def test_promote_excludes_watched_rows(promoted):
    _capture(promoted["captures"], "2026-09-01/120000_000.jpg")
    annotations.skip("2026-09-01/120000_000.jpg", store=promoted["store"])
    summary = promote(store=promoted["store"], captures_dir=promoted["captures"],
                      dataset_dir=promoted["dataset"], val_frac=0.0, test_frac=0.0)
    assert summary.watched == 1
    ds = promoted["dataset"]
    assert not (ds / "images" / "cap_20260901_120000_000.jpg").exists()
    assert not (ds / "labels" / "cap_20260901_120000_000.txt").exists()
    manifest = (ds / "train.txt").read_text()
    assert "cap_20260901_120000_000" not in manifest
    assert "watched" in str(summary)


def test_promote_adds_test_key_to_data_yaml(promoted):
    ds = promoted["dataset"]
    ds.mkdir()
    (ds / "data.yaml").write_text("path: dataset\ntrain: train.txt\nval: val.txt\n\nnames:\n  0: coffee_pot\n")
    promote(store=promoted["store"], captures_dir=promoted["captures"], dataset_dir=ds,
            val_frac=0.0, test_frac=0.0)
    text = (ds / "data.yaml").read_text()
    assert "test: test.txt" in text
    # idempotent: second run doesn't add it twice
    promote(store=promoted["store"], captures_dir=promoted["captures"], dataset_dir=ds,
            val_frac=0.0, test_frac=0.0)
    assert (ds / "data.yaml").read_text().count("test: test.txt") == 1
