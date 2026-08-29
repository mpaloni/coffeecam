import json

import numpy as np
import pytest
from PIL import Image

from coffeecam.augment_shift import augment_and_save, replay, shift_bbox, shift_image
from coffeecam.dataset import bbox_yolo_to_pixel, read_label, write_label


def test_shift_bbox_translates_and_clamps():
    # plain translation, fully in frame
    assert shift_bbox(10, 20, 40, 60, 5, -5, 100, 100) == (15, 15, 45, 55)
    # runs off the right/bottom edge -> clamped to the frame
    assert shift_bbox(80, 80, 95, 95, 20, 20, 100, 100) == (100, 100, 100, 100)


def test_shift_image_keeps_size_and_moves_content():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[0, 0] = [255, 255, 255]  # single white pixel at top-left
    img = Image.fromarray(arr)

    out = np.asarray(shift_image(img, 2, 3, fill="black"))
    assert out.shape == (10, 10, 3)
    assert tuple(out[3, 2]) == (255, 255, 255)  # moved to (x=2, y=3)
    assert tuple(out[0, 0]) == (0, 0, 0)  # exposed border filled black


def test_shift_image_wrap_matches_np_roll():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(8, 12, 3), dtype=np.uint8)
    out = np.asarray(shift_image(Image.fromarray(arr), 3, -2, fill="wrap"))
    assert np.array_equal(out, np.roll(arr, shift=(-2, 3), axis=(0, 1)))


def test_shift_image_rejects_bad_fill():
    with pytest.raises(ValueError):
        shift_image(Image.new("RGB", (4, 4)), 1, 1, fill="nope")


@pytest.fixture
def labeled_image(tmp_path):
    img_path = tmp_path / "images" / "frame.png"
    img_path.parent.mkdir(parents=True)
    Image.new("RGB", (200, 100), color="white").save(img_path)
    label_path = tmp_path / "labels" / "frame.txt"
    # pixel bbox (40, 20, 120, 70) on a 200x100 image
    write_label(label_path, 0, 0.4, 0.45, 0.4, 0.5)
    return img_path, label_path


def test_augment_and_save_writes_all_artifacts_and_manifest(tmp_path, labeled_image):
    img_path, label_path = labeled_image
    manifest = tmp_path / "augmentations.json"

    record = augment_and_save(
        img_path,
        dx=15,
        dy=-10,
        fill="edge",
        label_path=label_path,
        images_dir=tmp_path / "out/images",
        labels_dir=tmp_path / "out/labels",
        previews_dir=tmp_path / "out/previews",
        manifest_path=manifest,
    )

    for key in ("image", "label", "preview"):
        assert (tmp_path / record[key]).exists() or record[key].startswith(str(tmp_path))

    # label reflects the shifted, clamped pixel bbox (40,20,120,70) + (15,-10)
    boxes = read_label(tmp_path / "out/labels/frame_shift_x15_y-10.txt")
    assert len(boxes) == 1
    _, cx, cy, w, h = boxes[0]
    assert bbox_yolo_to_pixel(cx, cy, w, h, 200, 100) == (55, 10, 135, 60)

    entries = json.loads(manifest.read_text())
    assert len(entries) == 1
    assert entries[0]["dx"] == 15 and entries[0]["dy"] == -10
    assert entries[0]["new_bbox_px"] == [55, 10, 135, 60]


def test_manifest_upsert_is_idempotent(tmp_path, labeled_image):
    img_path, label_path = labeled_image
    manifest = tmp_path / "augmentations.json"
    kw = dict(
        label_path=label_path,
        images_dir=tmp_path / "images",
        labels_dir=tmp_path / "labels",
        previews_dir=tmp_path / "previews",
        manifest_path=manifest,
    )
    augment_and_save(img_path, dx=15, dy=-10, **kw)
    augment_and_save(img_path, dx=15, dy=-10, **kw)  # same shift again
    augment_and_save(img_path, dx=-5, dy=5, **kw)  # different shift

    entries = json.loads(manifest.read_text())
    assert len(entries) == 2  # first shift upserted, not duplicated


def test_replay_regenerates_deleted_artifacts(tmp_path, labeled_image):
    img_path, label_path = labeled_image
    manifest = tmp_path / "augmentations.json"
    rec = augment_and_save(
        img_path,
        dx=8,
        dy=8,
        label_path=label_path,
        images_dir=tmp_path / "images",
        labels_dir=tmp_path / "labels",
        previews_dir=tmp_path / "previews",
        manifest_path=manifest,
    )
    from pathlib import Path

    Path(rec["image"]).unlink()
    assert not Path(rec["image"]).exists()

    replay(manifest)
    assert Path(rec["image"]).exists()


def test_augment_rejects_shift_that_empties_the_box(tmp_path, labeled_image):
    img_path, label_path = labeled_image
    with pytest.raises(ValueError):
        augment_and_save(
            img_path,
            dx=500,
            dy=0,
            label_path=label_path,
            images_dir=tmp_path / "images",
            labels_dir=tmp_path / "labels",
            previews_dir=tmp_path / "previews",
            manifest_path=tmp_path / "m.json",
        )
