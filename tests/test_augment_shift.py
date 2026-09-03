import json

import numpy as np
import pytest
from PIL import Image

from coffeecam.augment_shift import (
    apply_occlusions,
    augment_and_save,
    generate,
    generate_balanced,
    generate_from_list,
    occlusions_for_coverage,
    replay,
    rotate_bbox,
    rotate_image,
    shift_bbox,
    shift_image,
)
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


# --- rotation --------------------------------------------------------------


def test_rotate_bbox_quarter_turn_maps_left_edge_to_bottom():
    # box hugging the left-centre of a 100x100 frame
    x1, y1, x2, y2 = rotate_bbox(10, 40, 30, 60, 90, 100, 100)
    # +90 deg CCW (visual, y-down): left edge swings down to the bottom-centre
    assert (round(x1), round(y1), round(x2), round(y2)) == (40, 70, 60, 90)


def test_rotate_image_preserves_size_and_is_noop_at_zero():
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (20, 30, 3), dtype=np.uint8))
    assert rotate_image(img, 0).size == img.size
    out = rotate_image(img, 12, fill="edge")
    assert out.size == img.size


def test_augment_with_rotation_records_angle_and_grows_box(tmp_path, labeled_image):
    img_path, label_path = labeled_image
    manifest = tmp_path / "aug.json"
    rec = augment_and_save(
        img_path, dx=0, dy=0, angle=20, label_path=label_path,
        images_dir=tmp_path / "images", labels_dir=tmp_path / "labels",
        previews_dir=tmp_path / "previews", manifest_path=manifest,
    )
    assert rec["angle"] == 20
    assert "rot20" in rec["image"]
    src = rec["src_bbox_px"]
    new = rec["new_bbox_px"]
    # axis-aligned box of rotated corners is at least as wide/tall as the source
    assert (new[2] - new[0]) >= (src[2] - src[0]) - 1
    assert json.loads(manifest.read_text())[0]["angle"] == 20


# --- occlusion -----------------------------------------------------------


def test_apply_occlusions_paints_patch_and_leaves_rest():
    img = Image.new("RGB", (40, 40), color="white")
    out = np.asarray(apply_occlusions(img, [{"x1": 5, "y1": 5, "x2": 15, "y2": 15, "fill": "black"}]))
    assert tuple(out[10, 10]) == (0, 0, 0)  # inside the patch
    assert tuple(out[30, 30]) == (255, 255, 255)  # untouched


def test_augment_with_occlusion_keeps_label_and_records_patches(tmp_path, labeled_image):
    img_path, label_path = labeled_image  # pixel bbox (40,20,120,70) on 200x100
    manifest = tmp_path / "aug.json"
    patch = {"x1": 40, "y1": 20, "x2": 70, "y2": 70, "fill": "gray"}
    rec = augment_and_save(
        img_path, dx=0, dy=0, occlusions=[patch], label_path=label_path,
        images_dir=tmp_path / "images", labels_dir=tmp_path / "labels",
        previews_dir=tmp_path / "previews", manifest_path=manifest,
    )
    assert "occ1" in rec["image"]
    # label is the un-occluded box, unchanged
    assert rec["new_bbox_px"] == [40, 20, 120, 70]
    assert json.loads(manifest.read_text())[0]["occlusions"][0]["x1"] == 40


def test_augment_rejects_occlusion_that_buries_the_box(tmp_path, labeled_image):
    img_path, label_path = labeled_image
    with pytest.raises(ValueError):
        augment_and_save(
            img_path, dx=0, dy=0,
            occlusions=[{"x1": 0, "y1": 0, "x2": 200, "y2": 100, "fill": "black"}],
            label_path=label_path,
            images_dir=tmp_path / "images", labels_dir=tmp_path / "labels",
            previews_dir=tmp_path / "previews", manifest_path=tmp_path / "m.json",
        )


# --- generate ----------------------------------------------------------


def test_generate_is_deterministic_and_replayable(tmp_path, labeled_image):
    img_path, label_path = labeled_image
    kw = dict(
        label_path=label_path,
        images_dir=tmp_path / "images", labels_dir=tmp_path / "labels",
        previews_dir=tmp_path / "previews",
    )
    a = generate(img_path, 8, seed=1, manifest_path=tmp_path / "a.json", **kw)
    b = generate(img_path, 8, seed=1, manifest_path=tmp_path / "b.json", **kw)
    assert [r["image"].split("/")[-1] for r in a] == [r["image"].split("/")[-1] for r in b]

    c = generate(img_path, 8, seed=2, manifest_path=tmp_path / "c.json", **kw)
    assert [r["image"] for r in a] != [r["image"] for r in c]

    from pathlib import Path

    for r in a:  # every recorded sample really landed on disk
        assert Path(r["image"]).exists()


def test_generate_from_list_augments_positives_and_skips_negatives(tmp_path):
    ds = tmp_path / "dataset"
    images, labels = ds / "images", ds / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    # two real positives, one negative, one already-synthetic entry
    for stem in ("cap_a", "cap_b"):
        Image.new("RGB", (200, 160), color="white").save(images / f"{stem}.jpg")
        write_label(labels / f"{stem}.txt", 0, 0.5, 0.5, 0.2, 0.2)
    Image.new("RGB", (200, 160), color="white").save(images / "cap_neg.jpg")
    (labels / "cap_neg.txt").write_text("")  # negative
    Image.new("RGB", (200, 160), color="white").save(images / "kahvi_shift_x5_y5.png")

    train_list = ds / "train.txt"
    train_list.write_text(
        "./images/cap_a.jpg\n./images/cap_b.jpg\n./images/cap_neg.jpg\n"
        "./images/kahvi_shift_x5_y5.png\n"
    )

    stats = generate_from_list(
        train_list, per=3, dataset_dir=ds, manifest_path=ds / "augmentations.json"
    )
    assert stats["sources"] == 2  # cap_a, cap_b
    assert stats["skipped"] == 1  # cap_neg (kahvi_* is silently ignored, not counted)
    assert 1 <= stats["generated"] <= 6

    made = sorted(p.name for p in images.glob("cap_*_shift_x*"))
    assert made and all("_rot" in n for n in made)
    # a label rode along for every generated image
    for p in images.glob("cap_*_shift_x*"):
        assert (labels / f"{p.stem}.txt").exists()


# --- balanced recipe -------------------------------------------------------


@pytest.fixture
def centered_small_box(tmp_path):
    """Small centred box on a roomy frame -- no transform in the recipe range
    pushes it out of frame, so the recipe counts land exactly."""
    img_path = tmp_path / "images" / "frame.png"
    img_path.parent.mkdir(parents=True)
    Image.new("RGB", (400, 400), color="white").save(img_path)
    label_path = tmp_path / "labels" / "frame.txt"
    write_label(label_path, 0, 0.5, 0.5, 0.12, 0.12)  # 48x48 px, centred
    return img_path, label_path


def test_occlusions_for_coverage_scales_with_target():
    import random

    box = (20.0, 20.0, 120.0, 80.0)  # 100 x 60
    [small] = occlusions_for_coverage(random.Random(0), box, 200, 200, 0.15)
    [big] = occlusions_for_coverage(random.Random(0), box, 200, 200, 0.6)
    span = lambda p: min((p["x2"] - p["x1"]) / 100, (p["y2"] - p["y1"]) / 60)
    assert span(small) < span(big)


def test_generate_balanced_lays_down_the_recipe(tmp_path, centered_small_box):
    img_path, label_path = centered_small_box
    recs = generate_balanced(
        img_path, n_shift=3, n_rotate=3, n_occlude=2, seed=1,
        label_path=label_path,
        images_dir=tmp_path / "images", labels_dir=tmp_path / "labels",
        previews_dir=tmp_path / "previews", manifest_path=tmp_path / "m.json",
    )
    names = [r["image"].split("/")[-1] for r in recs]
    pure_shift = [n for n in names if "_rot" not in n and "_occ" not in n]
    rotated = [n for n in names if "_rot" in n and "_occ" not in n]
    occluded = [n for n in names if "_occ" in n]
    assert len(pure_shift) == 3
    assert len(rotated) == 3
    assert len(occluded) == 2
    # every occluded copy carries at least one patch
    for r in recs:
        if "_occ" in r["image"]:
            assert len(r["occlusions"]) >= 1


def test_generate_balanced_is_deterministic(tmp_path, labeled_image):
    img_path, label_path = labeled_image
    kw = dict(
        label_path=label_path, images_dir=tmp_path / "i", labels_dir=tmp_path / "l",
        previews_dir=tmp_path / "p",
    )
    a = generate_balanced(img_path, seed=2, manifest_path=tmp_path / "a.json", **kw)
    b = generate_balanced(img_path, seed=2, manifest_path=tmp_path / "b.json", **kw)
    assert [r["image"].split("/")[-1] for r in a] == [r["image"].split("/")[-1] for r in b]


def test_generate_from_list_balanced_mode(tmp_path):
    ds = tmp_path / "dataset"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    Image.new("RGB", (200, 160), color="white").save(ds / "images" / "cap_a.jpg")
    write_label(ds / "labels" / "cap_a.txt", 0, 0.5, 0.5, 0.3, 0.3)
    (ds / "train.txt").write_text("./images/cap_a.jpg\n")

    stats = generate_from_list(
        ds / "train.txt", dataset_dir=ds, manifest_path=ds / "m.json",
        balanced=True, n_shift=2, n_rotate=2, n_occlude=1,
    )
    assert stats["sources"] == 1
    made = sorted(p.name for p in (ds / "images").glob("cap_a_shift_x*"))
    assert sum("_occ" in n for n in made) == 1
    assert sum("_rot" in n and "_occ" not in n for n in made) == 2


def test_generate_from_list_is_deterministic(tmp_path):
    ds = tmp_path / "dataset"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    Image.new("RGB", (200, 160), color="white").save(ds / "images" / "cap_a.jpg")
    write_label(ds / "labels" / "cap_a.txt", 0, 0.5, 0.5, 0.2, 0.2)
    (ds / "train.txt").write_text("./images/cap_a.jpg\n")

    s1 = generate_from_list(ds / "train.txt", per=4, dataset_dir=ds, manifest_path=ds / "m1.json")
    names1 = sorted(p.name for p in (ds / "images").glob("cap_a_shift_x*"))
    s2 = generate_from_list(ds / "train.txt", per=4, dataset_dir=ds, manifest_path=ds / "m2.json")
    names2 = sorted(p.name for p in (ds / "images").glob("cap_a_shift_x*"))
    assert s1 == s2 and names1 == names2
