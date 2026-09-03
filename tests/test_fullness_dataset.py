import pytest
from PIL import Image

from coffeecam import annotations, fullness_labels
from coffeecam.dataset import _split_bucket
from coffeecam.fullness_crop import CROP_SIZE
import random

from coffeecam.fullness_dataset import MAX_VARIANTS, MERGES, build, jitter_box, remap


@pytest.fixture
def stores(tmp_path):
    """captures/ with N frames, each a GT box + a fullness level."""
    captures = tmp_path / "captures"
    annot = captures / "annotations.jsonl"
    full = captures / "fullness.jsonl"
    levels = ["empty", "low", "half", "high", "full", "absent"] * 4  # 24 frames
    for i, lvl in enumerate(levels):
        rel = f"2026-09-0{i % 3 + 1}/{i:06d}.jpg"
        p = captures / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (424, 353), (80, 80, 80)).save(p, "JPEG")
        annotations.upsert(rel, [[291, 113, 373, 205]], store=annot)
        fullness_labels.upsert(rel, lvl, store=full)
    # one watched fullness row + one box-less annotation -> both skipped
    fullness_labels.skip("2026-09-01/999001.jpg", store=full)
    fullness_labels.upsert("2026-09-01/999002.jpg", "half", store=full)
    annotations.upsert("2026-09-01/999002.jpg", [], store=annot)  # negative, no box
    return captures


def test_dry_run_counts_and_skips(stores):
    s = build(captures_dir=stores, out_dir=stores.parent / "out", merge="none", dry_run=True)
    assert s.total == 24
    assert s.skipped_watched == 1
    assert s.skipped_no_box == 1
    assert not (stores.parent / "out").exists()  # dry run writes nothing


def test_split_matches_dataset_bucketing(stores):
    s = build(captures_dir=stores, out_dir=stores.parent / "out", dry_run=True)
    # Recompute the expected split independently from the same hash helper.
    expect = {"train": 0, "val": 0, "test": 0}
    for row in fullness_labels.load(stores / "fullness.jsonl").values():
        if row.skip or row.rel == "2026-09-01/999002.jpg":
            continue
        expect[_split_bucket(row.rel, seed=0, val_frac=0.15, test_frac=0.15)] += 1
    got = {k: sum(v.values()) for k, v in s.counts.items()}
    assert got == expect


def test_merge_coarse_collapses_levels(stores):
    s = build(captures_dir=stores, out_dir=stores.parent / "out", merge="coarse", dry_run=True)
    assert s.classes == ["absent", "empty", "lots", "some"]
    total = {c: sum(sp.get(c, 0) for sp in s.counts.values()) for c in s.classes}
    assert total["some"] == 8  # 4 low + 4 half
    assert total["lots"] == 8  # 4 high + 4 full
    assert total["empty"] == 4
    assert total["absent"] == 4


def test_build_writes_imagefolder_tree(stores, tmp_path):
    out = tmp_path / "fds"
    s = build(captures_dir=stores, out_dir=out, merge="binary")
    for split in ("train", "val", "test"):
        for cls in ("empty", "has_coffee", "absent"):
            assert (out / split / cls).is_dir()
    jpgs = list(out.rglob("*.jpg"))
    assert len(jpgs) == s.total == 24
    with Image.open(jpgs[0]) as im:
        assert im.size == (CROP_SIZE, CROP_SIZE)


def test_rebuild_is_idempotent_and_swaps_classes(stores, tmp_path):
    out = tmp_path / "fds"
    build(captures_dir=stores, out_dir=out, merge="none")
    assert (out / "train" / "low").exists()
    build(captures_dir=stores, out_dir=out, merge="binary")  # rebuild, different classes
    assert not (out / "train" / "low").exists()  # stale class dir gone
    assert (out / "train" / "has_coffee").exists()


def test_refuses_to_wipe_unrelated_dir(stores, tmp_path):
    out = tmp_path / "notours"
    out.mkdir()
    (out / "important.txt").write_text("keep me")
    with pytest.raises(RuntimeError, match="refusing to wipe"):
        build(captures_dir=stores, out_dir=out)
    assert (out / "important.txt").exists()


def test_balance_oversamples_train_only(stores, tmp_path):
    out = tmp_path / "fds"
    plain = build(captures_dir=stores, out_dir=tmp_path / "p", merge="coarse", dry_run=True)
    bal = build(captures_dir=stores, out_dir=out, merge="coarse", balance=True)

    # val/test identical to the un-balanced build
    for split in ("val", "test"):
        assert bal.counts[split] == plain.counts[split]
    # train classes pulled toward parity
    tc = bal.counts["train"]
    assert max(tc.values()) - min(tc.values()) <= 2
    assert max(tc.values()) <= MAX_VARIANTS * max(plain.counts["train"].values())
    # jittered variants really written, and still valid crops
    js = list(out.glob("train/*/*_j*.jpg"))
    assert js
    with Image.open(js[0]) as im:
        assert im.size == (CROP_SIZE, CROP_SIZE)


def test_balance_is_deterministic(stores, tmp_path):
    a = build(captures_dir=stores, out_dir=tmp_path / "a", merge="coarse", balance=True)
    b = build(captures_dir=stores, out_dir=tmp_path / "b", merge="coarse", balance=True)
    assert a.counts == b.counts
    names_a = sorted(p.relative_to(tmp_path / "a").as_posix() for p in (tmp_path / "a").rglob("*.jpg"))
    names_b = sorted(p.relative_to(tmp_path / "b").as_posix() for p in (tmp_path / "b").rglob("*.jpg"))
    assert names_a == names_b


def test_jitter_box_stays_near_original():
    rng = random.Random(0)
    box = (100.0, 100.0, 180.0, 190.0)
    for _ in range(50):
        x1, y1, x2, y2 = jitter_box(box, rng)
        assert x2 > x1 and y2 > y1
        assert abs((x1 + x2) / 2 - 140) < 40 and abs((y1 + y2) / 2 - 145) < 40


def test_remap_and_bad_merge():
    assert remap("full", "coarse") == "lots"
    assert remap("full", "none") == "full"
    with pytest.raises(ValueError):
        build(merge="nope", dry_run=True)
    assert set(MERGES) == {"none", "coarse", "binary"}
