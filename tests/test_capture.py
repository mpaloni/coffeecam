import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from coffeecam.capture import (
    DEFAULT_INSETS,
    apply_transform,
    dhash,
    frame_path,
    hamming,
    is_workday_window,
    iter_mjpeg,
    last_kept_hash,
    parse_days,
    process_frame,
    run_loop,
    should_keep,
)


def _img(w, h, color=(120, 120, 120)):
    return Image.new("RGB", (w, h), color)


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #
def test_apply_transform_crops_to_inset_box():
    out = apply_transform(_img(1000, 1000), insets=(0.1, 0.2, 0.3, 0.4))
    # width: 1 - left(.4) - right(.2) = .4 ; height: 1 - top(.1) - bottom(.3) = .6
    assert out.size == (400, 600)


def test_apply_transform_rotate_only():
    out = apply_transform(_img(1280, 720), insets=None)
    assert out.size == (1280, 720)


def test_apply_transform_default_insets_on_real_frame_shape():
    out = apply_transform(_img(1280, 720), insets=DEFAULT_INSETS)
    assert out.size == (424, 353)


def test_apply_transform_rejects_empty_crop():
    with pytest.raises(ValueError):
        apply_transform(_img(100, 100), insets=(0.6, 0.6, 0.6, 0.6))


def test_rotate180_is_visible():
    im = _img(4, 2)
    im.putpixel((0, 0), (255, 0, 0))
    rot = apply_transform(im, insets=None)
    assert rot.getpixel((3, 1)) == (255, 0, 0)
    assert rot.getpixel((0, 0)) != (255, 0, 0)


# --------------------------------------------------------------------------- #
# perceptual hash / dedup
# --------------------------------------------------------------------------- #
def test_dhash_identical_images_match():
    a = Image.effect_noise((64, 64), 30).convert("RGB")
    assert dhash(a) == dhash(a.copy())


def test_dhash_differs_for_different_scenes():
    left_dark = Image.fromarray(
        np.tile(np.concatenate([np.zeros((32, 32)), np.full((32, 32), 255)], axis=1).astype("uint8")[:, :, None], (1, 1, 3))
    )
    assert hamming(dhash(left_dark), dhash(left_dark.transpose(Image.FLIP_LEFT_RIGHT))) > 8


def test_should_keep_semantics():
    assert should_keep(0b1010, None, 8) is True  # no reference yet
    assert should_keep(0xFFFF, 0x0000, 4) is True  # far apart
    assert should_keep(0x0001, 0x0000, 4) is False  # within threshold


# --------------------------------------------------------------------------- #
# workday / path helpers
# --------------------------------------------------------------------------- #
def test_parse_days_range_and_list():
    assert parse_days("mon-fri") == {0, 1, 2, 3, 4}
    assert parse_days("mon,wed,fri") == {0, 2, 4}
    assert parse_days("sat-sun") == {5, 6}
    assert parse_days("fri-mon") == {4, 5, 6, 0}  # wraps


def test_is_workday_window():
    wed_10 = datetime(2026, 8, 26, 10, 0)
    wed_20 = datetime(2026, 8, 26, 20, 0)
    sun_10 = datetime(2026, 8, 30, 10, 0)
    days = parse_days("mon-fri")
    assert is_workday_window(wed_10, days, 7, 19)
    assert not is_workday_window(wed_20, days, 7, 19)
    assert not is_workday_window(sun_10, days, 7, 19)


def test_frame_path_layout():
    dt = datetime(2026, 8, 30, 7, 5, 9, 123000)
    assert frame_path(Path("captures"), dt).as_posix() == "captures/2026-08-30/070509.jpg"
    assert frame_path(Path("captures"), dt, subsecond=True).as_posix() == "captures/2026-08-30/070509_123.jpg"


# --------------------------------------------------------------------------- #
# MJPEG splitter
# --------------------------------------------------------------------------- #
def test_iter_mjpeg_splits_two_frames():
    j1 = b"\xff\xd8" + b"AAA" + b"\xff\xd9"
    j2 = b"\xff\xd8" + b"BBBB" + b"\xff\xd9"
    body = b"--boundary\r\nContent-Type: image/jpeg\r\n\r\n" + j1 + b"\r\n--boundary\r\n\r\n" + j2 + b"\r\n"
    chunks = [body[i : i + 7] for i in range(0, len(body), 7)]  # arbitrary fragmentation
    frames = list(iter_mjpeg(chunks))
    assert frames == [j1, j2]


def test_iter_mjpeg_respects_max_frames():
    j = b"\xff\xd8x\xff\xd9"
    assert len(list(iter_mjpeg([j * 5], max_frames=2))) == 2


# --------------------------------------------------------------------------- #
# process_frame: write + index
# --------------------------------------------------------------------------- #
def test_process_frame_writes_and_indexes(tmp_path):
    dt = datetime(2026, 8, 26, 9, 0, 0)
    kept, h, hb = process_frame(
        _img(1280, 720, (10, 200, 10)),
        dt,
        out=tmp_path,
        raw=False,
        quality=85,
        dedup=False,
        dedup_threshold=8,
        last_hash=None,
        last_heartbeat=0.0,
        heartbeat_secs=300.0,
    )
    assert kept is True
    saved = tmp_path / "2026-08-26" / "090000.jpg"
    assert saved.exists()
    assert Image.open(saved).size == (424, 353)  # transform applied
    rows = [json.loads(x) for x in (tmp_path / "2026-08-26" / "index.jsonl").read_text().splitlines()]
    assert rows[0]["kept"] is True and rows[0]["file"] == "2026-08-26/090000.jpg"


def test_process_frame_raw_keeps_full_size(tmp_path):
    dt = datetime(2026, 8, 26, 9, 0, 1)
    process_frame(
        _img(1280, 720), dt, out=tmp_path, raw=True, quality=85, dedup=False,
        dedup_threshold=8, last_hash=None, last_heartbeat=0.0, heartbeat_secs=300.0,
    )
    assert Image.open(tmp_path / "2026-08-26" / "090001.jpg").size == (1280, 720)


def test_process_frame_dedup_skips_near_duplicate(tmp_path):
    base = _img(1280, 720, (128, 128, 128))
    dt1 = datetime(2026, 8, 26, 9, 0, 0)
    kept1, h1, hb1 = process_frame(
        base, dt1, out=tmp_path, raw=False, quality=85, dedup=True, dedup_threshold=8,
        last_hash=None, last_heartbeat=dt1.timestamp(), heartbeat_secs=10_000.0,
    )
    dt2 = datetime(2026, 8, 26, 9, 0, 2)
    kept2, h2, hb2 = process_frame(
        base.copy(), dt2, out=tmp_path, raw=False, quality=85, dedup=True, dedup_threshold=8,
        last_hash=h1, last_heartbeat=hb1, heartbeat_secs=10_000.0,
    )
    assert kept1 is True and kept2 is False
    assert not (tmp_path / "2026-08-26" / "090002.jpg").exists()
    rows = [json.loads(x) for x in (tmp_path / "2026-08-26" / "index.jsonl").read_text().splitlines()]
    assert rows[-1]["kept"] is False


def test_process_frame_heartbeat_forces_keep_of_duplicate(tmp_path):
    base = _img(1280, 720, (128, 128, 128))
    dt1 = datetime(2026, 8, 26, 9, 0, 0)
    _, h1, _ = process_frame(
        base, dt1, out=tmp_path, raw=False, quality=85, dedup=True, dedup_threshold=8,
        last_hash=None, last_heartbeat=dt1.timestamp(), heartbeat_secs=300.0,
    )
    dt2 = datetime(2026, 8, 26, 9, 10, 0)  # 600s later, heartbeat due
    kept2, _, _ = process_frame(
        base.copy(), dt2, out=tmp_path, raw=False, quality=85, dedup=True, dedup_threshold=8,
        last_hash=h1, last_heartbeat=dt1.timestamp(), heartbeat_secs=300.0,
    )
    assert kept2 is True
    assert (tmp_path / "2026-08-26" / "091000.jpg").exists()


def test_last_kept_hash_reads_index(tmp_path):
    dt = datetime(2026, 8, 26, 9, 0, 0)
    process_frame(
        _img(1280, 720, (10, 20, 30)), dt, out=tmp_path, raw=False, quality=85, dedup=False,
        dedup_threshold=8, last_hash=None, last_heartbeat=0.0, heartbeat_secs=300.0,
    )
    assert last_kept_hash(tmp_path, dt) == dhash(apply_transform(_img(1280, 720, (10, 20, 30))))
    assert last_kept_hash(tmp_path, datetime(2020, 1, 1)) is None


# --------------------------------------------------------------------------- #
# run_loop: gating + fetch, fully offline
# --------------------------------------------------------------------------- #
def test_run_loop_skips_outside_window(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("coffeecam.capture.fetch_snapshot", lambda *a, **k: calls.append(1) or _img(1280, 720))
    # Sunday -> never in a mon-fri window; loop should fetch nothing.
    monkeypatch.setattr("coffeecam.capture.datetime", _FrozenNow(datetime(2026, 8, 30, 10, 0, 0)))
    run_loop(out=tmp_path, days=parse_days("mon-fri"), hours=(7, 19), interval=0, max_iterations=3, sleep=lambda _s: None)
    assert calls == []


def test_run_loop_captures_inside_window(tmp_path, monkeypatch):
    seq = iter([_img(1280, 720, (0, 0, 0)), _img(1280, 720, (255, 255, 255)), _img(1280, 720, (0, 0, 0))])
    monkeypatch.setattr("coffeecam.capture.fetch_snapshot", lambda *a, **k: next(seq))
    monkeypatch.setattr("coffeecam.capture.datetime", _FrozenNow(datetime(2026, 8, 26, 10, 0, 0)))
    run_loop(out=tmp_path, days=parse_days("mon-fri"), hours=(7, 19), interval=0, max_iterations=3, sleep=lambda _s: None)
    # Frozen clock -> same filename each iteration; the per-frame index still logs all three.
    rows = [json.loads(x) for x in (tmp_path / "2026-08-26" / "index.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["kept"] for r in rows)
    assert (tmp_path / "2026-08-26" / "100000.jpg").exists()


class _FrozenNow:
    """Stand-in for the module's `datetime` whose .now() returns a fixed value."""

    def __init__(self, value):
        self._value = value

    def now(self, tz=None):
        return self._value

    def __getattr__(self, name):
        return getattr(datetime, name)
