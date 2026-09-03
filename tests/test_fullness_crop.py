import pytest
from PIL import Image

from coffeecam.fullness_crop import (
    CROP_SIZE,
    DEFAULT_POT_BOX,
    _ASPECT,
    prepare_crop,
)


def _frame(w=640, h=360, color=(120, 90, 60)):
    return Image.new("RGB", (w, h), color)


def test_output_is_square_crop_size_rgb():
    out = prepare_crop(_frame(), (100, 100, 180, 190))
    assert out.size == (CROP_SIZE, CROP_SIZE)
    assert out.mode == "RGB"


def test_custom_size_honoured():
    out = prepare_crop(_frame(), (100, 100, 180, 190), size=48)
    assert out.size == (48, 48)


def test_none_box_uses_default_pot_box():
    frame = _frame()
    assert prepare_crop(frame, None).tobytes() == prepare_crop(frame, DEFAULT_POT_BOX).tobytes()


def test_never_returns_none_for_degenerate_box():
    out = prepare_crop(_frame(), (200, 200, 200, 200))
    assert out.size == (CROP_SIZE, CROP_SIZE)


def test_box_far_outside_frame_is_clamped_not_crash():
    out = prepare_crop(_frame(), (5000, 5000, 6000, 6000))
    assert out.size == (CROP_SIZE, CROP_SIZE)


def test_negative_box_is_clamped():
    out = prepare_crop(_frame(), (-50, -50, 40, 40))
    assert out.size == (CROP_SIZE, CROP_SIZE)


def test_letterbox_pads_black_for_wide_box():
    # A wide, short box on a uniform bright frame: after expansion to portrait
    # aspect it stays wider than tall only if clamped at a vertical edge. Use a
    # box wider than the target aspect so letterboxing adds black bars.
    frame = _frame(color=(255, 255, 255))
    out = prepare_crop(frame, (10, 150, 400, 170))
    px = out.load()
    # Top-left corner should be black padding, centre should be white content.
    assert px[0, 0] == (0, 0, 0)
    assert px[CROP_SIZE // 2, CROP_SIZE // 2] == (255, 255, 255)


def test_aspect_expansion_no_stretch_of_content():
    # A horizontal white band inside the box stays a band — wider than it is
    # tall — rather than being stretched to fill the square.
    frame = Image.new("RGB", (640, 360), (0, 0, 0))
    for y in range(160, 185):
        for x in range(280, 380):
            frame.putpixel((x, y), (255, 255, 255))
    out = prepare_crop(frame, (290, 120, 372, 210))
    xs = [x for x in range(CROP_SIZE) for y in range(CROP_SIZE) if out.load()[x, y][0] > 200]
    ys = [y for x in range(CROP_SIZE) for y in range(CROP_SIZE) if out.load()[x, y][0] > 200]
    assert xs and ys
    horiz_extent = max(xs) - min(xs)
    vert_extent = max(ys) - min(ys)
    assert horiz_extent > vert_extent * 2


def test_default_pot_box_matches_target_aspect_roughly():
    x1, y1, x2, y2 = DEFAULT_POT_BOX
    assert abs((x2 - x1) / (y2 - y1) - _ASPECT) < 0.15


def test_accepts_non_rgb_frame():
    out = prepare_crop(_frame().convert("L"), (100, 100, 180, 190))
    assert out.mode == "RGB"
    assert out.size == (CROP_SIZE, CROP_SIZE)
