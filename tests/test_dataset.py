from coffeecam.dataset import (
    bbox_pixel_to_yolo,
    bbox_yolo_to_pixel,
    clamp_bbox,
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
