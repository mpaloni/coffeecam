from PIL import Image

from coffeecam.annotate import annotate_image, write_example, write_label_lines
from coffeecam.dataset import read_label


def test_annotate_image_writes_label_and_preview(tmp_path, sample_image):
    labels_dir = tmp_path / "labels"
    previews_dir = tmp_path / "previews"

    label_path, preview_path = annotate_image(
        sample_image,
        bbox=(20, 10, 150, 80),
        labels_dir=labels_dir,
        previews_dir=previews_dir,
    )

    assert label_path.exists()
    assert preview_path.exists()

    boxes = read_label(label_path)
    assert len(boxes) == 1
    class_id, cx, cy, w, h = boxes[0]
    assert class_id == 0
    assert 0 < cx < 1 and 0 < cy < 1
    assert 0 < w < 1 and 0 < h < 1


def test_write_label_lines_multibox(tmp_path):
    label_path = tmp_path / "labels" / "foo.txt"
    write_label_lines(label_path, [(0, 0, 50, 50), (50, 50, 100, 100)], 100, 100)
    assert len(read_label(label_path)) == 2


def test_write_label_lines_negative_is_empty_file(tmp_path):
    label_path = tmp_path / "labels" / "neg.txt"
    write_label_lines(label_path, [], 100, 100)
    assert label_path.exists()
    assert label_path.read_text() == ""
    assert read_label(label_path) == []


def test_write_example_copies_image_and_writes_label(tmp_path, sample_image):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    image_dest, label_path = write_example(
        sample_image, [(20, 10, 150, 80)], images_dir=images_dir, labels_dir=labels_dir
    )

    assert image_dest == images_dir / "sample.png"
    assert image_dest.read_bytes() == sample_image.read_bytes()
    assert label_path == labels_dir / "sample.txt"
    assert len(read_label(label_path)) == 1


def test_write_example_dest_name_overrides_stem(tmp_path, sample_image):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    image_dest, label_path = write_example(
        sample_image,
        [(1, 2, 3, 4)],
        images_dir=images_dir,
        labels_dir=labels_dir,
        dest_name="cap_20260831_161913.jpg",
    )

    assert image_dest == images_dir / "cap_20260831_161913.jpg"
    assert label_path == labels_dir / "cap_20260831_161913.txt"


def test_write_example_negative_still_copies_image(tmp_path, sample_image):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"

    image_dest, label_path = write_example(
        sample_image, [], images_dir=images_dir, labels_dir=labels_dir
    )

    assert image_dest.exists()
    assert label_path.read_text() == ""
