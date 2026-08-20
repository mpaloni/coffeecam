from coffeecam.annotate import annotate_image
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
