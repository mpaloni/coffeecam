import pytest
from PIL import Image


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (200, 100), color="white").save(path)
    return path
