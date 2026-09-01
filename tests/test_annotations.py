import json

import pytest

from coffeecam import annotations
from coffeecam.annotations import Annotation, load, remove, upsert, validate_boxes


@pytest.fixture
def store(tmp_path):
    return tmp_path / "annotations.jsonl"


def test_load_missing_file_returns_empty(store):
    assert load(store) == {}


def test_upsert_creates_then_updates_in_place(store):
    upsert("a/1.jpg", [[10, 20, 30, 40]], store=store)
    rows = load(store)
    assert list(rows) == ["a/1.jpg"]
    assert rows["a/1.jpg"].boxes == [(10, 20, 30, 40)]

    upsert("a/1.jpg", [[11, 21, 31, 41]], note="nudged", store=store)
    rows = load(store)
    assert list(rows) == ["a/1.jpg"]  # still one row
    assert rows["a/1.jpg"].boxes == [(11, 21, 31, 41)]
    assert rows["a/1.jpg"].note == "nudged"


def test_upsert_negative_round_trips(store):
    ann = upsert("a/2.jpg", [], note="carafe removed", store=store)
    assert ann.boxes == []
    reloaded = load(store)["a/2.jpg"]
    assert reloaded.boxes == []
    assert reloaded.note == "carafe removed"


def test_upsert_return_value_matches_stored(store):
    ann = upsert("a/3.jpg", [[1, 2, 3, 4]], store=store)
    assert isinstance(ann, Annotation)
    assert load(store)["a/3.jpg"] == ann


def test_load_ignores_blank_and_malformed_lines(store):
    good = json.dumps({"rel": "a/1.jpg", "boxes": [[1, 2, 3, 4]], "labeled_at": "x", "note": ""})
    store.write_text(
        "\n".join(
            [
                good,
                "",
                "   ",
                "not json at all",
                "{}",  # dict but no rel
                json.dumps({"rel": "", "boxes": []}),  # empty rel
                json.dumps({"rel": "a/bad.jpg", "boxes": [[9, 9, 1, 1]]}),  # bad box
                json.dumps([1, 2, 3]),  # not an object
            ]
        )
        + "\n"
    )
    rows = load(store)
    assert list(rows) == ["a/1.jpg"]


def test_remove(store):
    assert remove("a/1.jpg", store=store) is False
    upsert("a/1.jpg", [[1, 2, 3, 4]], store=store)
    assert remove("a/1.jpg", store=store) is True
    assert load(store) == {}


def test_atomic_write_leaves_no_tmp_behind(store):
    upsert("a/1.jpg", [[1, 2, 3, 4]], store=store)
    upsert("a/2.jpg", [[5, 6, 7, 8]], store=store)
    remove("a/1.jpg", store=store)
    siblings = list(store.parent.iterdir())
    assert siblings == [store]
    assert not store.with_suffix(store.suffix + ".tmp").exists()


def test_store_is_written_sorted_by_rel(store):
    upsert("z/9.jpg", [[1, 2, 3, 4]], store=store)
    upsert("a/1.jpg", [[1, 2, 3, 4]], store=store)
    rels = [json.loads(line)["rel"] for line in store.read_text().splitlines()]
    assert rels == ["a/1.jpg", "z/9.jpg"]


@pytest.mark.parametrize(
    "boxes",
    [
        [[10, 10, 5, 20]],  # x2 <= x1
        [[10, 10, 20, 10]],  # y2 <= y1
        [[-1, 0, 10, 10]],  # negative
        [[0, 0, 10]],  # wrong arity
        [[0, 0, 10, "20"]],  # non-numeric
        [[0.0, 0.0, 10.5, 20.0]],  # non-integer float
        "nope",  # not a list
    ],
)
def test_upsert_rejects_bad_boxes(store, boxes):
    with pytest.raises(ValueError):
        upsert("a/1.jpg", boxes, store=store)
    assert not store.exists()  # nothing written on rejection


def test_upsert_rejects_blank_rel(store):
    with pytest.raises(ValueError):
        upsert("   ", [[1, 2, 3, 4]], store=store)


def test_validate_boxes_frame_bounds():
    assert validate_boxes([[0, 0, 10, 10]], frame_size=(10, 10)) == [(0, 0, 10, 10)]
    with pytest.raises(ValueError):
        validate_boxes([[0, 0, 11, 10]], frame_size=(10, 10))


def test_validate_boxes_accepts_integer_valued_float():
    assert validate_boxes([[0.0, 0.0, 10.0, 20.0]]) == [(0, 0, 10, 20)]


def test_default_store_constant():
    assert str(annotations.DEFAULT_STORE) == "captures/annotations.jsonl"
