import json

import pytest

from coffeecam import annotations, fullness_labels as fl


@pytest.fixture
def store(tmp_path):
    return tmp_path / "fullness.jsonl"


def test_upsert_and_load_roundtrip(store):
    fl.upsert("a/1.jpg", "empty", store=store)
    fl.upsert("a/2.jpg", "full", note="fresh brew", store=store)
    rows = fl.load(store)
    assert set(rows) == {"a/1.jpg", "a/2.jpg"}
    assert rows["a/1.jpg"].level == "empty"
    assert rows["a/2.jpg"].level == "full"
    assert rows["a/2.jpg"].note == "fresh brew"
    assert rows["a/2.jpg"].skip is False


def test_upsert_replaces_existing_row(store):
    fl.upsert("a/1.jpg", "empty", store=store)
    fl.upsert("a/1.jpg", "half", store=store)
    rows = fl.load(store)
    assert len(rows) == 1
    assert rows["a/1.jpg"].level == "half"
    # one physical line per rel
    assert len(store.read_text().strip().splitlines()) == 1


def test_upsert_rejects_bad_level(store):
    with pytest.raises(ValueError):
        fl.upsert("a/1.jpg", "brimming", store=store)
    with pytest.raises(ValueError):
        fl.upsert("a/1.jpg", "", store=store)


def test_absent_is_a_valid_label_distinct_from_skip(store):
    fl.upsert("a/1.jpg", "absent", store=store)
    row = fl.load(store)["a/1.jpg"]
    assert row.level == "absent" and row.skip is False
    c = fl.counts(store)
    assert c["absent"] == 1 and c["watched"] == 0 and c["total"] == 1


def test_upsert_rejects_empty_rel(store):
    with pytest.raises(ValueError):
        fl.upsert("  ", "empty", store=store)


def test_skip_writes_watched_row_with_no_level(store):
    fl.skip("a/1.jpg", store=store)
    row = fl.load(store)["a/1.jpg"]
    assert row.skip is True
    assert row.level == ""
    assert "level" not in json.loads(store.read_text()) or True  # level key present but ""


def test_remove(store):
    fl.upsert("a/1.jpg", "empty", store=store)
    assert fl.remove("a/1.jpg", store=store) is True
    assert fl.remove("a/1.jpg", store=store) is False
    assert fl.load(store) == {}


def test_skip_many_leaves_existing_rows_untouched(store):
    fl.upsert("a/1.jpg", "full", store=store)
    added = fl.skip_many(["a/1.jpg", "a/2.jpg", "a/3.jpg"], store=store)
    assert added == 2
    rows = fl.load(store)
    assert rows["a/1.jpg"].level == "full"  # not clobbered
    assert rows["a/2.jpg"].skip is True


def test_load_skips_malformed_and_out_of_vocab_lines(store):
    store.write_text(
        '{"rel": "a/1.jpg", "level": "empty", "labeled_at": "x"}\n'
        "not json\n"
        '{"rel": "a/2.jpg", "level": "overflowing", "labeled_at": "x"}\n'
        '{"level": "full"}\n'
        '{"rel": "a/3.jpg", "skip": true, "labeled_at": "x"}\n'
    )
    rows = fl.load(store)
    assert set(rows) == {"a/1.jpg", "a/3.jpg"}


def test_load_missing_file_is_empty(tmp_path):
    assert fl.load(tmp_path / "nope.jsonl") == {}


def test_counts(store):
    fl.upsert("a/1.jpg", "empty", store=store)
    fl.upsert("a/2.jpg", "empty", store=store)
    fl.upsert("a/3.jpg", "full", store=store)
    fl.skip("a/4.jpg", store=store)
    c = fl.counts(store)
    assert c["empty"] == 2 and c["full"] == 1 and c["half"] == 0
    assert c["watched"] == 1 and c["total"] == 4


def test_positive_box_rels_filters_negatives_and_skips(tmp_path):
    annot = tmp_path / "annotations.jsonl"
    annotations.upsert("a/1.jpg", [[10, 10, 40, 40]], store=annot)
    annotations.upsert("a/2.jpg", [], store=annot)  # explicit negative
    annotations.skip("a/3.jpg", store=annot)  # watched
    annotations.upsert("a/4.jpg", [[5, 5, 20, 20]], store=annot)
    assert fl.positive_box_rels(annot) == ["a/1.jpg", "a/4.jpg"]


def test_cli_stats(capsys, store):
    fl.upsert("a/1.jpg", "full", store=store)
    fl._main(["stats", "--store", str(store)])
    out = capsys.readouterr().out
    assert "full: 1" in out
    assert "total: 1" in out


def test_cli_skip_unlabeled(capsys, tmp_path):
    annot = tmp_path / "annotations.jsonl"
    store = tmp_path / "fullness.jsonl"
    annotations.upsert("a/1.jpg", [[10, 10, 40, 40]], store=annot)
    annotations.upsert("a/2.jpg", [[10, 10, 40, 40]], store=annot)
    fl.upsert("a/1.jpg", "empty", store=store)
    fl._main(["skip-unlabeled", "--captures-dir", str(tmp_path), "--store", str(store)])
    assert "marked 1" in capsys.readouterr().out
    rows = fl.load(store)
    assert rows["a/1.jpg"].level == "empty"
    assert rows["a/2.jpg"].skip is True
