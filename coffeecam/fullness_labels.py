"""Sidecar label store for the ``/fullness`` browser labeling endpoint.

Fill-level labels live in ``captures/fullness.jsonl`` — one upsertable JSON row
per frame, keyed by ``rel`` (the frame's path under the captures dir):
``{"rel": ..., "level": "empty", "labeled_at": ...}``. ``level`` is one of
:data:`LEVELS` (``empty``/``low``/``half``/``high``/``full``, shown in the UI as
a 1-5 scale). A row with ``"skip": true`` is *watched* — the carafe is absent,
mid-pour, or otherwise unlabelable — and ``level`` is empty; ``fullness_dataset``
drops it. ``captures/`` itself is never mutated.

Only frames that already have a positive ``coffee_pot`` box in
``annotations.jsonl`` belong in the queue (the box is needed to crop). This
module doesn't enforce that — the server builds the queue — but the CLI helpers
here do.

Pure and lock-free (unit-testable without threads); the Flask server owns the
lock that serializes writers. Mirrors ``coffeecam/annotations.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from coffeecam import annotations
from coffeecam.fullness import LEVELS  # ("empty", "low", "half", "high", "full")

# The /fullness UI presents these as a 1-5 scale (1 = empty .. 5 = full); the
# store keeps the names so fullness_dataset can build an ImageFolder tree from
# them directly. Same enum as coffeecam.fullness.FullnessResult.
__all__ = ["LEVELS", "FullnessLabel", "load", "upsert", "remove", "skip",
           "skip_many", "counts", "positive_box_rels"]

DEFAULT_STORE = Path("captures/fullness.jsonl")


@dataclass(frozen=True)
class FullnessLabel:
    rel: str
    level: str  # one of LEVELS; "" only when skip is True
    labeled_at: str
    note: str = ""
    skip: bool = False  # watched: carafe absent / mid-pour / unlabelable

    def to_json(self) -> dict:
        row = {"rel": self.rel, "level": self.level, "labeled_at": self.labeled_at}
        if self.note:
            row["note"] = self.note
        if self.skip:
            row["skip"] = True
        return row


def _valid_level(level: object) -> str:
    if not isinstance(level, str) or level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return level


def load(store: Path = DEFAULT_STORE) -> dict[str, FullnessLabel]:
    """Read the store into ``{rel: FullnessLabel}``. Missing file -> ``{}``.

    Blank and malformed lines are skipped rather than raising, so a partially
    written or hand-edited file still loads.
    """
    store = Path(store)
    out: dict[str, FullnessLabel] = {}
    if not store.exists():
        return out
    for line in store.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(row, dict):
            continue
        rel = row.get("rel")
        if not isinstance(rel, str) or not rel:
            continue
        skip = bool(row.get("skip", False))
        level = row.get("level", "")
        if not skip:
            try:
                level = _valid_level(level)
            except ValueError:
                continue
        else:
            level = ""
        labeled_at = row.get("labeled_at")
        if not isinstance(labeled_at, str):
            labeled_at = ""
        note = row.get("note", "")
        if not isinstance(note, str):
            note = ""
        out[rel] = FullnessLabel(
            rel=rel, level=level, labeled_at=labeled_at, note=note, skip=skip
        )
    return out


def _dump(rows: dict[str, FullnessLabel], store: Path) -> None:
    """Serialize the whole dict and atomically replace ``store``."""
    store = Path(store)
    store.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        json.dumps(rows[rel].to_json(), ensure_ascii=False) for rel in sorted(rows)
    )
    if body:
        body += "\n"
    tmp = store.with_suffix(store.suffix + ".tmp")
    tmp.write_text(body)
    os.replace(tmp, store)


def upsert(
    rel: str,
    level: str,
    *,
    note: str = "",
    store: Path = DEFAULT_STORE,
) -> FullnessLabel:
    """Insert or replace the row for ``rel``. Returns the saved label.

    Raises ``ValueError`` on a bad ``rel`` or a ``level`` outside :data:`LEVELS`.
    """
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("rel must be a non-empty string")
    level = _valid_level(level)
    label = FullnessLabel(
        rel=rel,
        level=level,
        labeled_at=datetime.now().isoformat(timespec="seconds"),
        note=note or "",
    )
    rows = load(store)
    rows[rel] = label
    _dump(rows, store)
    return label


def remove(rel: str, *, store: Path = DEFAULT_STORE) -> bool:
    """Drop the row for ``rel``. Returns ``True`` if a row was removed."""
    rows = load(store)
    if rel not in rows:
        return False
    del rows[rel]
    _dump(rows, store)
    return True


def skip(rel: str, *, store: Path = DEFAULT_STORE) -> FullnessLabel:
    """Mark ``rel`` *watched* (carafe absent / mid-pour / unlabelable).

    Overwrites any existing row for ``rel`` — call :func:`remove` first to un-skip.
    """
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("rel must be a non-empty string")
    label = FullnessLabel(
        rel=rel,
        level="",
        labeled_at=datetime.now().isoformat(timespec="seconds"),
        skip=True,
    )
    rows = load(store)
    rows[rel] = label
    _dump(rows, store)
    return label


def skip_many(rels: Iterable[str], *, store: Path = DEFAULT_STORE) -> int:
    """Mark every ``rel`` in ``rels`` watched in one rewrite. Existing rows are
    left untouched. Returns the number of new skip rows written."""
    rows = load(store)
    now = datetime.now().isoformat(timespec="seconds")
    added = 0
    for rel in rels:
        if not isinstance(rel, str) or not rel.strip() or rel in rows:
            continue
        rows[rel] = FullnessLabel(rel=rel, level="", labeled_at=now, skip=True)
        added += 1
    if added:
        _dump(rows, store)
    return added


def positive_box_rels(annot_store: Path) -> list[str]:
    """``rel`` of every frame with at least one ``coffee_pot`` box — the frames
    eligible for fullness labelling, sorted."""
    return sorted(
        rel
        for rel, ann in annotations.load(annot_store).items()
        if ann.boxes and not ann.skip
    )


def counts(store: Path = DEFAULT_STORE) -> dict[str, int]:
    """``{level: n}`` for each of :data:`LEVELS` plus ``watched`` and ``total``."""
    rows = load(store)
    out = {lvl: 0 for lvl in LEVELS}
    out["watched"] = 0
    for label in rows.values():
        if label.skip:
            out["watched"] += 1
        elif label.level in out:
            out[label.level] += 1
    out["total"] = len(rows)
    return out


def _main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="coffeecam fullness label store tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("stats", help="print class counts in the fullness store")
    st.add_argument("--store", type=Path, default=None)

    s = sub.add_parser(
        "skip-unlabeled",
        help="mark every box-positive frame with no fullness row as watched",
    )
    s.add_argument("--captures-dir", type=Path, default=Path("captures"))
    s.add_argument("--store", type=Path, default=None)
    s.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)
    store = args.store or DEFAULT_STORE

    if args.cmd == "stats":
        c = counts(store)
        for lvl in (*LEVELS, "watched", "total"):
            print(f"{lvl:>8}: {c[lvl]}")
        return

    annot_store = args.captures_dir / "annotations.jsonl"
    have = load(store)
    todo = [rel for rel in positive_box_rels(annot_store) if rel not in have]
    if args.dry_run:
        print(f"[dry-run] would mark {len(todo)} frame(s) watched")
        return
    n = skip_many(todo, store=store)
    print(f"marked {n} frame(s) watched")


if __name__ == "__main__":
    _main()
