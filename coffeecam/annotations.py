"""Sidecar label store for the ``/annotate`` browser labeling endpoint.

Annotations live in ``captures/annotations.jsonl`` — one upsertable JSON row per
frame, keyed by ``rel`` (the frame's path under the captures dir). Boxes are in
natural pixel space ``[x1, y1, x2, y2]``; an empty ``boxes`` list is an explicit
negative (frame seen, no pot). A row with ``"skip": true`` is *watched* — seen
and deliberately passed over; it drops out of the ``unlabeled`` queue but
``dataset.promote`` ignores it entirely (neither a positive nor a background
negative). ``captures/`` itself is never mutated; a separate ``dataset.promote``
step turns this file into a trainable ``dataset/``.

This module is pure and lock-free so it is unit-testable without threads — the
caller (the Flask server) owns the lock that serializes writers.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_STORE = Path("captures/annotations.jsonl")

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Annotation:
    rel: str
    boxes: list[Box]  # x1, y1, x2, y2 in natural pixel space; [] = explicit negative
    labeled_at: str
    note: str = ""
    skip: bool = False  # watched: seen and passed over; promote ignores it entirely

    def to_json(self) -> dict:
        row = {
            "rel": self.rel,
            "boxes": [list(b) for b in self.boxes],
            "labeled_at": self.labeled_at,
            "note": self.note,
        }
        if self.skip:
            row["skip"] = True
        return row


def _coerce_box(raw: object) -> Box:
    """Validate one box: exactly 4 ints, x2 > x1, y2 > y1, non-negative."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError(f"box must be 4 numbers [x1, y1, x2, y2], got {raw!r}")
    coords = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"box coordinate must be a number, got {v!r}")
        if isinstance(v, float) and not v.is_integer():
            raise ValueError(f"box coordinate must be a whole number, got {v!r}")
        coords.append(int(v))
    x1, y1, x2, y2 = coords
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        raise ValueError(f"box coordinates must be non-negative, got {coords}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"box must have x2 > x1 and y2 > y1, got {coords}")
    return x1, y1, x2, y2


def validate_boxes(boxes: object, *, frame_size: tuple[int, int] | None = None) -> list[Box]:
    """Return a list of clean :data:`Box` tuples, or raise ``ValueError``.

    ``frame_size`` is ``(width, height)``; when given, boxes must fit inside it.
    An empty list is valid — it records an explicit negative.
    """
    if not isinstance(boxes, (list, tuple)):
        raise ValueError(f"boxes must be a list, got {boxes!r}")
    out: list[Box] = []
    for raw in boxes:
        x1, y1, x2, y2 = _coerce_box(raw)
        if frame_size is not None:
            w, h = frame_size
            if x2 > w or y2 > h:
                raise ValueError(f"box {[x1, y1, x2, y2]} exceeds frame {frame_size}")
        out.append((x1, y1, x2, y2))
    return out


def load(store: Path = DEFAULT_STORE) -> dict[str, Annotation]:
    """Read the store into ``{rel: Annotation}``. Missing file -> ``{}``.

    Blank and malformed lines are skipped rather than raising, so a partially
    written or hand-edited file still loads.
    """
    store = Path(store)
    out: dict[str, Annotation] = {}
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
        try:
            boxes = validate_boxes(row.get("boxes", []))
        except ValueError:
            continue
        labeled_at = row.get("labeled_at")
        if not isinstance(labeled_at, str):
            labeled_at = ""
        note = row.get("note", "")
        if not isinstance(note, str):
            note = ""
        skip = bool(row.get("skip", False))
        out[rel] = Annotation(
            rel=rel, boxes=boxes, labeled_at=labeled_at, note=note, skip=skip
        )
    return out


def _dump(rows: dict[str, Annotation], store: Path) -> None:
    """Serialize the whole dict and atomically replace ``store``."""
    store = Path(store)
    store.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(rows[rel].to_json(), ensure_ascii=False)
        for rel in sorted(rows)
    ]
    body = "\n".join(lines)
    if body:
        body += "\n"
    tmp = store.with_suffix(store.suffix + ".tmp")
    tmp.write_text(body)
    os.replace(tmp, store)


def upsert(
    rel: str,
    boxes: object,
    *,
    note: str = "",
    frame_size: tuple[int, int] | None = None,
    store: Path = DEFAULT_STORE,
) -> Annotation:
    """Insert or replace the row for ``rel``. Returns the saved :class:`Annotation`.

    Raises ``ValueError`` on a bad ``rel`` or bad boxes; the whole store is
    rewritten via a tmp file + :func:`os.replace`.
    """
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("rel must be a non-empty string")
    clean = validate_boxes(boxes, frame_size=frame_size)
    ann = Annotation(
        rel=rel,
        boxes=clean,
        labeled_at=datetime.now().isoformat(timespec="seconds"),
        note=note or "",
    )
    rows = load(store)
    rows[rel] = ann
    _dump(rows, store)
    return ann


def remove(rel: str, *, store: Path = DEFAULT_STORE) -> bool:
    """Drop the row for ``rel``. Returns ``True`` if a row was removed."""
    rows = load(store)
    if rel not in rows:
        return False
    del rows[rel]
    _dump(rows, store)
    return True


def skip(rel: str, *, store: Path = DEFAULT_STORE) -> Annotation:
    """Mark ``rel`` as *watched* (seen, deliberately not labeled).

    Writes a ``skip`` row with no boxes; :func:`dataset.promote` drops it, and the
    ``unlabeled`` queue no longer serves it. Overwrites any existing row for
    ``rel`` — call :func:`remove` first to un-skip.
    """
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("rel must be a non-empty string")
    ann = Annotation(
        rel=rel,
        boxes=[],
        labeled_at=datetime.now().isoformat(timespec="seconds"),
        skip=True,
    )
    rows = load(store)
    rows[rel] = ann
    _dump(rows, store)
    return ann


def skip_many(rels: Iterable[str], *, store: Path = DEFAULT_STORE) -> int:
    """Mark every ``rel`` in ``rels`` as watched in one rewrite.

    Rows that already exist (labeled, negative, or already skipped) are left
    untouched. Returns the number of new skip rows written.
    """
    rows = load(store)
    now = datetime.now().isoformat(timespec="seconds")
    added = 0
    for rel in rels:
        if not isinstance(rel, str) or not rel.strip() or rel in rows:
            continue
        rows[rel] = Annotation(rel=rel, boxes=[], labeled_at=now, skip=True)
        added += 1
    if added:
        _dump(rows, store)
    return added


def _iter_captures(captures_dir: Path, *, start: str | None = None) -> list[str]:
    """``rel`` paths of every ``*.jpg`` under ``captures_dir``, sorted.

    ``start`` (``YYYY-MM-DD``) keeps only frames whose top-level day dir sorts
    ``>= start`` — mirrors the ``/annotate`` queue's ``start`` filter.
    """
    captures_dir = Path(captures_dir)
    out = []
    for p in sorted(captures_dir.rglob("*.jpg")):
        rel = p.relative_to(captures_dir).as_posix()
        if start and rel.split("/", 1)[0] < start:
            continue
        out.append(rel)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="coffeecam annotation store tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser(
        "skip-unlabeled",
        help="mark every capture with no row as watched (clears the unlabeled backlog)",
    )
    s.add_argument("--captures-dir", type=Path, default=Path("captures"))
    s.add_argument("--store", type=Path, default=None)
    s.add_argument("--start", default=None, help="only frames on/after this YYYY-MM-DD")
    s.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    store = args.store or DEFAULT_STORE
    have = load(store)
    rels = [r for r in _iter_captures(args.captures_dir, start=args.start) if r not in have]
    if args.dry_run:
        print(f"[dry-run] would mark {len(rels)} frame(s) watched")
        return
    n = skip_many(rels, store=store)
    print(f"marked {n} frame(s) watched")


if __name__ == "__main__":
    main()
