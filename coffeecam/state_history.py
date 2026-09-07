"""Persistent fullness-state timeline: one JSONL row per pipeline tick.

The Flask worker calls :func:`append_row` every tick with the current
``PipelineResult``. Rows land in ``<captures>/pipeline/state-YYYY-MM-DD.jsonl``
(keyed by the tick's own date, so a run spanning midnight splits cleanly). This
is independent of ``COFFEECAM_HARVEST`` — the log is small (~150 B/row) and
always on; HARVEST stays the opt-in heavy frame+crop dump.

A row is::

    {"ts": "2026-09-07T14:23:01", "level": "half", "score": 0.52,
     "method": "model:v2", "conf": 0.41, "bbox": [x1, y1, x2, y2],
     "timings_ms": {...}, "errors": [], "stale": false}

On a level change vs. the previous tick the caller also drops the frame/crop
artifacts for that tick (see ``server._harvest``) and passes their relative path
as ``artifact=``; it is recorded only on those transition rows.

:func:`load_rows` reads a day back; :func:`runs_from_rows` collapses consecutive
same-level rows into ``{level, start, end, duration_s, frames, artifact}``
segments — the "how the pot went from empty to full" view.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

FILE_PREFIX = "state-"
FILE_SUFFIX = ".jsonl"


def history_dir(captures_dir: Path | str) -> Path:
    return Path(captures_dir) / "pipeline"


def _classifier_top1(f) -> float | None:
    """Top-1 class probability from a ``ModelFullness`` result's ``detail.probs``
    — the classifier's confidence in its own call. ``None`` for estimators that
    don't expose a prob vector (Null/Brightness)."""
    probs = (f.detail or {}).get("probs")
    if not isinstance(probs, dict) or not probs:
        return None
    return round(max(probs.values()), 4)


def _row_from_result(result, *, artifact: str | None, frame_rel: str | None) -> dict:
    d = result.detection
    f = result.fullness
    row = {
        "ts": result.ts.isoformat(timespec="seconds"),
        "level": f.level,
        "score": None if f.score is None else round(f.score, 3),
        "p": _classifier_top1(f),  # classifier top-1 probability (confidence)
        "method": f.method,
        "conf": None if d is None else round(d.confidence, 3),  # detector confidence
        "bbox": None if d is None else [int(v) for v in d.bbox],
        "timings_ms": result.timings_ms,
        "errors": list(result.errors),
    }
    if frame_rel:
        row["frame"] = frame_rel
    if artifact:
        row["transition"] = True
        row["artifact"] = artifact
    return row


def path_for(captures_dir: Path | str, when: datetime) -> Path:
    return history_dir(captures_dir) / f"{FILE_PREFIX}{when:%Y-%m-%d}{FILE_SUFFIX}"


def append_row(
    captures_dir: Path | str,
    result,
    *,
    artifact: str | None = None,
    frame_rel: str | None = None,
) -> dict:
    """Append one row for ``result`` and return it. Creates the day file and the
    ``pipeline/`` dir on first use. ``frame_rel`` records the source capture path
    (set by the backfill; the live worker has no stored frame)."""
    row = _row_from_result(result, artifact=artifact, frame_rel=frame_rel)
    dst = path_for(captures_dir, result.ts)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    return row


def available_dates(captures_dir: Path | str) -> list[str]:
    """Sorted (ascending) list of ``YYYY-MM-DD`` strings that have a state log."""
    hd = history_dir(captures_dir)
    if not hd.is_dir():
        return []
    out = []
    for p in hd.glob(f"{FILE_PREFIX}*{FILE_SUFFIX}"):
        out.append(p.name[len(FILE_PREFIX):-len(FILE_SUFFIX)])
    return sorted(out)


def recent_dates(captures_dir: Path | str, days: int) -> list[str]:
    """The most recent ``days`` dates that have a state log (ascending)."""
    ds = available_dates(captures_dir)
    return ds[-days:] if days > 0 else ds


def load_span(
    captures_dir: Path | str, *, days: int, max_points: int | None = None
) -> tuple[list[str], list[dict]]:
    """Rows for the last ``days`` logged days, each tagged with ``_date``,
    concatenated in chronological order. When ``max_points`` is set and the row
    count exceeds it, evenly stride the rows down to roughly that many (transition
    rows — those with an ``artifact`` — are always kept)."""
    dates = recent_dates(captures_dir, days)
    rows: list[dict] = []
    for d in dates:
        for r in load_rows(captures_dir, date=d):
            r["_date"] = d
            rows.append(r)
    if max_points and len(rows) > max_points:
        step = len(rows) / max_points
        keep, acc = [], 0.0
        for i, r in enumerate(rows):
            if r.get("artifact") or i >= acc:
                keep.append(r)
                if i >= acc:
                    acc += step
        rows = keep
    return dates, rows


def load_rows(
    captures_dir: Path | str, *, date: str | None = None, limit: int | None = None
) -> list[dict]:
    """Rows for ``date`` (``YYYY-MM-DD``), or the latest day with a log when
    ``date`` is None. ``limit`` keeps only the most recent N rows. Malformed
    lines are skipped rather than raising."""
    if date is None:
        dates = available_dates(captures_dir)
        if not dates:
            return []
        date = dates[-1]
    src = history_dir(captures_dir) / f"{FILE_PREFIX}{date}{FILE_SUFFIX}"
    if not src.is_file():
        return []
    rows: list[dict] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if limit is not None and limit >= 0:
        rows = rows[-limit:]
    return rows


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def runs_from_rows(rows: list[dict]) -> list[dict]:
    """Collapse consecutive same-``level`` rows into timeline segments.

    Each run: ``{level, start, end, duration_s, frames, artifact}`` where
    ``artifact`` is the transition frame recorded on the row that opened the run
    (``None`` for the first run of the day, which has no prior state)."""
    runs: list[dict] = []
    for row in rows:
        level = row.get("level")
        ts = row.get("ts")
        if runs and runs[-1]["level"] == level:
            run = runs[-1]
            run["end"] = ts
            run["frames"] += 1
        else:
            runs.append(
                {
                    "level": level,
                    "start": ts,
                    "end": ts,
                    "frames": 1,
                    "artifact": row.get("artifact"),
                }
            )
    for run in runs:
        a, b = _parse_ts(run["start"]), _parse_ts(run["end"])
        run["duration_s"] = None if not (a and b) else round((b - a).total_seconds(), 1)
    return runs
