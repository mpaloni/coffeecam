"""Replay a day of stored ``captures/`` frames into the state-history log.

    .venv/bin/python -m coffeecam.backfill_history 2026-09-07 [--force] [--conf 0.15]

Reads the kept frames from ``captures/<day>/index.jsonl`` (falling back to a glob
of ``captures/<day>/*.jpg``), runs each through ``run_pipeline(..., transform=
False)`` — the frames are already privacy-cropped — and feeds the results to the
same ``server._record_history`` the live worker uses, so rows land in
``captures/pipeline/state-<day>.jsonl`` and level changes drop transition
artifacts, exactly as if the worker had seen those frames.

The live web service appends to the same file, so stop ``coffeecam-web`` before a
backfill and start it after. ``--force`` deletes an existing day log first
(otherwise the run refuses, to avoid appending duplicate rows).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

from coffeecam import server, state_history
from coffeecam.pipeline import DEFAULT_CONF, run_pipeline
from coffeecam.summary import DEFAULT_CAPTURES_DIR


def _kept_frames(captures_dir: Path, day: str) -> list[tuple[Path, datetime]]:
    """``(path, timestamp)`` for every kept frame of ``day``, chronological."""
    day_dir = captures_dir / day
    index = day_dir / "index.jsonl"
    out: list[tuple[Path, datetime]] = []
    if index.is_file():
        for line in index.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("kept") or "file" not in row:
                continue
            out.append((captures_dir / row["file"], datetime.fromisoformat(row["t"])))
    else:
        for p in sorted(day_dir.glob("*.jpg")):
            # 040001_331.jpg -> HH MM SS .fff
            stem = p.stem
            try:
                hms, _, ms = stem.partition("_")
                ts = datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H%M%S")
                if ms:
                    ts = ts.replace(microsecond=int(ms.ljust(6, "0")[:6]))
            except ValueError:
                continue
            out.append((p, ts))
    out.sort(key=lambda t: t[1])
    return out


def backfill(
    day: str,
    *,
    captures_dir: Path,
    conf: float = DEFAULT_CONF,
    force: bool = False,
    model=None,
    estimator=None,
) -> dict:
    log_path = state_history.path_for(captures_dir, datetime.strptime(day, "%Y-%m-%d"))
    if log_path.exists():
        if not force:
            raise SystemExit(
                f"{log_path} exists — pass --force to delete and rebuild it "
                f"(stop coffeecam-web first)"
            )
        log_path.unlink()

    frames = _kept_frames(captures_dir, day)
    if not frames:
        raise SystemExit(f"no kept frames for {day} under {captures_dir}")

    if model is None:
        from coffeecam.detect import load_model

        model = load_model()
    if estimator is None:
        from coffeecam.fullness import default_estimator

        estimator = default_estimator()
    print(f"[backfill] {len(frames)} frames, estimator {type(estimator).__name__}")

    prev_level: str | None = None
    transitions = 0
    counts: dict[str, int] = {}
    for i, (path, ts) in enumerate(frames):
        if not path.exists():
            continue
        with Image.open(path) as im:
            frame = im.convert("RGB")
        result = run_pipeline(
            frame, model=model, estimator=estimator, conf=conf, transform=False
        )
        result.ts = ts
        level = result.fullness.level
        if prev_level is not None and level != prev_level:
            transitions += 1
        prev_level = server._record_history(result, prev_level)
        counts[level] = counts.get(level, 0) + 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(frames)}  {ts:%H:%M:%S}  {level}")

    print(f"[backfill] done: {sum(counts.values())} rows, {transitions} transitions, {counts}")
    return {"rows": sum(counts.values()), "transitions": transitions, "levels": counts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("day", help="YYYY-MM-DD")
    ap.add_argument("--captures-dir", type=Path, default=DEFAULT_CAPTURES_DIR)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--force", action="store_true", help="delete an existing day log first")
    args = ap.parse_args(argv)
    backfill(args.day, captures_dir=args.captures_dir, conf=args.conf, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
