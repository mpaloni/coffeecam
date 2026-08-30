"""Pull frames from the break-room coffeecam and save them for the dataset.

The camera is reached over the documented SSH tunnel (homelab-docs
``runbooks/coffeecam-tunnel.md``): ``http://192.168.50.10:8888/snapshot`` is a
single JPEG, ``/stream`` is ``multipart/x-mixed-replace`` MJPEG. **The raw frames
are upside-down and unmasked** -- the 180 deg rotation and the privacy crop live
only in the CSS of the viewer page, so this module re-applies them itself before
anything hits disk (unless ``--raw`` is passed).

Two ways to run it:

  * ``python -m coffeecam.capture --once``      -- grab one frame and exit. Drive
    it from a systemd timer for the "one frame every N minutes" passive build.
  * ``python -m coffeecam.capture`` (no --once) -- run a loop at ``--interval``
    seconds, gated to the workday window (``--days`` / ``--hours``). With
    ``--dedup`` it keeps only frames that differ from the last kept one (plus a
    forced "heartbeat" frame every ``--heartbeat-secs``), which collapses a
    static break-room scene to a fraction of the frames.

Frames land in ``captures/YYYY-MM-DD/HHMMSS.jpg`` and every decision (kept or
skipped) is appended to ``captures/YYYY-MM-DD/index.jsonl``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_SOURCE = "http://192.168.50.10:8888"
# Privacy crop from runbooks/coffeecam-tunnel.md: CSS `inset(top right bottom left)`
# as fractions of the (already rotated) frame.
DEFAULT_INSETS = (0.181, 0.434, 0.329, 0.234)
DEFAULT_OUT = Path("captures")
DEFAULT_QUALITY = 85
DEFAULT_INTERVAL = 60.0
DEFAULT_DEDUP_THRESHOLD = 8  # hamming distance on a 64-bit dhash
DEFAULT_HEARTBEAT_SECS = 300.0
DEFAULT_DAYS = "mon-fri"
DEFAULT_HOURS = (7, 19)  # keep if 7 <= hour < 19

_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


# --------------------------------------------------------------------------- #
# Frame transform
# --------------------------------------------------------------------------- #
def apply_transform(
    img: Image.Image,
    *,
    rotate180: bool = True,
    insets: tuple[float, float, float, float] | None = DEFAULT_INSETS,
) -> Image.Image:
    """Rotate 180 deg and crop to the privacy region.

    ``insets`` is ``(top, right, bottom, left)`` as fractions of the frame,
    applied *after* rotation (matching the viewer page's CSS). ``None`` skips
    the crop; ``rotate180=False`` skips the flip.
    """
    if rotate180:
        img = img.rotate(180)
    if insets is None:
        return img
    top, right, bottom, left = insets
    w, h = img.size
    box = (
        round(w * left),
        round(h * top),
        round(w * (1.0 - right)),
        round(h * (1.0 - bottom)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"insets {insets} leave an empty crop for a {w}x{h} frame")
    return img.crop(box)


# --------------------------------------------------------------------------- #
# Perceptual hash for change detection (numpy + PIL, no extra dependency)
# --------------------------------------------------------------------------- #
def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """64-bit difference hash: row-wise "is each pixel brighter than its left neighbour"."""
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    a = np.asarray(small, dtype=np.int16)
    diff = a[:, 1:] > a[:, :-1]
    packed = np.packbits(diff.reshape(-1))
    return int.from_bytes(packed.tobytes(), "big")


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def should_keep(new_hash: int, last_hash: int | None, threshold: int) -> bool:
    """Keep the frame if there is no reference yet or it differs enough from it."""
    if last_hash is None:
        return True
    return hamming(new_hash, last_hash) > threshold


# --------------------------------------------------------------------------- #
# Workday / path helpers
# --------------------------------------------------------------------------- #
def parse_days(spec: str) -> set[int]:
    """"mon-fri" or "mon,wed,fri" (or a mix) -> set of weekday ints (Mon=0)."""
    days: set[int] = set()
    for part in spec.lower().replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            i, j = _WEEKDAYS.index(lo), _WEEKDAYS.index(hi)
            rng = range(i, j + 1) if i <= j else list(range(i, 7)) + list(range(0, j + 1))
            days.update(rng)
        else:
            days.add(_WEEKDAYS.index(part))
    if not days:
        raise ValueError(f"no valid days in {spec!r}")
    return days


def is_workday_window(dt: datetime, days: set[int], start_hour: int, end_hour: int) -> bool:
    return dt.weekday() in days and start_hour <= dt.hour < end_hour


def frame_path(base: Path, dt: datetime, *, subsecond: bool = False) -> Path:
    stem = dt.strftime("%H%M%S")
    if subsecond:
        stem += f"_{dt.microsecond // 1000:03d}"
    return base / dt.strftime("%Y-%m-%d") / f"{stem}.jpg"


def index_path(base: Path, dt: datetime) -> Path:
    return base / dt.strftime("%Y-%m-%d") / "index.jsonl"


def last_kept_hash(base: Path, dt: datetime) -> int | None:
    """Reseed dedup state after a restart by reading today's index."""
    path = index_path(base, dt)
    if not path.exists():
        return None
    result = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kept") and row.get("hash"):
            result = int(row["hash"], 16)
    return result


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def fetch_snapshot(source_url: str = DEFAULT_SOURCE, timeout: float = 10.0) -> Image.Image:
    import requests

    resp = requests.get(f"{source_url.rstrip('/')}/snapshot", timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def iter_mjpeg(byte_chunks, max_frames: int | None = None):
    """Yield JPEG byte strings from an MJPEG multipart body.

    Boundary-agnostic: scans for JPEG start (``FF D8``) / end (``FF D9``) markers,
    which is enough for this camera's stream.
    """
    buf = b""
    count = 0
    for chunk in byte_chunks:
        buf += chunk
        while True:
            start = buf.find(b"\xff\xd8")
            end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
            if start == -1 or end == -1:
                break
            yield buf[start : end + 2]
            buf = buf[end + 2 :]
            count += 1
            if max_frames is not None and count >= max_frames:
                return


def iter_stream_frames(source_url: str = DEFAULT_SOURCE, timeout: float = 10.0):
    import requests

    with requests.get(f"{source_url.rstrip('/')}/stream", stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for jpeg in iter_mjpeg(resp.iter_content(chunk_size=8192)):
            yield Image.open(io.BytesIO(jpeg)).convert("RGB")


def grab_stream_frame(source_url: str = DEFAULT_SOURCE, timeout: float = 10.0) -> Image.Image:
    """One frame off the live stream, connection closed before returning."""
    for frame in iter_stream_frames(source_url, timeout=timeout):
        return frame
    raise RuntimeError("stream ended before a full JPEG frame arrived")


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
def _write_frame(img: Image.Image, path: Path, quality: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=quality)
    return path.stat().st_size


def _append_index(base: Path, dt: datetime, row: dict) -> None:
    path = index_path(base, dt)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def process_frame(
    img: Image.Image,
    dt: datetime,
    *,
    out: Path,
    raw: bool,
    quality: int,
    dedup: bool,
    dedup_threshold: int,
    last_hash: int | None,
    last_heartbeat: float,
    heartbeat_secs: float,
    subsecond: bool = False,
    source: str = "snapshot",
) -> tuple[bool, int | None, float]:
    """Transform, decide keep/skip, write + log. Returns (kept, new_last_hash, new_last_heartbeat)."""
    frame = img if raw else apply_transform(img)
    h = dhash(frame)
    now = dt.timestamp()
    heartbeat_due = (now - last_heartbeat) >= heartbeat_secs

    if dedup and not heartbeat_due and not should_keep(h, last_hash, dedup_threshold):
        _append_index(
            out,
            dt,
            {
                "t": dt.isoformat(timespec="milliseconds" if subsecond else "seconds"),
                "kept": False,
                "hash": f"{h:016x}",
                "dist": None if last_hash is None else hamming(h, last_hash),
                "src": source,
            },
        )
        return False, last_hash, last_heartbeat

    path = frame_path(out, dt, subsecond=subsecond)
    size = _write_frame(frame, path, quality)
    _append_index(
        out,
        dt,
        {
            "t": dt.isoformat(timespec="milliseconds" if subsecond else "seconds"),
            "kept": True,
            "file": str(path.relative_to(out)),
            "bytes": size,
            "hash": f"{h:016x}",
            "dist": None if last_hash is None else hamming(h, last_hash),
            "heartbeat": bool(heartbeat_due and dedup),
            "src": source,
        },
    )
    return True, h, (now if heartbeat_due else last_heartbeat)


def capture_once(
    *,
    source_url: str = DEFAULT_SOURCE,
    out: Path = DEFAULT_OUT,
    raw: bool = False,
    quality: int = DEFAULT_QUALITY,
    timeout: float = 10.0,
) -> Path | None:
    dt = datetime.now()
    img = fetch_snapshot(source_url, timeout=timeout)
    kept, _, _ = process_frame(
        img,
        dt,
        out=out,
        raw=raw,
        quality=quality,
        dedup=False,
        dedup_threshold=DEFAULT_DEDUP_THRESHOLD,
        last_hash=None,
        last_heartbeat=0.0,
        heartbeat_secs=DEFAULT_HEARTBEAT_SECS,
    )
    return frame_path(out, dt) if kept else None


def run_loop(
    *,
    source_url: str = DEFAULT_SOURCE,
    mode: str = "snapshot",
    interval: float = DEFAULT_INTERVAL,
    out: Path = DEFAULT_OUT,
    raw: bool = False,
    quality: int = DEFAULT_QUALITY,
    dedup: bool = False,
    dedup_threshold: int = DEFAULT_DEDUP_THRESHOLD,
    heartbeat_secs: float = DEFAULT_HEARTBEAT_SECS,
    days: set[int] | None = None,
    hours: tuple[int, int] = DEFAULT_HOURS,
    timeout: float = 10.0,
    max_iterations: int | None = None,
    sleep=time.sleep,
) -> None:
    days = days if days is not None else parse_days(DEFAULT_DAYS)
    start_hour, end_hour = hours
    subsecond = mode == "stream"

    last_hash = last_kept_hash(out, datetime.now())
    # Start the heartbeat clock now so the first forced-keep is heartbeat_secs away,
    # not immediate (the first real frame is kept anyway, via should_keep(None)).
    last_heartbeat = time.time()
    kept_total = skipped_total = 0
    iterations = 0

    def _outside_window_sleep():
        # Re-check once a minute; cheap and robust across midnight / DST.
        sleep(min(60.0, max(1.0, interval)))

    while max_iterations is None or iterations < max_iterations:
        dt = datetime.now()
        if not is_workday_window(dt, days, start_hour, end_hour):
            iterations += 1
            _outside_window_sleep()
            continue

        try:
            if mode == "stream":
                img = grab_stream_frame(source_url, timeout=timeout)
            else:
                img = fetch_snapshot(source_url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive across camera/tunnel blips
            print(f"[{dt.isoformat(timespec='seconds')}] fetch failed: {exc}", file=sys.stderr)
            iterations += 1
            sleep(interval)
            continue

        kept, last_hash, last_heartbeat = process_frame(
            img,
            dt,
            out=out,
            raw=raw,
            quality=quality,
            dedup=dedup,
            dedup_threshold=dedup_threshold,
            last_hash=last_hash,
            last_heartbeat=last_heartbeat,
            heartbeat_secs=heartbeat_secs,
            subsecond=subsecond,
            source=mode,
        )
        kept_total += kept
        skipped_total += not kept
        iterations += 1
        sleep(interval)

    print(f"loop done: kept={kept_total} skipped={skipped_total}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-url", default=DEFAULT_SOURCE, help=f"camera base URL (default {DEFAULT_SOURCE})")
    p.add_argument("--mode", choices=("snapshot", "stream"), default="snapshot")
    p.add_argument("--once", action="store_true", help="grab a single frame and exit (ignores --dedup/--days/--hours)")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="seconds between frames in loop mode")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--raw", action="store_true", help="save the full unrotated/uncropped frame (privacy: don't)")
    p.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="JPEG quality (default 85)")
    p.add_argument("--dedup", action="store_true", help="skip frames that look like the last kept one")
    p.add_argument("--dedup-threshold", type=int, default=DEFAULT_DEDUP_THRESHOLD, help="hamming distance to count as changed")
    p.add_argument("--heartbeat-secs", type=float, default=DEFAULT_HEARTBEAT_SECS, help="force-keep a frame at least this often under --dedup")
    p.add_argument("--days", default=DEFAULT_DAYS, help='workdays, e.g. "mon-fri" or "mon,wed,fri"')
    p.add_argument("--hours", type=int, nargs=2, metavar=("START", "END"), default=list(DEFAULT_HOURS), help="keep if START <= hour < END")
    p.add_argument("--max-iterations", type=int, default=None, help=argparse.SUPPRESS)  # tests
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.once:
        path = capture_once(
            source_url=args.source_url,
            out=args.out,
            raw=args.raw,
            quality=args.quality,
        )
        print(f"saved {path}" if path else "no frame saved")
        return

    run_loop(
        source_url=args.source_url,
        mode=args.mode,
        interval=args.interval,
        out=args.out,
        raw=args.raw,
        quality=args.quality,
        dedup=args.dedup,
        dedup_threshold=args.dedup_threshold,
        heartbeat_secs=args.heartbeat_secs,
        days=parse_days(args.days),
        hours=tuple(args.hours),
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
