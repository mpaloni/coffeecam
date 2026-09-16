"""Flask dev server: runs the pipeline on a timer and serves every stage.

    .venv/bin/python -m coffeecam.server

A daemon thread fetches a snapshot every COFFEECAM_REFRESH_SECS, runs
`pipeline.run_pipeline`, and stashes the rendered JPEGs; the routes just serve the
last result, so browser refreshes and multiple viewers cost nothing. The classify
stage runs `fullness.default_estimator()` (`ModelFullness` when
`models/FULLNESS_CHECKPOINT` resolves, else `NullFullness`).

Env:
  COFFEECAM_SOURCE_URL   camera base URL           (default: see hosts.env / hosts.env.example)
  COFFEECAM_REFRESH_SECS seconds between runs      (default 10)
  COFFEECAM_CONF         detector confidence floor (default 0.15)
  COFFEECAM_NORMALIZE    1/0 black-pad to train ar (default 1)
  COFFEECAM_HARVEST      1/0 save frame+crop+json  (default 0) -> captures/pipeline/
  COFFEECAM_HOST / COFFEECAM_PORT                  (default 0.0.0.0 / 8000)
  COFFEECAM_SUMMARY_TTL  seconds to cache /summary (default 300)
  COFFEECAM_CAPTURES_DIR  frame dir for /summary + /annotate (default captures/)
  COFFEECAM_DATASET_DIR   output dir for POST /annotate/promote (default dataset/)
  COFFEECAM_ARTIFACTS_DIR  dir served read-only at /artifacts (default scratchpad/)
  COFFEECAM_STATE_HISTORY  1/0 append per-tick fullness row (default 1)
  COFFEECAM_TRANSITION_ARTIFACTS  1/0 dump frame+crop on level change (default 1)

/history is the persisted fullness-state timeline: one JSONL row per pipeline
tick under captures/pipeline/state-YYYY-MM-DD.jsonl (always on, independent of
COFFEECAM_HARVEST), and on every level change the transition frame+crop are
saved alongside. /history renders a segmented day bar + transition thumbnails;
/history.json returns run-collapsed segments, /history/rows.json the raw ticks,
/history/long[.json] a fill-level graph over the last ?days= logged days.
Backfill a day from stored frames with `python -m coffeecam.backfill_history`;
`python -m coffeecam.fullness_confidence` summarises classifier confidence.
Full docs: docs/PIPELINE.md.

/artifacts is a read-only gallery of COFFEECAM_ARTIFACTS_DIR (images + text/json/
log) — drop a file in, refresh, no restart. /fullness/compare.gif renders
fullness-v1 vs the brightness heuristic over the held-out test split.

/summary[.gif] renders every captured frame so far into one animated GIF, each
frame annotated with the current detector's result box (query params: set,
annotate, gt, frames, ms, scale, conf, rebuild). ?set=captures (default) | train |
val | test picks the image set — the dataset splits point the annotator at the
promoted, YOLO-labelled frames. ?gt=1 additionally draws each split frame's
YOLO ground-truth box in cyan (no effect on ?set=captures). /summary.json
returns the build metadata.

/compare[.gif] stitches two or more detectors' result boxes side by side on the
same frames (query params: model=NAME=run_or_pt repeatable — default is the
pre-retrain baseline vs models/CHECKPOINT — plus set, frames, ms, scale, conf,
rebuild). /compare.json returns per-model strong/weak/none counts. No mAP: use
the `python -m coffeecam.compare` CLI for scored (`.val()`) comparisons.

/viewer is the scrubbable version of the same timelapse: an HTML page with
play/pause and a frame slider, backed by /viewer/frame/<i>.jpg (the annotated
frames, served individually) and /viewer/manifest.json (per-frame source path,
timestamp and detection). Same annotate/frames/scale/conf/rebuild query params.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from coffeecam import state_history
from coffeecam.capture import DEFAULT_SOURCE, fetch_snapshot
from coffeecam.pipeline import DEFAULT_CONF, PipelineResult, run_pipeline
from coffeecam.summary import (
    DEFAULT_CAPTURES_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_DURATION_MS,
    FRAMESETS,
    SummaryEmpty,
    build_summary_gif,
    collect_frames,
    detect_on_frame,
)
from coffeecam.viewer import DEFAULT_MAX_FRAMES as VIEWER_DEFAULT_FRAMES
from coffeecam.viewer import build_viewer

_lock = threading.Lock()
_latest: PipelineResult | None = None
_latest_jpeg: dict[str, bytes] = {}
_last_fetch_error: str | None = None
_worker_started = False

# Set once the pipeline worker has loaded the detector; reused by /summary so it
# doesn't have to load its own copy of the model.
_model = None
_summary_lock = threading.Lock()
_summary_cache: dict | None = None
_viewer_lock = threading.Lock()
_viewer_cache: dict | None = None
# Serializes writers to captures/annotations.jsonl (the /annotate label store).
# The store itself is cheap to read, so reads happen per-request with no cache.
_annot_lock = threading.Lock()
# Same, for captures/fullness.jsonl (the /fullness label store).
_fullness_lock = threading.Lock()
# Lazily-loaded singleton for on-demand /fullness/suggest.json calls — separate
# from the worker's own estimator instance (loaded fresh on first use, not at
# startup, so the /fullness page works even before the worker thread ticks).
_fullness_estimator = None


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(int(default))).strip().lower() in ("1", "true", "yes", "on")


def _cfg() -> dict:
    return {
        "source_url": os.environ.get("COFFEECAM_SOURCE_URL", DEFAULT_SOURCE),
        "refresh": float(os.environ.get("COFFEECAM_REFRESH_SECS", "10")),
        "conf": float(os.environ.get("COFFEECAM_CONF", str(DEFAULT_CONF))),
        "normalize": _env_bool("COFFEECAM_NORMALIZE", True),
        "harvest": _env_bool("COFFEECAM_HARVEST", False),
    }


def _encode(img) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _store(result: PipelineResult) -> None:
    jpegs = {"frame": _encode(result.frame), "bounded": _encode(result.bounded)}
    if result.normalized is not None:
        jpegs["normalized"] = _encode(result.normalized)
    if result.crop is not None:
        jpegs["crop"] = _encode(result.crop)
    global _latest, _latest_jpeg
    with _lock:
        _latest = result
        _latest_jpeg = jpegs


def _pipeline_dir() -> Path:
    """`<captures>/pipeline/` — home of the HARVEST dump, the state-history
    JSONL, and the transition frame/crop artifacts."""
    return _annot_captures_dir() / "pipeline"


def _harvest(result: PipelineResult) -> str:
    """Write ``<stem>_frame.jpg`` / ``_crop.jpg`` / ``.json`` for this tick under
    ``<captures>/pipeline/<day>/`` and return the frame's path relative to
    ``<captures>/pipeline`` (e.g. ``2026-09-07/142301_frame.jpg``)."""
    day = result.ts.strftime("%Y-%m-%d")
    stem = result.ts.strftime("%H%M%S")
    base = _pipeline_dir() / day
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{stem}_frame.jpg").write_bytes(_encode(result.frame))
    if result.crop is not None:
        (base / f"{stem}_crop.jpg").write_bytes(_encode(result.crop))
    (base / f"{stem}.json").write_text(json.dumps(_result_dict(result), indent=None))
    return f"{day}/{stem}_frame.jpg"


def _record_history(
    result: PipelineResult, prev_level: str | None, *, frame_rel: str | None = None
) -> str | None:
    """Append a state-history row for this tick; on a level change vs. the
    previous tick also dump the transition frame/crop artifacts. Best-effort —
    never raises into the worker loop. Returns this tick's level. ``frame_rel``
    is set by the backfill to record the source capture path."""
    level = result.fullness.level
    if not _env_bool("COFFEECAM_STATE_HISTORY", True):
        return level
    changed = prev_level is not None and level != prev_level
    try:
        artifact = None
        if changed and _env_bool("COFFEECAM_TRANSITION_ARTIFACTS", True):
            try:
                artifact = _harvest(result)
            except Exception as exc:  # noqa: BLE001
                print(f"transition artifact failed: {exc}")
        state_history.append_row(
            _annot_captures_dir(), result, artifact=artifact, frame_rel=frame_rel
        )
    except Exception as exc:  # noqa: BLE001
        print(f"state history failed: {exc}")
    return level


def _worker(model) -> None:
    global _last_fetch_error
    cfg = _cfg()
    from coffeecam.fullness import default_estimator

    estimator = default_estimator()  # load the fullness model once, not per frame
    print(f"[fullness] estimator: {type(estimator).__name__}")
    prev_level: str | None = None
    while True:
        try:
            snap = fetch_snapshot(cfg["source_url"])
            _last_fetch_error = None
            result = run_pipeline(
                snap, model=model, estimator=estimator,
                normalize=cfg["normalize"], conf=cfg["conf"],
            )
            _store(result)
            prev_level = _record_history(result, prev_level)
            if cfg["harvest"]:
                try:
                    _harvest(result)
                except Exception as exc:  # noqa: BLE001
                    print(f"harvest failed: {exc}")
        except Exception as exc:  # noqa: BLE001 — camera/tunnel down; keep serving the last frame
            _last_fetch_error = str(exc)
            print(f"[{datetime.now():%H:%M:%S}] pipeline tick failed: {exc}")
        time.sleep(cfg["refresh"])


def _start_worker() -> None:
    global _worker_started, _model
    if _worker_started:
        return
    _worker_started = True
    from coffeecam.detect import load_model

    model = load_model()
    _model = model
    threading.Thread(target=_worker, args=(model,), daemon=True, name="coffeecam-pipeline").start()


def _result_dict(result: PipelineResult) -> dict:
    d = result.detection
    return {
        "ts": result.ts.isoformat(timespec="seconds"),
        "stale_seconds": round((datetime.now() - result.ts).total_seconds(), 1),
        "detection": None
        if d is None
        else {"bbox": list(d.bbox), "confidence": round(d.confidence, 3)},
        "fullness": {
            "level": result.fullness.level,
            "score": result.fullness.score,
            "method": result.fullness.method,
            "detail": result.fullness.detail,
        },
        "timings_ms": result.timings_ms,
        "errors": result.errors,
        "fetch_error": _last_fetch_error,
        "images": {
            "frame": "/frame.jpg",
            "normalized": "/normalized.jpg" if result.normalized is not None else None,
            "bounded": "/bounded.jpg",
            "crop": "/crop.jpg" if result.crop is not None else None,
        },
    }


def _arg_int(name: str, default: int) -> int:
    try:
        return int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default


def _arg_float(name: str, default: float) -> float:
    try:
        return float(request.args.get(name, default))
    except (TypeError, ValueError):
        return default


def _arg_bool(name: str, default: bool) -> bool:
    raw = request.args.get(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def _summary_args() -> dict:
    # Route defaults trade fidelity for a lighter payload (a full-res annotated
    # GIF of every frame runs ~18 MB); pass ?scale=1&frames=240 for the lot.
    frameset = request.args.get("set", "captures").strip().lower()
    if frameset not in FRAMESETS:
        frameset = "captures"
    return dict(
        frameset=frameset,
        annotate=_arg_bool("annotate", True),
        gt=_arg_bool("gt", False),
        max_frames=_arg_int("frames", 160),
        ms=_arg_int("ms", DEFAULT_DURATION_MS),
        scale=_arg_float("scale", 0.6),
        conf=_arg_float("conf", DEFAULT_CONF),
        force=_arg_bool("rebuild", False),
    )


def _build_summary(*, frameset: str, annotate: bool, gt: bool, max_frames: int, ms: int, scale: float, conf: float, force: bool):
    """Cached wrapper around `summary.build_summary_gif`. Rebuilds when the
    parameters change, on `?rebuild=1`, or once the cached GIF is older than
    COFFEECAM_SUMMARY_TTL seconds (default 300) so new captures roll in."""
    global _summary_cache
    ttl = float(os.environ.get("COFFEECAM_SUMMARY_TTL", "300"))
    captures_dir = Path(os.environ.get("COFFEECAM_CAPTURES_DIR", DEFAULT_CAPTURES_DIR))
    dataset_dir = Path(os.environ.get("COFFEECAM_DATASET_DIR", DEFAULT_DATASET_DIR))
    sig = (str(captures_dir), str(dataset_dir), frameset, annotate, gt, max_frames, ms, round(scale, 3), round(conf, 3))
    with _summary_lock:
        cached = _summary_cache
        if cached and not force and cached["sig"] == sig and (time.time() - cached["at"]) < ttl:
            return cached["gif"], cached["meta"]
        gif, meta = build_summary_gif(
            captures_dir=captures_dir,
            frameset=frameset,
            dataset_dir=dataset_dir,
            model=_model if annotate else None,
            conf=conf,
            max_frames=max_frames,
            duration_ms=ms,
            scale=scale,
            gt=gt,
        )
        meta = {**meta, "model_ready": _model is not None, "cache_ttl_s": ttl}
        _summary_cache = {"sig": sig, "gif": gif, "meta": meta, "at": time.time()}
        return gif, meta


def _viewer_args() -> dict:
    return dict(
        annotate=_arg_bool("annotate", True),
        max_frames=_arg_int("frames", VIEWER_DEFAULT_FRAMES),
        scale=_arg_float("scale", 0.75),
        conf=_arg_float("conf", DEFAULT_CONF),
        force=_arg_bool("rebuild", False),
    )


def _build_viewer(*, annotate: bool, max_frames: int, scale: float, conf: float, force: bool):
    """Cached wrapper around `viewer.build_viewer`. Same cache discipline as
    `_build_summary`: rebuilds when the params change, on `?rebuild=1`, or once the
    cached build is older than COFFEECAM_VIEWER_TTL (falls back to
    COFFEECAM_SUMMARY_TTL, then 300s) so new captures roll in."""
    global _viewer_cache
    ttl = float(
        os.environ.get("COFFEECAM_VIEWER_TTL", os.environ.get("COFFEECAM_SUMMARY_TTL", "300"))
    )
    captures_dir = Path(os.environ.get("COFFEECAM_CAPTURES_DIR", DEFAULT_CAPTURES_DIR))
    sig = (str(captures_dir), annotate, max_frames, round(scale, 3), round(conf, 3))
    with _viewer_lock:
        cached = _viewer_cache
        if cached and not force and cached["sig"] == sig and (time.time() - cached["at"]) < ttl:
            return cached["build"]
        build = build_viewer(
            captures_dir=captures_dir,
            model=_model if annotate else None,
            conf=conf,
            max_frames=max_frames,
            scale=scale,
        )
        build.meta = {**build.meta, "model_ready": _model is not None, "cache_ttl_s": ttl}
        _viewer_cache = {"sig": sig, "build": build, "at": time.time()}
        return build


# --- /compare: two (or more) detectors side by side on the same frames --------

_compare_lock = threading.Lock()
_compare_cache: dict | None = None
_compare_models: dict[str, object] = {}  # resolved-weights-path -> loaded YOLO

# `?model=NAME=run_or_pt` is repeatable; with none given we diff the pre-retrain
# baseline against whatever `models/CHECKPOINT` now points at.
DEFAULT_COMPARE_MODELS = (
    ("nomosaic-2", "runs/detect/runs/train-nomosaic-2"),
    ("checkpoint", ""),  # "" -> models/CHECKPOINT (the live weights, reuses _model)
)


def _compare_model_for(spec: str):
    """Load (and cache) the model for a ``run dir | .pt | ""`` spec. ``""`` means
    the live checkpoint — reuse the worker's already-loaded ``_model`` if there."""
    from coffeecam.compare import resolve_model_weights
    from coffeecam.detect import load_model, resolve_weights

    weights = resolve_weights(None) if spec == "" else resolve_model_weights(spec)
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"no weights at {weights}")
    key = str(weights.resolve())
    if spec == "" and _model is not None and key == str(Path(resolve_weights(None)).resolve()):
        return _model
    if key not in _compare_models:
        _compare_models[key] = load_model(weights)
    return _compare_models[key]


def _compare_args() -> dict:
    raw = request.args.getlist("model")
    pairs = []
    for item in raw:
        name, _, spec = item.partition("=")
        pairs.append((name.strip() or spec.strip(), spec.strip()))
    if not pairs:
        pairs = list(DEFAULT_COMPARE_MODELS)
    frameset = request.args.get("set", "captures").strip().lower()
    if frameset not in FRAMESETS:
        frameset = "captures"
    return dict(
        pairs=tuple(pairs),
        frameset=frameset,
        max_frames=_arg_int("frames", 80),
        ms=_arg_int("ms", 350),
        scale=_arg_float("scale", 0.6),
        conf=_arg_float("conf", DEFAULT_CONF),
        force=_arg_bool("rebuild", False),
    )


def _build_compare(*, pairs, frameset, max_frames, ms, scale, conf, force):
    """Cached wrapper around `compare.build_comparison_gif`. Same cache discipline
    as `_build_summary`. No mAP here — `.val()` is too slow for a request; use the
    `coffeecam.compare` CLI for scored comparisons."""
    global _compare_cache
    from coffeecam.compare import build_comparison_gif
    from coffeecam.summary import resolve_frames

    ttl = float(os.environ.get("COFFEECAM_SUMMARY_TTL", "300"))
    captures_dir = Path(os.environ.get("COFFEECAM_CAPTURES_DIR", DEFAULT_CAPTURES_DIR))
    dataset_dir = Path(os.environ.get("COFFEECAM_DATASET_DIR", DEFAULT_DATASET_DIR))
    sig = (str(captures_dir), str(dataset_dir), pairs, frameset, max_frames, ms,
           round(scale, 3), round(conf, 3))
    with _compare_lock:
        cached = _compare_cache
        if cached and not force and cached["sig"] == sig and (time.time() - cached["at"]) < ttl:
            return cached["gif"], cached["meta"]
        named = {name: _compare_model_for(spec) for name, spec in pairs}
        refs = resolve_frames(frameset, captures_dir=captures_dir, dataset_dir=dataset_dir)
        gif, stats = build_comparison_gif(
            refs, named, conf=conf, scale=scale, max_frames=max_frames, duration_ms=ms
        )
        meta = {**stats, "set": frameset, "specs": {n: (s or "models/CHECKPOINT") for n, s in pairs}}
        _compare_cache = {"sig": sig, "gif": gif, "meta": meta, "at": time.time()}
        return gif, meta


# --- /artifacts: read-only static gallery of a scratch dir --------------------

_ARTIFACT_IMG = {".gif", ".png", ".jpg", ".jpeg", ".webp", ".svg"}
_ARTIFACT_SUFFIXES = _ARTIFACT_IMG | {".json", ".txt", ".csv", ".md", ".log"}


def _artifacts_dir() -> Path:
    """Dir served by /artifacts. Default `scratchpad/` (gitignored, where compare
    / summary outputs already land); override with COFFEECAM_ARTIFACTS_DIR.

    Resolved to an absolute path: Flask's `send_from_directory` joins a *relative*
    directory onto the package root (`coffeecam/`), not the cwd, which would miss
    the repo-root `scratchpad/`."""
    return Path(os.environ.get("COFFEECAM_ARTIFACTS_DIR", "scratchpad")).resolve()


_ARTIFACTS_PAGE = """<!doctype html><meta charset=utf-8><title>coffeecam artifacts</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{{font:14px system-ui,sans-serif;margin:0;background:#14161a;color:#e6e6e6}}
 header{{padding:10px 16px;background:#1d2026;border-bottom:1px solid #2c2f36}}
 h1{{font-size:15px;margin:0;font-weight:600}} a{{color:#6ab0ff}} .muted{{color:#8a909a}}
 .grid{{display:flex;flex-wrap:wrap;gap:14px;padding:16px}}
 figure{{margin:0;flex:0 1 320px;background:#1d2026;border:1px solid #2c2f36;border-radius:8px;padding:8px}}
 figure img{{display:block;width:100%;height:auto;background:#000;border-radius:4px}}
 figcaption{{font:12px ui-monospace,monospace;color:#e6e6e6;margin-top:6px;word-break:break-all}}
 small{{color:#8a909a}} a{{text-decoration:none}}
</style>
<header><h1>coffeecam artifacts <span class=muted>&middot; {dir}/ &middot;
 <a href="/">/</a></span></h1></header>
<div class=grid>
{body}
</div>
"""


# --- /annotate: browser bbox-labeling backed by captures/annotations.jsonl ---

def _annot_captures_dir() -> Path:
    return Path(os.environ.get("COFFEECAM_CAPTURES_DIR", DEFAULT_CAPTURES_DIR))


def _annot_store_path() -> Path:
    return _annot_captures_dir() / "annotations.jsonl"


def _safe_capture_path(rel: str) -> Path | None:
    """Resolve ``rel`` under the captures dir, guarding against ``..`` traversal.

    Returns the absolute path (which may not exist), or ``None`` for an empty rel
    or one that escapes the captures dir.
    """
    if not isinstance(rel, str) or not rel.strip():
        return None
    base = _annot_captures_dir().resolve()
    path = (base / rel).resolve()
    if path != base and base not in path.parents:
        return None
    return path


def _annot_queue():
    """Ordered labeling queue for the current query params.

    Returns ``(frames, counts, picked, captures_dir)`` where ``frames`` is the
    JSON-ready list and ``picked`` is the parallel list of ``(rel, FrameRef,
    labeled)``. Nothing outside this request should index ``picked`` by
    position: ``/annotate/frame.jpg`` and ``/annotate/suggest.json`` address
    frames by ``?rel=`` because the queue shifts as frames are labeled.

    Params: ``filter=unlabeled|labeled|watched|all`` (default unlabeled),
    ``stride=N`` (take every Nth frame, default 1), ``start=YYYY-MM-DD``.

    A *watched* row (``skip=True``) counts as neither labeled nor unlabeled: it
    is out of the default queue but reachable via ``filter=watched``.
    """
    from coffeecam import annotations

    captures_dir = _annot_captures_dir()
    refs = collect_frames(captures_dir)
    anns = annotations.load(_annot_store_path())

    start = request.args.get("start")
    filt = request.args.get("filter", "unlabeled")
    stride = max(1, _arg_int("stride", 1))

    rows = []
    for r in refs:
        rel = r.path.relative_to(captures_dir).as_posix()
        if start and r.day < start:
            continue
        ann = anns.get(rel)
        is_skip = ann is not None and ann.skip
        is_l = ann is not None and not ann.skip
        rows.append((rel, r, is_l, is_skip))

    total = len(rows)
    labeled = sum(1 for _, _, is_l, _ in rows if is_l)
    watched = sum(1 for _, _, _, is_s in rows if is_s)

    strided = rows[::stride]
    if filt == "labeled":
        picked = [x for x in strided if x[2]]
    elif filt == "watched":
        picked = [x for x in strided if x[3]]
    elif filt == "all":
        picked = list(strided)
    else:
        picked = [x for x in strided if not x[2] and not x[3]]

    frames = []
    for i, (rel, r, is_l, is_skip) in enumerate(picked):
        ann = anns.get(rel)
        frames.append({
            "i": i,
            "rel": rel,
            "ts": r.ts_label,
            "labeled": is_l,
            "skip": is_skip,
            "boxes": [list(b) for b in ann.boxes] if ann else [],
        })
    counts = {
        "total": total,
        "labeled": labeled,
        "watched": watched,
        "remaining": total - labeled - watched,
    }
    return frames, counts, picked, captures_dir


# --- /fullness: browser fill-level labeling backed by captures/fullness.jsonl ---

def _fullness_store_path() -> Path:
    return _annot_captures_dir() / "fullness.jsonl"


def _fullness_queue():
    """Ordered fill-level labeling queue for the current query params.

    Returns ``(frames, counts, picked, captures_dir)`` where ``picked`` is the
    parallel list of ``(rel, box, level, skip)``. ``/fullness/crop.jpg`` and
    ``/fullness/frame.jpg`` address frames by ``?rel=`` (not by position in
    ``picked``); the crop route re-reads the GT ``coffee_pot`` box from the
    annotation store by ``rel``.

    Only frames with a positive box in ``annotations.jsonl`` are eligible.
    Params: ``filter=unlabeled|labeled|watched|all`` (default unlabeled),
    ``stride=N`` (default 1), ``start=YYYY-MM-DD`` (frame's day dir).
    """
    from coffeecam import annotations, fullness_labels

    captures_dir = _annot_captures_dir()
    anns = annotations.load(_annot_store_path())
    labels = fullness_labels.load(_fullness_store_path())

    start = request.args.get("start")
    filt = request.args.get("filter", "unlabeled")
    stride = max(1, _arg_int("stride", 1))

    rows = []
    for rel in sorted(anns):
        ann = anns[rel]
        if not ann.boxes or ann.skip:
            continue
        if start and rel.split("/", 1)[0] < start:
            continue
        lab = labels.get(rel)
        is_skip = lab is not None and lab.skip
        is_l = lab is not None and not lab.skip
        rows.append((rel, list(ann.boxes[0]), lab.level if is_l else "", is_skip))

    total = len(rows)
    labeled = sum(1 for _, _, lvl, _ in rows if lvl)
    watched = sum(1 for *_, is_s in rows if is_s)

    strided = rows[::stride]
    if filt == "labeled":
        picked = [x for x in strided if x[2]]
    elif filt == "watched":
        picked = [x for x in strided if x[3]]
    elif filt == "all":
        picked = list(strided)
    else:
        picked = [x for x in strided if not x[2] and not x[3]]

    frames = [
        {"i": i, "rel": rel, "box": box, "level": lvl, "skip": is_skip}
        for i, (rel, box, lvl, is_skip) in enumerate(picked)
    ]
    counts = {
        "total": total,
        "labeled": labeled,
        "watched": watched,
        "remaining": total - labeled - watched,
    }
    return frames, counts, picked, captures_dir


def _get_fullness_estimator():
    """Lazy singleton `FullnessEstimator` for `/fullness/suggest.json`. Reloaded
    only once per process; `NullFullness` (no weights yet) is cached too so a
    fresh clone doesn't retry the resolve on every request."""
    global _fullness_estimator
    if _fullness_estimator is None:
        from coffeecam.fullness import default_estimator

        _fullness_estimator = default_estimator()
    return _fullness_estimator


def _fullness_crop_jpeg(frame_path: Path, box, *, full: bool = False) -> bytes | None:
    """JPEG bytes of ``prepare_crop(frame, box)`` (or the full frame when
    ``full``). ``None`` if the frame file is gone."""
    from PIL import Image as _Image

    from coffeecam.fullness_crop import prepare_crop

    if not frame_path.exists():
        return None
    with _Image.open(frame_path) as im:
        frame = im.convert("RGB")
    img = frame if full else prepare_crop(frame, tuple(box))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


_ANNOTATE_PAGE = """<!doctype html><meta charset=utf-8><title>coffeecam annotate</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{font:14px system-ui,sans-serif;margin:0;background:#14161a;color:#e6e6e6}
 header{padding:10px 16px;background:#1d2026;border-bottom:1px solid #2c2f36}
 h1{font-size:15px;margin:0;font-weight:600} .muted{color:#8a909a} a{color:#6ab0ff}
 .wrap{max-width:1000px;margin:0 auto;padding:16px;display:flex;gap:16px;flex-wrap:wrap}
 .stage{position:relative;flex:1 1 480px;align-self:flex-start;max-width:100%}
 .stage img{display:block;width:100%;height:auto;background:#000;border:1px solid #2c2f36;border-radius:8px}
 .stage canvas{position:absolute;left:0;top:0;touch-action:none;cursor:crosshair}
 .col{flex:0 0 210px;display:flex;flex-direction:column;gap:8px}
 button,select{font:13px system-ui;padding:6px 10px;background:#2c2f36;color:#e6e6e6;border:1px solid #3a3f47;border-radius:6px;cursor:pointer;text-align:left}
 button:hover,select:hover{background:#3a3f47}
 .row{display:flex;gap:6px} .row button{flex:1}
 label.f{font-size:12px;color:#8a909a;display:flex;justify-content:space-between;align-items:center;gap:6px}
 label.f select{flex:1}
 .cap{font:12px ui-monospace,Menlo,monospace;color:#8a909a;word-break:break-all;line-height:1.6}
 .cap b{color:#e6e6e6} .k{color:#6ab0ff}
 .done{padding:20px;background:#1d2026;border:1px solid #2c2f36;border-radius:8px;line-height:1.8}
 .done code{background:#0e1013;padding:2px 6px;border-radius:4px}
</style>
<header><h1>coffeecam annotate
 <span class=muted>&middot; draw a <b>coffee_pot</b> box &middot;
 <a href="/viewer">/viewer</a> &middot; <a href="/">/</a></span></h1></header>
<div class=wrap>
 <div class=stage id=stage><img id=view alt=""><canvas id=cv></canvas></div>
 <div class=col>
  <div class=cap id=hdr>loading&hellip;</div>
  <div class=cap id=count></div>
  <div class=row><button id=prev>&larr; prev</button><button id=next>next &rarr;</button></div>
  <button id=save>save + next &nbsp;<span class=k>Space</span></button>
  <button id=neg>negative (no pot) &nbsp;<span class=k>x</span></button>
  <button id=skip>skip / watched &nbsp;<span class=k>s</span></button>
  <button id=hint>suggest a box &nbsp;<span class=k>h</span></button>
  <button id=clear>clear boxes &nbsp;<span class=k>d</span></button>
  <button id=del>delete saved label &nbsp;<span class=k>&#9003;</span></button>
  <button id=skiprest>skip rest of queue</button>
  <button id=reload>reload queue</button>
  <label class=f>filter <select id=filter>
    <option value=unlabeled selected>unlabeled</option>
    <option value=labeled>labeled</option>
    <option value=watched>watched</option>
    <option value=all>all</option></select></label>
  <label class=f>stride <select id=stride>
    <option>1</option><option>2</option><option>3</option><option>5</option><option>10</option>
   </select></label>
  <div class=cap muted id=meta></div>
 </div>
</div>
<script>
const params = new URLSearchParams(location.search);
const $ = id => document.getElementById(id);
const view = $('view'), cv = $('cv'), ctx = cv.getContext('2d');
const HANDLE = 8;  // px, display-space hit radius for corner handles
let queue = [], counts = {}, pos = 0;
let boxes = [], suggestion = null, sel = -1, undo = [];
let drag = null;  // {mode:'new'|'move'|'nw'|'ne'|'sw'|'se', ox, oy, orig}

const qs = () => {
  const p = new URLSearchParams();
  p.set('filter', $('filter').value);
  p.set('stride', $('stride').value);
  if (params.get('start')) p.set('start', params.get('start'));
  return p;
};
const sx = () => view.clientWidth / (view.naturalWidth || 1);   // natural -> display
const sy = () => view.clientHeight / (view.naturalHeight || 1);

function fitCanvas() {
  cv.width = view.clientWidth; cv.height = view.clientHeight;
  cv.style.width = view.clientWidth + 'px'; cv.style.height = view.clientHeight + 'px';
  draw();
}
addEventListener('resize', fitCanvas);

function draw() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (suggestion) for (const b of suggestion) {
    ctx.setLineDash([6, 4]); ctx.strokeStyle = '#e0a020'; ctx.lineWidth = 2;
    ctx.strokeRect(b[0]*sx(), b[1]*sy(), (b[2]-b[0])*sx(), (b[3]-b[1])*sy());
  }
  ctx.setLineDash([]);
  boxes.forEach((b, i) => {
    ctx.strokeStyle = i === sel ? '#6ab0ff' : '#ff4040'; ctx.lineWidth = 2;
    const x = b.x1*sx(), y = b.y1*sy(), w = (b.x2-b.x1)*sx(), h = (b.y2-b.y1)*sy();
    ctx.strokeRect(x, y, w, h);
    if (i === sel) {
      ctx.fillStyle = '#6ab0ff';
      for (const [hx, hy] of [[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
        ctx.fillRect(hx-HANDLE/2, hy-HANDLE/2, HANDLE, HANDLE);
    }
  });
}

const toNat = e => {
  const r = cv.getBoundingClientRect();
  return [ (e.clientX - r.left) / sx(), (e.clientY - r.top) / sy() ];
};
function norm(b) {
  return { x1: Math.min(b.x1,b.x2), y1: Math.min(b.y1,b.y2),
           x2: Math.max(b.x1,b.x2), y2: Math.max(b.y1,b.y2) };
}
function hitHandle(b, nx, ny) {
  const rx = HANDLE / sx(), ry = HANDLE / sy();
  const c = { nw:[b.x1,b.y1], ne:[b.x2,b.y1], sw:[b.x1,b.y2], se:[b.x2,b.y2] };
  for (const k in c)
    if (Math.abs(nx-c[k][0]) < rx && Math.abs(ny-c[k][1]) < ry) return k;
  return null;
}
function inside(b, nx, ny) { return nx>b.x1 && nx<b.x2 && ny>b.y1 && ny<b.y2; }

cv.addEventListener('pointerdown', e => {
  cv.setPointerCapture(e.pointerId);
  const [nx, ny] = toNat(e);
  if (sel >= 0) {
    const h = hitHandle(boxes[sel], nx, ny);
    if (h) { pushUndo(); drag = { mode:h, orig:{...boxes[sel]} }; return; }
    if (inside(boxes[sel], nx, ny)) {
      pushUndo(); drag = { mode:'move', ox:nx, oy:ny, orig:{...boxes[sel]} }; return;
    }
  }
  for (let i = boxes.length-1; i >= 0; i--)
    if (inside(boxes[i], nx, ny)) { sel = i; draw(); return; }
  pushUndo();
  boxes.push({ x1:nx, y1:ny, x2:nx, y2:ny }); sel = boxes.length-1;
  drag = { mode:'new' };
  draw();
});
cv.addEventListener('pointermove', e => {
  if (!drag) return;
  const [nx, ny] = toNat(e), b = boxes[sel];
  if (drag.mode === 'new') { b.x2 = nx; b.y2 = ny; }
  else if (drag.mode === 'move') {
    const dx = nx-drag.ox, dy = ny-drag.oy;
    b.x1 = drag.orig.x1+dx; b.y1 = drag.orig.y1+dy;
    b.x2 = drag.orig.x2+dx; b.y2 = drag.orig.y2+dy;
  } else {
    if (drag.mode.includes('w')) b.x1 = nx; else b.x2 = nx;
    if (drag.mode[0] === 'n') b.y1 = ny; else b.y2 = ny;
  }
  draw();
});
cv.addEventListener('pointerup', () => {
  if (drag) { boxes[sel] = norm(boxes[sel]); drag = null; draw(); }
});

function pushUndo() { undo.push(JSON.stringify(boxes)); if (undo.length > 50) undo.shift(); }
function doUndo() { if (undo.length) { boxes = JSON.parse(undo.pop()); sel = boxes.length-1; draw(); } }

function clampInt(b) {
  const W = view.naturalWidth, H = view.naturalHeight;
  const x1 = Math.max(0, Math.round(b.x1)), y1 = Math.max(0, Math.round(b.y1));
  const x2 = Math.min(W, Math.round(b.x2)), y2 = Math.min(H, Math.round(b.y2));
  return [x1, y1, x2, y2];
}
const valid = b => { const [x1,y1,x2,y2] = clampInt(b); return x2 > x1 && y2 > y1; };

async function loadQueue() {
  const r = await fetch('/annotate/queue.json?' + qs());
  const d = await r.json();
  queue = d.frames; counts = d.counts;
  pos = 0; show();
}
function show() {
  undo = []; suggestion = null; sel = -1;
  if (pos >= queue.length) { return showDone(); }
  $('stage').style.display = '';
  const f = queue[pos];
  boxes = (f.boxes || []).map(b => ({ x1:b[0], y1:b[1], x2:b[2], y2:b[3] }));
  if (boxes.length) sel = 0;
  $('hdr').innerHTML = '<b>' + f.rel + '</b><br>' + f.ts + ' &middot; coffee_pot' +
    (f.labeled ? ' &middot; <span style="color:#4caf50">saved</span>' : '') +
    (f.skip ? ' &middot; <span style="color:#e0a020">watched</span>' : '');
  $('count').textContent = counts.labeled + ' / ' + counts.total +
    ' labeled &middot; ' + (counts.watched || 0) + ' watched &middot; ' +
    (queue.length - pos) + ' in queue';
  view.onload = () => { fitCanvas(); if (!f.labeled && !f.skip) getSuggestion(); };
  view.src = '/annotate/frame.jpg?rel=' + encodeURIComponent(f.rel) + '&' + qs();
}
function showDone() {
  $('stage').style.display = 'none';
  $('hdr').innerHTML = '<b>queue done</b>';
  $('meta').innerHTML =
    '<div class=done>Labeled ' + counts.labeled + ' / ' + counts.total + ' frames.<br>' +
    'Build the training set:<br><button id=promote>run promote</button>' +
    '<br>or: <code>python -m coffeecam.dataset promote</code></div>';
  const p = $('promote');
  if (p) p.onclick = async () => {
    p.textContent = 'promoting…';
    const r = await fetch('/annotate/promote?confirm=1', { method:'POST' });
    const d = await r.json();
    p.outerHTML = '<div>' + (d.summary || d.error) + '</div>';
  };
}
async function getSuggestion() {
  try {
    const r = await fetch('/annotate/suggest.json?rel=' + encodeURIComponent(queue[pos].rel) + '&' + qs());
    if (!r.ok) return;
    const d = await r.json();
    suggestion = d.source === 'model' ? d.boxes : null;
    draw();
  } catch (_) {}
}
function acceptSuggestion() {
  if (!suggestion) return;
  pushUndo();
  for (const b of suggestion) boxes.push({ x1:b[0], y1:b[1], x2:b[2], y2:b[3] });
  sel = boxes.length-1; suggestion = null; draw();
}

async function post(url, body) {
  const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body) });
  return { ok: r.ok, data: await r.json().catch(() => ({})) };
}
async function save(negative) {
  const f = queue[pos];
  const payload = negative ? [] : boxes.filter(valid).map(clampInt);
  if (!negative && payload.length === 0) { $('count').textContent = 'draw a box first (or press x for negative)'; return; }
  const { ok, data } = await post('/annotate/label', { rel: f.rel, boxes: payload });
  if (!ok) { $('count').textContent = 'save failed: ' + (data.error || '?'); return; }
  if (!f.labeled) counts.labeled++;
  f.labeled = true; f.boxes = payload;
  pos++; show();
}
async function delSaved() {
  const f = queue[pos]; if (!f || !f.labeled) return;
  const { data } = await post('/annotate/label/delete', { rel: f.rel });
  if (data.removed) { counts.labeled--; f.labeled = false; f.boxes = []; boxes = []; sel = -1; show(); }
}
async function skipFrame() {
  const f = queue[pos]; if (!f) return;
  const { ok, data } = await post('/annotate/skip', { rel: f.rel });
  if (!ok) { $('count').textContent = 'skip failed: ' + (data.error || '?'); return; }
  if (!f.skip) counts.watched = (counts.watched || 0) + 1;
  f.skip = true;
  pos++; show();
}
async function skipRest() {
  const left = queue.slice(pos).filter(f => !f.labeled && !f.skip).length;
  if (!left) { $('count').textContent = 'nothing unlabeled left to skip'; return; }
  if (!confirm('Mark ' + left + '+ unlabeled frame(s) as watched? '
      + '(everything not yet labeled, not just this strided view)')) return;
  const r = await fetch('/annotate/skip-queue?' + qs(), { method:'POST' });
  const d = await r.json();
  $('count').textContent = 'marked ' + (d.skipped || 0) + ' watched';
  loadQueue();
}

$('prev').onclick = () => { if (pos > 0) { pos--; show(); } };
$('next').onclick = () => { pos++; show(); };
$('save').onclick = () => save(false);
$('neg').onclick = () => save(true);
$('skip').onclick = skipFrame;
$('skiprest').onclick = skipRest;
$('hint').onclick = () => suggestion ? acceptSuggestion() : getSuggestion();
$('clear').onclick = () => { pushUndo(); boxes = []; sel = -1; draw(); };
$('del').onclick = delSaved;
$('reload').onclick = loadQueue;
$('filter').onchange = loadQueue;
$('stride').onchange = loadQueue;

addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); save(false); }
  else if (e.key === 'x') save(true);
  else if (e.key === 's') skipFrame();
  else if (e.key === 'h') $('hint').onclick();
  else if (e.key === 'd') $('clear').onclick();
  else if (e.key === 'z') doUndo();
  else if (e.key === 'Backspace') { e.preventDefault(); delSaved(); }
  else if (e.key === 'Delete') { if (sel >= 0) { pushUndo(); boxes.splice(sel, 1); sel = -1; draw(); } }
  else if (e.key === 'ArrowLeft' && sel < 0) $('prev').onclick();
  else if (e.key === 'ArrowRight' && sel < 0) $('next').onclick();
  else if (e.key.startsWith('Arrow') && sel >= 0) {
    e.preventDefault(); pushUndo();
    const d = e.shiftKey ? 10 : 1, b = boxes[sel];
    if (e.key === 'ArrowLeft') { b.x1 -= d; b.x2 -= d; }
    if (e.key === 'ArrowRight') { b.x1 += d; b.x2 += d; }
    if (e.key === 'ArrowUp') { b.y1 -= d; b.y2 -= d; }
    if (e.key === 'ArrowDown') { b.y1 += d; b.y2 += d; }
    draw();
  }
});

loadQueue().catch(err => { $('hdr').textContent = 'load failed: ' + err; });
</script>
"""


_FULLNESS_PAGE = """<!doctype html><meta charset=utf-8><title>coffeecam fullness</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{font:14px system-ui,sans-serif;margin:0;background:#14161a;color:#e6e6e6}
 header{padding:10px 16px;background:#1d2026;border-bottom:1px solid #2c2f36}
 h1{font-size:15px;margin:0;font-weight:600} .muted{color:#8a909a} a{color:#6ab0ff}
 .wrap{max-width:1000px;margin:0 auto;padding:16px;display:flex;gap:16px;flex-wrap:wrap}
 .imgs{flex:1 1 480px;display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap}
 .imgs figure{margin:0}
 .imgs figcaption{font:12px ui-monospace,monospace;color:#8a909a;margin-bottom:4px}
 #crop{width:192px;height:192px;image-rendering:pixelated}
 #frame{max-width:360px;height:auto}
 .imgs img{display:block;background:#000;border:1px solid #2c2f36;border-radius:8px}
 .col{flex:0 0 220px;display:flex;flex-direction:column;gap:8px}
 button,select{font:13px system-ui;padding:6px 10px;background:#2c2f36;color:#e6e6e6;border:1px solid #3a3f47;border-radius:6px;cursor:pointer;text-align:left}
 button:hover,select:hover{background:#3a3f47}
 button.lvl{font-weight:600} .row{display:flex;gap:6px} .row button{flex:1}
 .k{color:#6ab0ff} .cap{font:12px ui-monospace,Menlo,monospace;color:#8a909a;word-break:break-all;line-height:1.6}
 .cap b{color:#e6e6e6}
 label.f{font-size:12px;color:#8a909a;display:flex;justify-content:space-between;gap:6px}
 label.f select{flex:1}
 .done{padding:20px;background:#1d2026;border:1px solid #2c2f36;border-radius:8px;line-height:1.8}
</style>
<header><h1>coffeecam fullness
 <span class=muted>&middot; label the <b>fill level</b> of the crop the model sees &middot;
 <a href="/annotate">/annotate</a> &middot; <a href="/fullness/compare.gif">v1 vs heuristic</a>
 &middot; <a href="/">/</a></span></h1></header>
<div class=wrap>
 <div class=imgs id=imgs>
  <figure><figcaption>model input &mdash; prepare_crop(GT box)</figcaption>
   <img id=crop alt=""></figure>
  <figure><figcaption>full frame (context)</figcaption><img id=frame alt=""></figure>
 </div>
 <div class=col>
  <div class=cap id=hdr>loading&hellip;</div>
  <div class=cap id=model>&nbsp;</div>
  <div class=cap id=count></div>
  <button class=lvl id=lvl-1>1 &mdash; empty &nbsp;<span class=k>1</span></button>
  <button class=lvl id=lvl-2>2 &mdash; low &nbsp;<span class=k>2</span></button>
  <button class=lvl id=lvl-3>3 &mdash; half &nbsp;<span class=k>3</span></button>
  <button class=lvl id=lvl-4>4 &mdash; high &nbsp;<span class=k>4</span></button>
  <button class=lvl id=lvl-5>5 &mdash; full &nbsp;<span class=k>5</span></button>
  <button class=lvl id=lvl-absent>no pot in frame &nbsp;<span class=k>w</span></button>
  <button class=lvl id=lvl-unsure>unsure / ambiguous &nbsp;<span class=k>u</span></button>
  <button id=skip>skip / watched &nbsp;<span class=k>s</span></button>
  <button id=del>delete saved label &nbsp;<span class=k>&#9003;</span></button>
  <div class=cap muted style="line-height:1.5">
   Crop loose but on the carafe &rarr; judge from whichever image is clearer.
   Crop on the wrong thing (wall, mug) &rarr; fix the box in <a href="/annotate">/annotate</a>
   or <b>skip</b> &mdash; don't label it. <b>skip</b> also when neither image is
   legible. <b>no pot</b> = carafe genuinely off the warmer. <b>unsure</b> = the
   crop is legible but even you can't call the level (glare, odd angle, blur,
   mid-pour) &mdash; trains the model to say "unsure" instead of guessing.</div>
  <div class=row><button id=prev>&larr; prev</button><button id=next>next &rarr;</button></div>
  <button id=skiprest>skip rest of queue</button>
  <button id=reload>reload queue</button>
  <label class=f>filter <select id=filter>
    <option value=unlabeled selected>unlabeled</option>
    <option value=labeled>labeled</option>
    <option value=watched>watched</option>
    <option value=all>all</option></select></label>
  <label class=f>stride <select id=stride>
    <option>1</option><option>2</option><option>3</option><option>5</option><option>10</option>
   </select></label>
  <label class=f><input type=checkbox id=showmodel checked> show model prediction</label>
  <label class=f><input type=checkbox id=sortconf> sort by lowest confidence
   <span class=muted>(annotate these first)</span></label>
 </div>
</div>
<script>
const params = new URLSearchParams(location.search);
const $ = id => document.getElementById(id);
let queue = [], counts = {}, pos = 0;
// 1-5 scale; index 0 unused so n maps straight to LEVELS[n]. 'absent' (no pot in
// frame) and 'unsure' (legible but unjudgeable) are separate labels, keyed 'w'
// and 'u', not part of the scale.
const LEVELS = [null, 'empty', 'low', 'half', 'high', 'full'];
const ABSENT = 'absent';
const UNSURE = 'unsure';
const DOT = ' \\u00b7 ';

const qs = () => {
  const p = new URLSearchParams();
  p.set('filter', $('filter').value);
  p.set('stride', $('stride').value);
  if (params.get('start')) p.set('start', params.get('start'));
  return p;
};
// Separate from qs(): the bulk model pass is opt-in (checkbox) and only
// belongs on /fullness/queue.json, not the per-frame crop/frame image URLs.
const queueQs = () => {
  const p = qs();
  if ($('sortconf').checked) { p.set('model', '1'); p.set('sort', 'confidence'); }
  return p;
};

async function post(url, body) {
  const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body) });
  return { ok: r.ok, data: await r.json().catch(() => ({})) };
}

async function loadQueue() {
  const r = await fetch('/fullness/queue.json?' + queueQs());
  const d = await r.json();
  queue = d.frames; counts = d.counts; pos = 0; show();
}
function fmtLevel(level) {
  const n = LEVELS.indexOf(level);
  return !level ? '' : (n > 0 ? n + ' \\u2014 ' + level : level);
}
async function showModel(f) {
  const el = $('model');
  if (!f || !$('showmodel').checked) { el.innerHTML = '&nbsp;'; return; }
  // The bulk queue pass (sort-by-confidence) already carries model_level/
  // model_conf for this frame; reuse it instead of a second request.
  if (f.model_level !== undefined) {
    return renderModel(f.model_level, f.model_conf, null, f.level);
  }
  el.textContent = 'model: \\u2026';
  try {
    const r = await fetch('/fullness/suggest.json?rel=' + encodeURIComponent(f.rel));
    const d = await r.json();
    if (f !== queue[pos]) return;  // stale response from a fast prev/next
    if (d.error) { el.textContent = 'model: ' + d.error; return; }
    const top = d.probs ? Math.max(...Object.values(d.probs)) : null;
    renderModel(d.level, top, d.probs, f.level);
  } catch (err) { if (f === queue[pos]) el.textContent = 'model: request failed'; }
}
function renderModel(level, conf, probs, savedLevel) {
  const agree = savedLevel && level === savedLevel;
  const color = !savedLevel ? '#8a909a' : (agree ? '#4caf50' : '#e0562a');
  const confTxt = conf == null ? '' : ' (' + (conf * 100).toFixed(0) + '%)';
  let html = 'model: <span style="color:' + color + '">' + fmtLevel(level) + confTxt + '</span>';
  if (probs) {
    html += '<br><span class=muted>' + Object.entries(probs)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => k + ' ' + (v * 100).toFixed(0) + '%').join(' &middot; ') + '</span>';
  }
  $('model').innerHTML = html;
}
function show() {
  if (pos >= queue.length) return showDone();
  $('imgs').style.display = '';
  const f = queue[pos];
  const lvlName = fmtLevel(f.level);
  $('hdr').innerHTML = '<b>' + f.rel + '</b>' +
    (f.level ? DOT + '<span style="color:#4caf50">' + lvlName + '</span>' : '') +
    (f.skip ? DOT + '<span style="color:#e0a020">watched</span>' : '');
  $('count').innerHTML =
    counts.labeled + ' / ' + counts.total + ' labeled<br>' +
    (counts.watched || 0) + ' watched<br>' +
    (queue.length - pos) + ' in queue';
  // Blank first so a stale crop never lingers under the next frame's label.
  $('crop').removeAttribute('src'); $('frame').removeAttribute('src');
  $('crop').src = '/fullness/crop.jpg?rel=' + encodeURIComponent(f.rel) + '&' + qs();
  $('frame').src = '/fullness/frame.jpg?rel=' + encodeURIComponent(f.rel) + '&' + qs();
  showModel(f);
}
function showDone() {
  $('imgs').style.display = 'none';
  $('model').innerHTML = '&nbsp;';
  $('hdr').innerHTML = '<b>queue done</b>';
  $('count').innerHTML = '<div class=done>Labeled ' + counts.labeled + ' / ' +
    counts.total + ' box-positive frames.<br>' +
    'Class counts: ' + LEVELS.slice(1).map(
      (l, i) => (i + 1) + '/' + l + ' ' + (counts[l] || 0)).join(' &middot; ') +
    ' &middot; ' + ABSENT + ' ' + (counts[ABSENT] || 0) +
    ' &middot; ' + UNSURE + ' ' + (counts[UNSURE] || 0) +
    '<br>Next: <code>python -m coffeecam.fullness_dataset</code></div>';
}
async function label(level) {
  const f = queue[pos]; if (!f) return;
  const { ok, data } = await post('/fullness/label', { rel: f.rel, level });
  if (!ok) { $('count').textContent = 'save failed: ' + (data.error || '?'); return; }
  if (!f.level) counts.labeled++;
  f.level = level; f.skip = false;
  pos++; show();
}
async function skipFrame() {
  const f = queue[pos]; if (!f) return;
  const { ok, data } = await post('/fullness/skip', { rel: f.rel });
  if (!ok) { $('count').textContent = 'skip failed: ' + (data.error || '?'); return; }
  if (!f.skip) counts.watched = (counts.watched || 0) + 1;
  f.skip = true; pos++; show();
}
async function delSaved() {
  const f = queue[pos]; if (!f || (!f.level && !f.skip)) return;
  const { data } = await post('/fullness/label/delete', { rel: f.rel });
  if (data.removed) { if (f.level) counts.labeled--; f.level = ''; f.skip = false; show(); }
}
async function skipRest() {
  const left = queue.slice(pos).filter(f => !f.level && !f.skip).length;
  if (!left) { $('count').textContent = 'nothing unlabeled left to skip'; return; }
  if (!confirm('Mark ' + left + '+ unlabeled frame(s) as watched?')) return;
  const r = await fetch('/fullness/skip-queue?' + qs(), { method:'POST' });
  const d = await r.json();
  $('count').textContent = 'marked ' + (d.skipped || 0) + ' watched';
  loadQueue();
}

for (let n = 1; n <= 5; n++) $('lvl-' + n).onclick = () => label(LEVELS[n]);
$('lvl-absent').onclick = () => label(ABSENT);
$('lvl-unsure').onclick = () => label(UNSURE);
$('skip').onclick = skipFrame;
$('del').onclick = delSaved;
$('prev').onclick = () => { if (pos > 0) { pos--; show(); } };
$('next').onclick = () => { pos++; show(); };
$('skiprest').onclick = skipRest;
$('reload').onclick = loadQueue;
$('filter').onchange = loadQueue;
$('stride').onchange = loadQueue;
$('sortconf').onchange = loadQueue;
$('showmodel').onchange = () => showModel(queue[pos]);

addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key >= '1' && e.key <= '5') label(LEVELS[+e.key]);
  else if (e.key === 'w') label(ABSENT);
  else if (e.key === 'u') label(UNSURE);
  else if (e.key === 's') skipFrame();
  else if (e.key === 'Backspace') { e.preventDefault(); delSaved(); }
  else if (e.key === 'ArrowLeft') $('prev').onclick();
  else if (e.key === 'ArrowRight') $('next').onclick();
});

loadQueue().catch(err => { $('hdr').textContent = 'load failed: ' + err; });
</script>
"""


_VIEWER_PAGE = """<!doctype html><meta charset=utf-8><title>coffeecam viewer</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{font:14px system-ui,sans-serif;margin:0;background:#14161a;color:#e6e6e6}
 header{padding:10px 16px;background:#1d2026;border-bottom:1px solid #2c2f36}
 h1{font-size:15px;margin:0;font-weight:600}
 .wrap{max-width:900px;margin:0 auto;padding:16px}
 img{display:block;width:100%;height:auto;background:#000;border:1px solid #2c2f36;border-radius:8px}
 .bar{display:flex;align-items:center;gap:12px;margin:12px 0}
 button,select{font:14px system-ui;padding:6px 12px;background:#2c2f36;color:#e6e6e6;border:1px solid #3a3f47;border-radius:6px;cursor:pointer}
 button:hover,select:hover{background:#3a3f47}
 input[type=range]{flex:1}
 .cap{font:12px ui-monospace,Menlo,monospace;color:#8a909a;word-break:break-all;line-height:1.7}
 .cap b{color:#e6e6e6}
 .badge{display:inline-block;padding:1px 7px;border-radius:4px;font-size:11px;font-weight:600}
 .strong{background:#c82020;color:#fff} .weak_only{background:#e0a020;color:#141414} .no_box{background:#333;color:#bbb}
 .muted{color:#8a909a} a{color:#6ab0ff}
</style>
<header><h1>coffeecam viewer
 <span class=muted>&middot; scrubbable capture timelapse &middot;
 <a href="/summary">/summary</a> (gif) &middot; <a href="/">/</a> (live)</span></h1></header>
<div class=wrap>
 <img id=view alt="">
 <div class=bar>
  <button id=play>&#9654; Play</button>
  <input type=range id=slider min=1 max=1 value=1>
  <span class=muted id=counter>&ndash;</span>
  <select id=speed>
   <option value=250>slow</option>
   <option value=120 selected>120ms</option>
   <option value=60>fast</option>
  </select>
 </div>
 <div class=cap id=cap>loading&hellip;</div>
 <div class="cap muted" id=meta></div>
 <div class=muted style="font-size:12px;margin-top:8px">&larr;/&rarr; step &middot; space play/pause</div>
</div>
<script>
const params = new URLSearchParams(location.search);
params.delete('rebuild');
const qs = params.toString() ? '?' + params.toString() : '';
const $ = id => document.getElementById(id);
const view = $('view'), slider = $('slider'), playBtn = $('play'), speed = $('speed');
const counter = $('counter'), cap = $('cap'), metaEl = $('meta');
let frames = [], timer = null, cur = 1;

const frameURL = i => '/viewer/frame/' + i + '.jpg' + qs;

function render(i){
  cur = Math.min(Math.max(1, i), frames.length || 1);
  slider.value = cur;
  const f = frames[cur - 1];
  if(!f) return;
  view.src = frameURL(cur);
  counter.textContent = cur + ' / ' + frames.length;
  cap.innerHTML = '<b>' + f.path + '</b><br>' + f.ts +
    ' &middot; <span class="badge ' + f.kind + '">' + f.kind.replace('_', ' ') + '</span>';
}
function stop(){ if(timer){ clearInterval(timer); timer = null; } playBtn.innerHTML = '&#9654; Play'; }
function play(){
  if(timer || !frames.length) return;
  playBtn.innerHTML = '&#10073;&#10073; Pause';
  timer = setInterval(() => render(cur >= frames.length ? 1 : cur + 1), +speed.value);
}
playBtn.onclick = () => timer ? stop() : play();
slider.oninput = () => { stop(); render(+slider.value); };
speed.onchange = () => { if(timer){ stop(); play(); } };
addEventListener('keydown', e => {
  if(e.key === 'ArrowRight'){ stop(); render(cur + 1); }
  else if(e.key === 'ArrowLeft'){ stop(); render(cur - 1); }
  else if(e.key === ' '){ e.preventDefault(); playBtn.onclick(); }
});

fetch('/viewer/manifest.json' + location.search).then(r => r.ok
  ? r.json().then(d => {
      frames = d.frames;
      slider.max = frames.length;
      frames.forEach(f => { new Image().src = frameURL(f.i); });  // warm the cache
      const m = d.meta;
      metaEl.textContent = m.rendered_frames + ' of ' + m.source_frames + ' frames · ' +
        m.span[0] + '  →  ' + m.span[1] +
        (m.detections ? ' · strong ' + m.detections.strong + ' / weak ' +
          m.detections.weak_only + ' / none ' + m.detections.no_box : '');
      render(1);
    })
  : r.text().then(t => { cap.textContent = 'no captures yet: ' + t; }));
</script>
"""


_HISTORY_PAGE = """<!doctype html><meta charset=utf-8><title>coffeecam history</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{font:14px system-ui,sans-serif;margin:0;background:#14161a;color:#e6e6e6}
 header{padding:10px 16px;background:#1d2026;border-bottom:1px solid #2c2f36}
 h1{font-size:15px;margin:0;font-weight:600} .muted{color:#8a909a} a{color:#6ab0ff}
 .wrap{max-width:1000px;margin:0 auto;padding:16px}
 select{font:13px system-ui;padding:5px 8px;background:#2c2f36;color:#e6e6e6;border:1px solid #3a3f47;border-radius:6px}
 .bar{display:flex;height:46px;width:100%;border:1px solid #2c2f36;border-radius:8px;overflow:hidden;margin:14px 0}
 .seg{position:relative;min-width:2px}
 .seg span{position:absolute;left:3px;top:2px;font:10px ui-monospace,monospace;color:#0b0d10;white-space:nowrap}
 .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#8a909a;margin-bottom:8px}
 .legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:4px;vertical-align:-1px}
 table{border-collapse:collapse;width:100%;font:12px ui-monospace,Menlo,monospace}
 th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #2c2f36;vertical-align:top}
 th{color:#8a909a;font-weight:600}
 td img{display:block;width:150px;height:auto;border:1px solid #2c2f36;border-radius:4px}
 .lvl{font-weight:600}
</style>
<header><h1>coffeecam history
 <span class=muted>&middot; fullness state timeline &middot;
 <a href="/viewer">/viewer</a> &middot; <a href="/history/long">/history/long</a> (N-day graph) &middot;
 <a href="/history.json">json</a> &middot; <a href="/">/</a></span></h1></header>
<div class=wrap>
 <label class=muted>day <select id=date></select></label>
 <span class=muted id=summary></span>
 <div class=legend id=legend></div>
 <div class=bar id=bar></div>
 <table><thead><tr><th>transition</th><th>at</th><th>held</th><th>frames</th><th>frame</th></tr></thead>
  <tbody id=rows></tbody></table>
 <p class=muted id=empty hidden>No state history for this day yet.</p>
</div>
<script>
const $ = id => document.getElementById(id);
const COLORS = {empty:'#5b6472', low:'#c98b2e', some:'#c98b2e', half:'#d9c04a',
                high:'#7bc46b', lots:'#3f9e57', full:'#3f9e57',
                unknown:'#3a3f47', absent:'#2c2f36', error:'#c04040'};
const fmt = s => s ? s.slice(11, 19) : '?';
const held = s => s == null ? '?' : s >= 3600 ? (s/3600).toFixed(1)+'h'
  : s >= 60 ? (s/60).toFixed(1)+'m' : s.toFixed(0)+'s';

async function load(date) {
  const q = date ? '?date=' + encodeURIComponent(date) : '';
  const d = await (await fetch('/history.json' + q)).json();
  const sel = $('date');
  sel.innerHTML = (d.dates || []).map(x =>
    '<option' + (x === d.date ? ' selected' : '') + '>' + x + '</option>').join('');
  const runs = d.runs || [];
  $('empty').hidden = runs.length > 0;
  $('summary').textContent = runs.length
    ? ' \\u00b7 ' + d.rows + ' ticks \\u00b7 ' + runs.length + ' segments' : '';
  const seen = [...new Set(runs.map(r => r.level))];
  $('legend').innerHTML = seen.map(l =>
    '<span><i style="background:' + (COLORS[l] || '#666') + '"></i>' + l + '</span>').join('');
  const total = runs.reduce((a, r) => a + (r.duration_s || 0), 0) || 1;
  $('bar').innerHTML = runs.map(r => {
    const w = ((r.duration_s || 0) / total * 100).toFixed(3);
    return '<div class=seg title="' + r.level + '  ' + fmt(r.start) + ' \\u2192 ' + fmt(r.end) +
      '  (' + held(r.duration_s) + ')" style="width:' + w + '%;background:' +
      (COLORS[r.level] || '#666') + '"><span>' + r.level + '</span></div>';
  }).join('');
  $('rows').innerHTML = runs.map((r, i) => {
    const prev = i ? runs[i - 1].level : '\\u2014';
    const img = r.artifact
      ? '<a href="/history/artifact/' + r.artifact + '"><img loading=lazy src="/history/artifact/' +
        r.artifact + '"></a>' : '';
    return '<tr><td class=lvl>' + prev + ' \\u2192 <span style="color:' +
      (COLORS[r.level] || '#aaa') + '">' + r.level + '</span></td><td>' + fmt(r.start) +
      '</td><td>' + held(r.duration_s) + '</td><td>' + r.frames + '</td><td>' + img + '</td></tr>';
  }).join('');
}
$('date').onchange = e => load(e.target.value);
load().catch(err => { $('summary').textContent = 'load failed: ' + err; });
</script>
"""


_LONGHISTORY_PAGE = """<!doctype html><meta charset=utf-8><title>coffeecam history (long)</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
 body{font:14px system-ui,sans-serif;margin:0;background:#14161a;color:#e6e6e6}
 header{padding:10px 16px;background:#1d2026;border-bottom:1px solid #2c2f36}
 h1{font-size:15px;margin:0;font-weight:600} .muted{color:#8a909a} a{color:#6ab0ff}
 .wrap{max-width:1180px;margin:0 auto;padding:16px}
 select{font:13px system-ui;padding:5px 8px;background:#2c2f36;color:#e6e6e6;border:1px solid #3a3f47;border-radius:6px}
 label.c{font-size:12px;color:#8a909a;margin-left:14px}
 .chart{position:relative;margin:14px 0;background:#1a1d22;border:1px solid #2c2f36;border-radius:8px}
 .chart svg{display:block;width:100%;height:auto}
 .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:#8a909a}
 .legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:4px;vertical-align:-1px}
 #readout{position:absolute;pointer-events:none;background:#0e1013;border:1px solid #3a3f47;border-radius:6px;
   padding:6px 8px;font:12px ui-monospace,Menlo,monospace;white-space:pre;display:none;z-index:2}
 text{fill:#8a909a;font:11px ui-monospace,monospace}
 .gl{stroke:#2c2f36} .day{stroke:#3a3f47;stroke-dasharray:3 3}
</style>
<header><h1>coffeecam history <span class=muted>(long)</span>
 <span class=muted>&middot; fullness over the last N days &middot;
 <a href="/history">/history</a> (single day) &middot; <a href="/history/long.json">json</a> &middot;
 <a href="/">/</a></span></h1></header>
<div class=wrap>
 <label class=muted>days <select id=days>
   <option>3</option><option selected>7</option><option>14</option><option>30</option></select></label>
 <label class=c><input type=checkbox id=conf> confidence (p)</label>
 <span class=muted id=summary></span>
 <div class=chart id=chart><div id=readout></div></div>
 <div class=legend id=legend></div>
</div>
<script>
const $ = id => document.getElementById(id);
const COLORS = {empty:'#5b6472', low:'#c98b2e', some:'#c98b2e', half:'#d9c04a',
                high:'#7bc46b', lots:'#3f9e57', full:'#3f9e57',
                unknown:'#3a3f47', absent:'#2c2f36', error:'#c04040'};
const SVGNS = 'http://www.w3.org/2000/svg';
const W = 1160, H = 360, L = 44, R = 14, T = 14, B = 44;
const params = new URLSearchParams(location.search);
if (params.get('days')) $('days').value = params.get('days');

const el = (n, a) => { const e = document.createElementNS(SVGNS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; };

let pts = [], dates = [], geom = null;

async function load() {
  const days = $('days').value;
  history.replaceState(null, '', '?days=' + days);
  const d = await (await fetch('/history/long.json?days=' + days)).json();
  pts = d.points || []; dates = d.dates || [];
  $('summary').textContent = ' \\u00b7 ' + pts.length + ' points over ' + dates.length + ' days';
  const seen = [...new Set(pts.map(p => p.l))];
  $('legend').innerHTML = seen.map(l =>
    '<span><i style="background:' + (COLORS[l] || '#666') + '"></i>' + l + '</span>').join('') +
    '<span class=muted>&nbsp; y = fill 0..1 (score); each column = one day, 00:00\\u201324:00</span>';
  draw();
}

function draw() {
  const c = $('chart');
  [...c.querySelectorAll('svg')].forEach(s => s.remove());
  const svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H });
  const nd = dates.length || 1;
  const colW = (W - L - R) / nd;
  const x = (date, sec) => L + dates.indexOf(date) * colW + (sec / 86400) * colW;
  const y = s => T + (1 - s) * (H - T - B);
  geom = { x, y, colW };

  // y gridlines at the fill anchors
  for (const [s, name] of [[0,'empty'],[0.33,'some'],[0.5,'half'],[0.83,'lots'],[1,'full']]) {
    svg.appendChild(el('line', { class:'gl', x1:L, x2:W-R, y1:y(s), y2:y(s) }));
    const tx = el('text', { x:4, y:y(s)+3 }); tx.textContent = name; svg.appendChild(tx);
  }
  // day columns + labels
  dates.forEach((dt, i) => {
    const gx = L + i * colW;
    if (i) svg.appendChild(el('line', { class:'day', x1:gx, x2:gx, y1:T, y2:H-B }));
    const tx = el('text', { x:gx+4, y:H-B+16 }); tx.textContent = dt.slice(5); svg.appendChild(tx);
    const noon = el('text', { x:gx+colW/2-14, y:H-B+30 }); noon.textContent = '12:00';
    noon.setAttribute('opacity', 0.5); svg.appendChild(noon);
  });

  // stepped score path, broken across day boundaries and null scores
  let dpath = '', started = false, prevDate = null;
  for (const p of pts) {
    if (p.s == null) { started = false; continue; }
    const px = x(p.date, p.tod), py = y(p.s);
    if (!started || p.date !== prevDate) { dpath += ' M' + px + ' ' + py; started = true; }
    else { dpath += ' H' + px + ' V' + py; }
    prevDate = p.date;
  }
  svg.appendChild(el('path', { d:dpath, fill:'none', stroke:'#6ab0ff', 'stroke-width':1.5 }));

  if ($('conf').checked) {
    let cp = '', on = false;
    for (const p of pts) {
      if (p.p == null) { on = false; continue; }
      const px = x(p.date, p.tod), py = y(p.p);
      cp += (on ? ' L' : ' M') + px + ' ' + py; on = true;
    }
    svg.appendChild(el('path', { d:cp, fill:'none', stroke:'#e0a020',
      'stroke-width':1, opacity:0.55 }));
  }

  // dots
  for (const p of pts) {
    if (p.s == null) continue;
    svg.appendChild(el('circle', { cx:x(p.date, p.tod), cy:y(p.s), r:2,
      fill:COLORS[p.l] || '#888' }));
  }

  const hit = el('rect', { x:L, y:T, width:W-L-R, height:H-T-B, fill:'transparent' });
  svg.appendChild(hit);
  const guide = el('line', { class:'gl', y1:T, y2:H-B, stroke:'#6ab0ff', opacity:0 });
  svg.appendChild(guide);
  c.appendChild(svg);

  const ro = $('readout');
  svg.addEventListener('mousemove', ev => {
    const r = svg.getBoundingClientRect();
    const mx = (ev.clientX - r.left) / r.width * W;
    let best = null, bd = 1e9;
    for (const p of pts) {
      if (p.s == null) continue;
      const d = Math.abs(geom.x(p.date, p.tod) - mx);
      if (d < bd) { bd = d; best = p; }
    }
    if (!best || bd > geom.colW) { ro.style.display = 'none'; guide.setAttribute('opacity', 0); return; }
    const gx = geom.x(best.date, best.tod);
    guide.setAttribute('x1', gx); guide.setAttribute('x2', gx); guide.setAttribute('opacity', 0.5);
    ro.textContent = best.t.replace('T', '  ') + '\\n' + best.l +
      '   score ' + (best.s == null ? '-' : best.s.toFixed(2)) +
      '   p ' + (best.p == null ? '-' : best.p.toFixed(2));
    ro.style.display = 'block';
    ro.style.left = Math.min(ev.clientX - r.left + 12, r.width - 190) + 'px';
    ro.style.top = (ev.clientY - r.top + 12) + 'px';
  });
  svg.addEventListener('mouseleave', () => {
    ro.style.display = 'none'; guide.setAttribute('opacity', 0);
  });
}

$('days').onchange = load;
$('conf').onchange = draw;
load().catch(err => { $('summary').textContent = 'load failed: ' + err; });
</script>
"""


_PAGE = """<!doctype html><meta charset=utf-8><title>coffeecam pipeline</title>
<meta http-equiv=refresh content="{refresh}">
<style>
 body{{font:14px system-ui,sans-serif;margin:0;background:#14161a;color:#e6e6e6}}
 header{{padding:12px 16px;background:#1d2026;border-bottom:1px solid #2c2f36}}
 h1{{font-size:15px;margin:0;font-weight:600}}
 .verdict{{font-size:22px;margin-top:4px}} .muted{{color:#8a909a}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;padding:16px}}
 figure{{margin:0;background:#1d2026;border:1px solid #2c2f36;border-radius:8px;overflow:hidden}}
 figcaption{{padding:6px 10px;font-size:12px;color:#8a909a;border-bottom:1px solid #2c2f36}}
 img{{display:block;width:100%;height:auto;background:#000}}
 pre{{padding:0 16px 20px;color:#8a909a;white-space:pre-wrap;font-size:12px}}
 .err{{color:#ff6b6b}}
</style>
<header>
 <h1>coffeecam pipeline <span class=muted>· refresh {refresh}s · updated {ts} ({stale}s ago)</span></h1>
 <div class=verdict>fullness: <b>{level}</b> <span class=muted>{score} · {method}</span></div>
 {errline}
</header>
<div class=grid>
 <figure><figcaption>1 · acquire (rotate 180 + privacy crop)</figcaption><img src="/frame.jpg?t={ts}"></figure>
 <figure><figcaption>2 · normalize (black-pad to training aspect)</figcaption><img src="/normalized.jpg?t={ts}"></figure>
 <figure><figcaption>3 · detect{detconf}</figcaption><img src="/bounded.jpg?t={ts}"></figure>
 <figure><figcaption>4 · crop &rarr; classify</figcaption><img src="/crop.jpg?t={ts}"></figure>
</div>
<pre>timings_ms: {timings}
{jsonlink} · <a href="/fullness.json" style="color:#6ab0ff">/fullness.json</a> · <a href="/summary" style="color:#6ab0ff">/summary</a> (annotated capture timelapse) · <a href="/compare" style="color:#6ab0ff">/compare</a> (old vs new detector) · <a href="/viewer" style="color:#6ab0ff">/viewer</a> (scrubbable) · <a href="/history" style="color:#6ab0ff">/history</a> (state timeline) · <a href="/history/long" style="color:#6ab0ff">/history/long</a> (N-day graph) · <a href="/annotate" style="color:#6ab0ff">/annotate</a> (label frames) · <a href="/fullness" style="color:#6ab0ff">/fullness</a> (label fill level) · <a href="/artifacts" style="color:#6ab0ff">/artifacts</a> (scratch gallery)</pre>
"""


def create_app(start_worker: bool = True) -> Flask:
    app = Flask(__name__)
    if start_worker:
        _start_worker()

    def _serve_jpeg(key: str):
        with _lock:
            data = _latest_jpeg.get(key)
        if data is None:
            return Response("not available", status=404, mimetype="text/plain")
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/frame.jpg")
    def frame_jpg():
        return _serve_jpeg("frame")

    @app.get("/normalized.jpg")
    def normalized_jpg():
        return _serve_jpeg("normalized")

    @app.get("/bounded.jpg")
    def bounded_jpg():
        return _serve_jpeg("bounded")

    @app.get("/crop.jpg")
    def crop_jpg():
        return _serve_jpeg("crop")

    @app.get("/pipeline.json")
    def pipeline_json():
        with _lock:
            result = _latest
        if result is None:
            return jsonify({"status": "warming-up", "fetch_error": _last_fetch_error}), 503
        return jsonify(_result_dict(result))

    @app.get("/fullness.json")
    def fullness_json():
        with _lock:
            result = _latest
        if result is None:
            return jsonify({"level": "unknown", "score": None, "method": "warming-up"}), 503
        f = result.fullness
        return jsonify(
            {"level": f.level, "score": f.score, "method": f.method, "detail": f.detail,
             "ts": result.ts.isoformat(timespec="seconds")}
        )

    @app.get("/healthz")
    def healthz():
        with _lock:
            result = _latest
        refresh = _cfg()["refresh"]
        if result is None:
            return jsonify({"status": "warming-up", "fetch_error": _last_fetch_error}), 503
        stale = (datetime.now() - result.ts).total_seconds()
        degraded = stale > max(3 * refresh, 30) or _last_fetch_error is not None
        body = {
            "status": "degraded" if degraded else "ok",
            "stale_seconds": round(stale, 1),
            "fetch_error": _last_fetch_error,
            "pipeline_errors": result.errors,
        }
        return jsonify(body), (503 if degraded else 200)

    @app.get("/summary")
    @app.get("/summary.gif")
    def summary_gif():
        try:
            gif, _ = _build_summary(**_summary_args())
        except SummaryEmpty as exc:
            return Response(str(exc), status=404, mimetype="text/plain")
        return Response(gif, mimetype="image/gif", headers={"Cache-Control": "no-store"})

    @app.get("/summary.json")
    def summary_json():
        try:
            _, meta = _build_summary(**_summary_args())
        except SummaryEmpty as exc:
            return jsonify({"error": str(exc)}), 404
        return jsonify(meta)

    @app.get("/compare")
    @app.get("/compare.gif")
    def compare_gif():
        try:
            gif, _ = _build_compare(**_compare_args())
        except SummaryEmpty as exc:
            return Response(str(exc), status=404, mimetype="text/plain")
        except (FileNotFoundError, ValueError) as exc:
            return Response(str(exc), status=400, mimetype="text/plain")
        return Response(gif, mimetype="image/gif", headers={"Cache-Control": "no-store"})

    @app.get("/compare.json")
    def compare_json():
        try:
            _, meta = _build_compare(**_compare_args())
        except SummaryEmpty as exc:
            return jsonify({"error": str(exc)}), 404
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(meta)

    @app.get("/fullness/compare.gif")
    def fullness_compare_gif():
        from coffeecam.fullness_compare import NoTestData, build_test_gif

        try:
            gif, _ = build_test_gif()
        except NoTestData as exc:
            return Response(str(exc), status=404, mimetype="text/plain")
        return Response(gif, mimetype="image/gif", headers={"Cache-Control": "no-store"})

    @app.get("/fullness/compare.json")
    def fullness_compare_json():
        from coffeecam.fullness_compare import NoTestData, build_test_gif

        try:
            _, scoreboard = build_test_gif()
        except NoTestData as exc:
            return jsonify({"error": str(exc)}), 404
        return jsonify(scoreboard)

    # --- /artifacts: read-only gallery of a scratch dir, no restart to add files ---

    @app.get("/artifacts")
    @app.get("/artifacts/")
    def artifacts_index():
        base = _artifacts_dir()
        if not base.is_dir():
            return Response(f"{base} does not exist", status=404, mimetype="text/plain")
        rows = []
        for p in sorted(base.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not p.is_file() or p.suffix.lower() not in _ARTIFACT_SUFFIXES:
                continue
            rel = p.relative_to(base).as_posix()
            kb = p.stat().st_size / 1024
            when = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            thumb = (f'<img src="/artifacts/{rel}" loading=lazy>'
                     if p.suffix.lower() in _ARTIFACT_IMG else "")
            rows.append(
                f'<figure><a href="/artifacts/{rel}">{thumb}<figcaption>{rel}</figcaption></a>'
                f'<small>{kb:,.0f} KB &middot; {when}</small></figure>'
            )
        body = "\n".join(rows) or "<p class=muted>no artifacts yet</p>"
        return Response(_ARTIFACTS_PAGE.format(dir=base, body=body), mimetype="text/html")

    @app.get("/artifacts/<path:name>")
    def artifacts_file(name: str):
        base = _artifacts_dir()
        if Path(name).suffix.lower() not in _ARTIFACT_SUFFIXES:
            return Response("unsupported file type", status=415, mimetype="text/plain")
        try:
            return send_from_directory(base, name, max_age=0)  # 404s safely on traversal
        except NotADirectoryError:
            return Response("not found", status=404, mimetype="text/plain")

    @app.get("/viewer")
    def viewer_page():
        return Response(_VIEWER_PAGE, mimetype="text/html")

    @app.get("/viewer/manifest.json")
    def viewer_manifest():
        try:
            build = _build_viewer(**_viewer_args())
        except SummaryEmpty as exc:
            return jsonify({"error": str(exc)}), 404
        return jsonify({"frames": build.manifest, "meta": build.meta})

    @app.get("/viewer/frame/<int:i>.jpg")
    def viewer_frame(i: int):
        args = {**_viewer_args(), "force": False}  # never rebuild on a per-frame fetch
        try:
            build = _build_viewer(**args)
        except SummaryEmpty as exc:
            return Response(str(exc), status=404, mimetype="text/plain")
        try:
            data = build.frame(i)
        except IndexError:
            return Response("frame out of range", status=404, mimetype="text/plain")
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=60"})

    # --- /history: persisted fullness-state timeline ----------------------

    @app.get("/history")
    def history_page():
        return Response(_HISTORY_PAGE, mimetype="text/html")

    @app.get("/history.json")
    def history_json():
        captures = _annot_captures_dir()
        dates = state_history.available_dates(captures)
        date = request.args.get("date") or (dates[-1] if dates else None)
        limit = _arg_int("limit", 0) or None
        rows = state_history.load_rows(captures, date=date, limit=limit)
        return jsonify({
            "date": date,
            "dates": dates,
            "rows": len(rows),
            "runs": state_history.runs_from_rows(rows),
        })

    @app.get("/history/long")
    def history_long_page():
        return Response(_LONGHISTORY_PAGE, mimetype="text/html")

    @app.get("/history/long.json")
    def history_long_json():
        """Fullness points across the last ?days= logged days (default 7, capped
        60). ?max= strides the payload down (default 4000, transitions kept).
        Each point: ``{t, date, tod (secs past midnight), s (score), l, p}``."""
        days = max(1, min(_arg_int("days", 7), 60))
        cap = _arg_int("max", 4000)
        dates, rows = state_history.load_span(
            _annot_captures_dir(), days=days, max_points=cap or None
        )
        pts = []
        for r in rows:
            ts = r.get("ts")
            if not ts:
                continue
            t = ts.split("T", 1)
            tod = 0
            if len(t) == 2:
                hh, mm, ss = (t[1].split(":") + ["0", "0", "0"])[:3]
                tod = int(hh) * 3600 + int(mm) * 60 + int(float(ss))
            pts.append({
                "t": ts, "date": r.get("_date", t[0]), "tod": tod,
                "s": r.get("score"), "l": r.get("level"), "p": r.get("p"),
            })
        return jsonify({"days": days, "dates": dates, "points": pts})

    @app.get("/history/rows.json")
    def history_rows_json():
        """Raw per-tick rows (no run collapsing) for the given ?date=."""
        captures = _annot_captures_dir()
        date = request.args.get("date")
        limit = _arg_int("limit", 0) or None
        return jsonify({"rows": state_history.load_rows(captures, date=date, limit=limit)})

    @app.get("/history/artifact/<path:name>")
    def history_artifact(name: str):
        if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".json"}:
            return Response("unsupported file type", status=415, mimetype="text/plain")
        try:
            return send_from_directory(_pipeline_dir().resolve(), name, max_age=0)
        except NotADirectoryError:
            return Response("not found", status=404, mimetype="text/plain")

    # --- /annotate labeling endpoint ---------------------------------------

    @app.get("/annotate")
    def annotate_page():
        return Response(_ANNOTATE_PAGE, mimetype="text/html")

    @app.get("/annotate/queue.json")
    def annotate_queue():
        frames, counts, _, _ = _annot_queue()
        return jsonify({"frames": frames, "counts": counts})

    @app.get("/annotate/frame.jpg")
    def annotate_frame():
        # Addressed by ``rel`` (not a positional queue index): the browser froze
        # its queue at load and every save shifts the ``unlabeled`` indices, so a
        # bare ``i`` served the wrong frame. ``no-store`` because ``rel`` is a
        # stable key we must never serve a stale image for.
        path = _safe_capture_path(request.args.get("rel", ""))
        if path is None:
            return Response("bad rel", status=400, mimetype="text/plain")
        if not path.exists():
            return Response("frame gone", status=404, mimetype="text/plain")
        return Response(
            path.read_bytes(),
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/annotate/suggest.json")
    def annotate_suggest():
        if _model is None:
            return jsonify({"error": "detector not loaded"}), 503
        path = _safe_capture_path(request.args.get("rel", ""))
        if path is None or not path.exists():
            return jsonify({"error": "frame not found"}), 404
        from PIL import Image as _Image

        with _Image.open(path) as im:
            frame = im.convert("RGB")
        det, _strong = detect_on_frame(frame, _model, conf=_arg_float("conf", DEFAULT_CONF))
        if det is None:
            return jsonify({"boxes": [], "source": "none", "conf": None})
        return jsonify({
            "boxes": [list(det.bbox)],
            "source": "model",
            "conf": round(det.confidence, 3),
        })

    @app.post("/annotate/label")
    def annotate_label():
        from coffeecam import annotations

        data = request.get_json(silent=True) or {}
        rel = data.get("rel")
        if not isinstance(rel, str) or not rel.strip():
            return jsonify({"error": "missing rel"}), 400
        boxes = data.get("boxes", [])
        frame_size = None
        src = _annot_captures_dir() / rel
        if src.exists():
            from PIL import Image as _Image

            with _Image.open(src) as im:
                frame_size = im.size
        try:
            with _annot_lock:
                ann = annotations.upsert(
                    rel, boxes,
                    note=data.get("note", ""),
                    frame_size=frame_size,
                    store=_annot_store_path(),
                )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(ann.to_json())

    @app.post("/annotate/label/delete")
    def annotate_label_delete():
        from coffeecam import annotations

        data = request.get_json(silent=True) or {}
        rel = data.get("rel")
        if not isinstance(rel, str) or not rel.strip():
            return jsonify({"error": "missing rel"}), 400
        with _annot_lock:
            removed = annotations.remove(rel, store=_annot_store_path())
        return jsonify({"removed": removed})

    @app.post("/annotate/skip")
    def annotate_skip():
        from coffeecam import annotations

        data = request.get_json(silent=True) or {}
        rel = data.get("rel")
        if not isinstance(rel, str) or not rel.strip():
            return jsonify({"error": "missing rel"}), 400
        with _annot_lock:
            ann = annotations.skip(rel, store=_annot_store_path())
        return jsonify(ann.to_json())

    @app.post("/annotate/skip-queue")
    def annotate_skip_queue():
        """Mark every currently-unlabeled frame watched (honours ``start``;
        ignores ``stride`` so no gaps are left behind)."""
        from coffeecam import annotations

        captures_dir = _annot_captures_dir()
        anns = annotations.load(_annot_store_path())
        start = request.args.get("start")
        rels = [
            r.path.relative_to(captures_dir).as_posix()
            for r in collect_frames(captures_dir)
            if not (start and r.day < start)
        ]
        todo = [rel for rel in rels if rel not in anns]
        with _annot_lock:
            n = annotations.skip_many(todo, store=_annot_store_path())
        return jsonify({"skipped": n})

    @app.post("/annotate/promote")
    def annotate_promote():
        if request.args.get("confirm") != "1":
            return jsonify({"error": "pass ?confirm=1 to run promote"}), 400
        from coffeecam.dataset import DEFAULT_DATASET_DIR, promote

        dataset_dir = Path(os.environ.get("COFFEECAM_DATASET_DIR", DEFAULT_DATASET_DIR))
        with _annot_lock:
            summary = promote(
                store=_annot_store_path(),
                captures_dir=_annot_captures_dir(),
                dataset_dir=dataset_dir,
            )
        return jsonify({
            "summary": str(summary),
            "train": summary.train,
            "val": summary.val,
            "test": summary.test,
            "negatives": summary.negatives,
            "skipped_missing": summary.skipped_missing,
            "watched": summary.watched,
        })

    # --- /fullness fill-level labeling endpoint ---------------------------

    @app.get("/fullness")
    def fullness_page():
        return Response(_FULLNESS_PAGE, mimetype="text/html")

    @app.get("/fullness/queue.json")
    def fullness_queue():
        frames, counts, picked, captures_dir = _fullness_queue()
        from coffeecam import fullness_labels

        store_counts = fullness_labels.counts(_fullness_store_path())
        # queue counts (total/labeled/watched) are over box-positive frames and
        # stay authoritative; fold in only the per-level breakdown.
        for lvl in fullness_labels.LABELS:
            counts[lvl] = store_counts.get(lvl, 0)

        # ?model=1 runs the current classifier over every frame in the picked
        # set (top-1 level + confidence only, not the full prob vector — keeps
        # the payload small) so the browser can flag where it's weak. ?sort=
        # confidence then orders ascending, lowest-confidence first: the frames
        # most worth annotating next.
        if _arg_bool("model", False):
            from PIL import Image as _Image

            from coffeecam.fullness_crop import prepare_crop

            estimator = _get_fullness_estimator()
            for f, (rel, box, _lvl, _skip) in zip(frames, picked):
                path = captures_dir / rel
                if not path.exists():
                    continue
                try:
                    with _Image.open(path) as im:
                        crop = prepare_crop(im.convert("RGB"), tuple(box))
                    result = estimator.estimate(crop)
                except Exception:  # noqa: BLE001 — best-effort, never break the queue
                    continue
                probs = result.detail.get("probs") or {}
                f["model_level"] = result.level
                f["model_conf"] = max(probs.values()) if probs else None

            if request.args.get("sort") == "confidence":
                frames.sort(key=lambda f: (f.get("model_conf") is None, f.get("model_conf", 1.0)))

        return jsonify({"frames": frames, "counts": counts})

    @app.get("/fullness/suggest.json")
    def fullness_suggest():
        # Mirrors /annotate/suggest.json: the classifier's own read of the frame
        # the human is about to label, so agreement/disagreement is visible
        # before you commit a label.
        from PIL import Image as _Image

        from coffeecam import annotations
        from coffeecam.fullness_crop import prepare_crop

        rel = request.args.get("rel", "")
        path = _safe_capture_path(rel)
        if path is None or not path.exists():
            return jsonify({"error": "frame not found"}), 404
        ann = annotations.load(_annot_store_path()).get(rel)
        if ann is None or not ann.boxes:
            return jsonify({"error": "no box for frame"}), 404
        estimator = _get_fullness_estimator()
        with _Image.open(path) as im:
            crop = prepare_crop(im.convert("RGB"), tuple(ann.boxes[0]))
        result = estimator.estimate(crop)
        return jsonify({
            "level": result.level, "score": result.score, "method": result.method,
            "probs": result.detail.get("probs"),
        })

    @app.get("/fullness/crop.jpg")
    def fullness_crop():
        # Addressed by ``rel`` for the same reason as ``/annotate/frame.jpg``.
        from coffeecam import annotations

        rel = request.args.get("rel", "")
        path = _safe_capture_path(rel)
        if path is None:
            return Response("bad rel", status=400, mimetype="text/plain")
        ann = annotations.load(_annot_store_path()).get(rel)
        if ann is None or not ann.boxes:
            return Response("no box for frame", status=404, mimetype="text/plain")
        jpeg = _fullness_crop_jpeg(path, ann.boxes[0])
        if jpeg is None:
            return Response("frame gone", status=404, mimetype="text/plain")
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/fullness/frame.jpg")
    def fullness_full_frame():
        path = _safe_capture_path(request.args.get("rel", ""))
        if path is None:
            return Response("bad rel", status=400, mimetype="text/plain")
        jpeg = _fullness_crop_jpeg(path, None, full=True)
        if jpeg is None:
            return Response("frame gone", status=404, mimetype="text/plain")
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/fullness/label")
    def fullness_label():
        from coffeecam import fullness_labels

        data = request.get_json(silent=True) or {}
        rel = data.get("rel")
        if not isinstance(rel, str) or not rel.strip():
            return jsonify({"error": "missing rel"}), 400
        try:
            with _fullness_lock:
                label = fullness_labels.upsert(
                    rel, data.get("level", ""),
                    note=data.get("note", ""),
                    store=_fullness_store_path(),
                )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(label.to_json())

    @app.post("/fullness/label/delete")
    def fullness_label_delete():
        from coffeecam import fullness_labels

        data = request.get_json(silent=True) or {}
        rel = data.get("rel")
        if not isinstance(rel, str) or not rel.strip():
            return jsonify({"error": "missing rel"}), 400
        with _fullness_lock:
            removed = fullness_labels.remove(rel, store=_fullness_store_path())
        return jsonify({"removed": removed})

    @app.post("/fullness/skip")
    def fullness_skip():
        from coffeecam import fullness_labels

        data = request.get_json(silent=True) or {}
        rel = data.get("rel")
        if not isinstance(rel, str) or not rel.strip():
            return jsonify({"error": "missing rel"}), 400
        with _fullness_lock:
            label = fullness_labels.skip(rel, store=_fullness_store_path())
        return jsonify(label.to_json())

    @app.post("/fullness/skip-queue")
    def fullness_skip_queue():
        """Mark every box-positive frame with no fullness row watched (honours
        ``start``; ignores ``stride``)."""
        from coffeecam import fullness_labels

        start = request.args.get("start")
        have = fullness_labels.load(_fullness_store_path())
        todo = [
            rel
            for rel in fullness_labels.positive_box_rels(_annot_store_path())
            if rel not in have and not (start and rel.split("/", 1)[0] < start)
        ]
        with _fullness_lock:
            n = fullness_labels.skip_many(todo, store=_fullness_store_path())
        return jsonify({"skipped": n})

    @app.get("/")
    def index():
        with _lock:
            result = _latest
        refresh = int(_cfg()["refresh"])
        if result is None:
            return f'<meta http-equiv=refresh content="{refresh}"><p>warming up… {_last_fetch_error or ""}', 503
        f = result.fullness
        d = result.detection
        errs = list(result.errors)
        if _last_fetch_error:
            errs.append(f"fetch: {_last_fetch_error}")
        return _PAGE.format(
            refresh=refresh,
            ts=result.ts.strftime("%H:%M:%S"),
            stale=round((datetime.now() - result.ts).total_seconds(), 1),
            level=f.level,
            score="" if f.score is None else f"{f.score:.2f}",
            method=f.method,
            errline=f'<div class=err>{" · ".join(errs)}</div>' if errs else "",
            detconf="" if d is None else f" · conf {d.confidence:.3f} · bbox {list(d.bbox)}",
            timings=result.timings_ms,
            jsonlink='<a href="/pipeline.json" style="color:#6ab0ff">/pipeline.json</a>',
        )

    return app


if __name__ == "__main__":
    _start_worker()
    create_app(start_worker=False).run(
        host=os.environ.get("COFFEECAM_HOST", "0.0.0.0"),
        port=int(os.environ.get("COFFEECAM_PORT", "8000")),
        threaded=True,
    )
