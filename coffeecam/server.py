"""Flask dev server: runs the pipeline on a timer and serves every stage.

    .venv/bin/python -m coffeecam.server

A daemon thread fetches a snapshot every COFFEECAM_REFRESH_SECS, runs
`pipeline.run_pipeline`, and stashes the rendered JPEGs; the routes just serve the
last result, so browser refreshes and multiple viewers cost nothing. The classify
stage is the brightness-heuristic placeholder — the point of the per-stage
endpoints is to have the detector/crop visible now and slot a real classifier in
later.

Env:
  COFFEECAM_SOURCE_URL   camera base URL           (default http://192.168.50.10:8888)
  COFFEECAM_REFRESH_SECS seconds between runs      (default 10)
  COFFEECAM_CONF         detector confidence floor (default 0.15)
  COFFEECAM_NORMALIZE    1/0 black-pad to train ar (default 1)
  COFFEECAM_HARVEST      1/0 save frame+crop+json  (default 0) -> captures/pipeline/
  COFFEECAM_HOST / COFFEECAM_PORT                  (default 0.0.0.0 / 8000)
  COFFEECAM_SUMMARY_TTL  seconds to cache /summary (default 300)
  COFFEECAM_CAPTURES_DIR  frame dir for /summary + /annotate (default captures/)
  COFFEECAM_DATASET_DIR   output dir for POST /annotate/promote (default dataset/)

/summary[.gif] renders every captured frame so far into one animated GIF, each
frame annotated with the current detector's result box (query params: annotate,
frames, ms, scale, conf, rebuild). /summary.json returns the build metadata.

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

from flask import Flask, Response, jsonify, request

from coffeecam.capture import DEFAULT_SOURCE, fetch_snapshot
from coffeecam.pipeline import DEFAULT_CONF, PipelineResult, run_pipeline
from coffeecam.summary import (
    DEFAULT_CAPTURES_DIR,
    DEFAULT_DURATION_MS,
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


def _harvest(result: PipelineResult) -> None:
    day = result.ts.strftime("%Y-%m-%d")
    stem = result.ts.strftime("%H%M%S")
    base = Path("captures/pipeline") / day
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{stem}_frame.jpg").write_bytes(_encode(result.frame))
    if result.crop is not None:
        (base / f"{stem}_crop.jpg").write_bytes(_encode(result.crop))
    (base / f"{stem}.json").write_text(json.dumps(_result_dict(result), indent=None))


def _worker(model) -> None:
    global _last_fetch_error
    cfg = _cfg()
    while True:
        try:
            snap = fetch_snapshot(cfg["source_url"])
            _last_fetch_error = None
            result = run_pipeline(snap, model=model, normalize=cfg["normalize"], conf=cfg["conf"])
            _store(result)
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
    return dict(
        annotate=_arg_bool("annotate", True),
        max_frames=_arg_int("frames", 160),
        ms=_arg_int("ms", DEFAULT_DURATION_MS),
        scale=_arg_float("scale", 0.6),
        conf=_arg_float("conf", DEFAULT_CONF),
        force=_arg_bool("rebuild", False),
    )


def _build_summary(*, annotate: bool, max_frames: int, ms: int, scale: float, conf: float, force: bool):
    """Cached wrapper around `summary.build_summary_gif`. Rebuilds when the
    parameters change, on `?rebuild=1`, or once the cached GIF is older than
    COFFEECAM_SUMMARY_TTL seconds (default 300) so new captures roll in."""
    global _summary_cache
    ttl = float(os.environ.get("COFFEECAM_SUMMARY_TTL", "300"))
    captures_dir = Path(os.environ.get("COFFEECAM_CAPTURES_DIR", DEFAULT_CAPTURES_DIR))
    sig = (str(captures_dir), annotate, max_frames, ms, round(scale, 3), round(conf, 3))
    with _summary_lock:
        cached = _summary_cache
        if cached and not force and cached["sig"] == sig and (time.time() - cached["at"]) < ttl:
            return cached["gif"], cached["meta"]
        gif, meta = build_summary_gif(
            captures_dir=captures_dir,
            model=_model if annotate else None,
            conf=conf,
            max_frames=max_frames,
            duration_ms=ms,
            scale=scale,
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


# --- /annotate: browser bbox-labeling backed by captures/annotations.jsonl ---

def _annot_captures_dir() -> Path:
    return Path(os.environ.get("COFFEECAM_CAPTURES_DIR", DEFAULT_CAPTURES_DIR))


def _annot_store_path() -> Path:
    return _annot_captures_dir() / "annotations.jsonl"


def _annot_queue():
    """Ordered labeling queue for the current query params.

    Returns ``(frames, counts, picked, captures_dir)`` where ``frames`` is the
    JSON-ready list, ``picked`` is the parallel list of ``(rel, FrameRef,
    labeled)`` so ``/annotate/frame/<i>`` and ``/suggest/<i>`` can resolve ``i``
    against the exact same ordering.

    Params: ``filter=unlabeled|labeled|all`` (default unlabeled),
    ``stride=N`` (take every Nth frame, default 1), ``start=YYYY-MM-DD``.
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
        rows.append((rel, r, rel in anns))

    total = len(rows)
    labeled = sum(1 for _, _, is_l in rows if is_l)

    strided = rows[::stride]
    if filt == "labeled":
        picked = [x for x in strided if x[2]]
    elif filt == "all":
        picked = list(strided)
    else:
        picked = [x for x in strided if not x[2]]

    frames = []
    for i, (rel, r, is_l) in enumerate(picked):
        ann = anns.get(rel)
        frames.append({
            "i": i,
            "rel": rel,
            "ts": r.ts_label,
            "labeled": is_l,
            "boxes": [list(b) for b in ann.boxes] if ann else [],
        })
    counts = {"total": total, "labeled": labeled, "remaining": total - labeled}
    return frames, counts, picked, captures_dir


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
  <button id=hint>suggest a box &nbsp;<span class=k>h</span></button>
  <button id=clear>clear boxes &nbsp;<span class=k>d</span></button>
  <button id=del>delete saved label &nbsp;<span class=k>&#9003;</span></button>
  <button id=reload>reload queue</button>
  <label class=f>filter <select id=filter>
    <option value=unlabeled selected>unlabeled</option>
    <option value=labeled>labeled</option>
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
    (f.labeled ? ' &middot; <span style="color:#4caf50">saved</span>' : '');
  $('count').textContent = counts.labeled + ' / ' + counts.total +
    ' labeled &middot; ' + (queue.length - pos) + ' in queue';
  view.onload = () => { fitCanvas(); if (!f.labeled) getSuggestion(); };
  view.src = '/annotate/frame/' + f.i + '.jpg?' + qs();
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
    const r = await fetch('/annotate/suggest/' + queue[pos].i + '.json?' + qs());
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

$('prev').onclick = () => { if (pos > 0) { pos--; show(); } };
$('next').onclick = () => { pos++; show(); };
$('save').onclick = () => save(false);
$('neg').onclick = () => save(true);
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
{jsonlink} · <a href="/summary" style="color:#6ab0ff">/summary</a> (annotated capture timelapse) · <a href="/viewer" style="color:#6ab0ff">/viewer</a> (scrubbable) · <a href="/annotate" style="color:#6ab0ff">/annotate</a> (label frames)</pre>
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

    # --- /annotate labeling endpoint ---------------------------------------

    @app.get("/annotate")
    def annotate_page():
        return Response(_ANNOTATE_PAGE, mimetype="text/html")

    @app.get("/annotate/queue.json")
    def annotate_queue():
        frames, counts, _, _ = _annot_queue()
        return jsonify({"frames": frames, "counts": counts})

    @app.get("/annotate/frame/<int:i>.jpg")
    def annotate_frame(i: int):
        _, _, picked, captures_dir = _annot_queue()
        if i < 0 or i >= len(picked):
            return Response("frame out of range", status=404, mimetype="text/plain")
        path = captures_dir / picked[i][0]
        if not path.exists():
            return Response("frame gone", status=404, mimetype="text/plain")
        return Response(
            path.read_bytes(),
            mimetype="image/jpeg",
            headers={"Cache-Control": "private, max-age=300"},
        )

    @app.get("/annotate/suggest/<int:i>.json")
    def annotate_suggest(i: int):
        if _model is None:
            return jsonify({"error": "detector not loaded"}), 503
        _, _, picked, captures_dir = _annot_queue()
        if i < 0 or i >= len(picked):
            return jsonify({"error": "frame out of range"}), 404
        from PIL import Image as _Image

        with _Image.open(captures_dir / picked[i][0]) as im:
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
        })

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
