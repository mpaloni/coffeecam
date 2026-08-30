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
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from datetime import datetime

from flask import Flask, Response, jsonify

from coffeecam.capture import DEFAULT_SOURCE, fetch_snapshot
from coffeecam.pipeline import DEFAULT_CONF, PipelineResult, run_pipeline

_lock = threading.Lock()
_latest: PipelineResult | None = None
_latest_jpeg: dict[str, bytes] = {}
_last_fetch_error: str | None = None
_worker_started = False


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
    from pathlib import Path

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
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    from coffeecam.detect import load_model

    model = load_model()
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
{jsonlink}</pre>
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
