# Browser labeling (`/annotate`) + dataset audit tooling

Web UI to draw `coffee_pot` bounding boxes on captured frames, turning raw
camera captures into a training set — no external tool (Label Studio/CVAT)
needed for one class, usually one box per frame.

**Shipped and current.** Original design: git history of this file (the
build-order plan has been superseded by this doc). Labels go to a sidecar,
never mutate `captures/`:

```json
{"rel": "2026-08-31/161913_556.jpg", "boxes": [[171, 88, 249, 206]], "labeled_at": "2026-09-01T13:22:04", "note": ""}
```

`boxes: []` = explicit negative (frame seen, no pot). A `skip: true` row is
*watched* — seen, deliberately not labeled — and is dropped entirely by
`dataset.promote`. `rel` is the stable key (path under `captures/`).

## Routes (`coffeecam/server.py`)

| Route | Behavior |
|---|---|
| `GET /annotate` | the labeling page |
| `GET /annotate/queue.json` | `{frames, counts}`. `filter=unlabeled\|labeled\|all\|watched`, `stride=N`, `start=YYYY-MM-DD` |
| `GET /annotate/frame.jpg?rel=` | raw frame bytes, no box drawn |
| `GET /annotate/suggest.json?rel=` | live-model prefill box, 503 if no model loaded |
| `POST /annotate/label` | body `{rel, boxes}` |
| `POST /annotate/label/delete` | body `{rel}` |
| `POST /annotate/skip` / `skip-queue` | mark watched |
| `POST /annotate/promote?confirm=1` | runs `dataset.promote()` |

UI keys: `Space`/`Enter` save+advance · `x` negative · `s` skip/watched · `h`
hint (model prefill) · `z` undo · arrows nudge a box (Shift = 10px).

## The stale-index bug — fixed 2026-09-04

**Symptom:** the image shown could drift from the `rel` a label was actually
saved against, because routes addressed frames by a positional index `i` into
a queue that was rebuilt (and re-ordered) on every request — a `save()` or a
new capture landing mid-session shifted every later index.

**Fix (`8a54f56`, verified still in place):** `/annotate/frame.jpg`,
`/annotate/suggest.json`, and the two `/fullness/{crop,frame}.jpg` routes now
all take `?rel=<rel>` instead of `<int:i>`, resolved and traversal-guarded by
`_safe_capture_path()`, serving `Cache-Control: no-store` (was
`max-age=300` on the unstable ordinal URL — a second, compounding bug: the
browser could serve a cached image from a *different* frame for up to 5 min).
`i` survives only as an on-screen cursor / `queue.length` math, never sent on
a write. Both `_ANNOTATE_PAGE` and the `/fullness` page's JS build URLs from
`f.rel`. Covered by `test_annotate_frame_by_rel_is_stable_across_saves` in
`tests/test_server.py`.

### Audit of pre-fix data — findings not fully closed out

The bug had been present since the labeling UI shipped, so a
`coffeecam/annotation_audit.py` tool was built to estimate the blast radius:
for every box-positive row, run the detector on that row's frame and compare
IoU against the saved box. Run 2026-09-04 over 573 rows
(`models/best-trackB-sess.pt`):

| bucket | rows | % |
|---|---|---|
| OK (IoU ≥ 0.5) | 379 | 66% |
| LIKELY_MISMATCH (IoU < 0.2, confident detection elsewhere) | 80 | 14% |
| INCONCLUSIVE (no/weak detection) | 70 | 12% |
| NEGATIVE_HAS_POT (row says "no pot", detector disagrees ≥0.5 conf) | 44 | 8% |

Skewed toward the earliest days — 2026-08-31 was 57% flagged, 2026-09-04 (post
fix) 0%. The offset was **not systematic** (no single constant shift explains
the mismatches), so a scripted correction was ruled out — the recommendation
was to re-label the flagged `rel`s by hand through the fixed UI, then re-run
the audit to confirm the flagged count dropped.

**Status: the re-run never happened.** `annotation_audit.csv` and
`audit_contact/*.jpg` are both still timestamped 2026-09-04 — the same run
that produced the table above. `captures/annotations.jsonl` has grown
substantially since (934 rows at audit time → 2244 now), but that reflects
ongoing labeling of new captures, not confirmation that the originally
flagged rows were fixed. Before trusting the current detector/fullness
training data's label quality, re-run:

```bash
.venv/bin/python -m coffeecam.annotation_audit --weights <current best .pt> --contact --fullness
```

and compare bucket counts against the table above.

## Gotchas

- **Frame index vs. stable key** — `queue.json` returns both `i` (ephemeral,
  for the current queue ordering) and `rel` (stable). Always write with `rel`.
- **Near-duplicate frames waste labeling effort.** `captures/` accumulates
  5-minute heartbeat frames from a static scene. `stride=N` is the cheap
  mitigation shipped; a perceptual/luma-diff "distinct only" filter (using
  `coffeecam/image_summary.py`'s mean-luma code) was scoped but never built.
- **Coordinate mapping** — boxes are stored/POSTed in natural pixel space; the
  browser converts every pointer event by `naturalWidth / clientWidth`.
- **UI has no execution test coverage** — `_ANNOTATE_PAGE` is asserted to
  serve the right HTML strings; canvas drawing/handle math has never run in a
  headless browser.

## Tests

`tests/test_annotations.py` (pure store logic), `tests/test_annotation_audit.py`
(18 cases), route coverage in `tests/test_server.py`.
