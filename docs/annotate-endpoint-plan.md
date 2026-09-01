# `/annotate` — browser labeling endpoint

Handoff plan for a web UI to draw `coffee_pot` bounding boxes on captured frames,
so the ~250 real frames in `captures/` become a training set. Closes the
**Labeling** section of `TODO.md`.

Status: **in progress.** Done: §4.1 `annotations.py` + tests, §4.2 `annotate.write_example`
+ tests, §4.3 `dataset.promote` + split manifests + tests, §4.4 server routes + tests,
§4.5 the `_ANNOTATE_PAGE` UI (canvas draw/move/resize, prefill, keys, done+promote state),
§4.6 tests (`tests/test_annotations.py`, promote tests in `test_dataset.py`, route tests in
`test_server.py`), §8.7 docs (README §1c, `docs/PIPELINE.md` §/annotate, TODO.md Labeling).
Remaining: §8.6 — label a first batch of ~30 real frames via `/annotate`, `promote`,
retrain, confirm val/test mAP reflects real held-out frames.

---

## 1. Why build this (not Label Studio / CVAT)

`TODO.md` suggests standing up labelImg / CVAT / Label Studio. For this project
that is overkill:

- **One class, usually one box per frame.** The whole labeling interaction is
  "drag a rectangle, next".
- **Live-model prefill for free.** The server already holds a loaded detector
  (`server._model`). Seeding each frame with the current model's guess turns
  most frames into an accept-or-nudge, and doubles as a review of where the
  model is wrong.
- **No install, no export step, same box.** Runs in the Flask app already on
  `192.168.50.16:8000`; writes labels where training reads them.

If labeling needs outgrow one box / one class (multiple pots, occlusion
polygons, fill-level segmentation), revisit an off-the-shelf tool then.

## 2. What already exists to build on

| Piece | Where | Reuse |
|---|---|---|
| YOLO bbox math | `coffeecam/dataset.py` — `clamp_bbox`, `bbox_pixel_to_yolo`, `bbox_yolo_to_pixel`, `write_label`, `read_label` | all conversions; do not re-derive |
| Single-image labeler | `coffeecam/annotate.py` — `annotate_image(path, bbox, …)` writes YOLO label + preview PNG | refactor so CLI and the new `promote` share one "write example into `dataset/`" fn |
| Frame walk + sampling | `coffeecam/summary.py` — `collect_frames()`, `_sample()`, `FrameRef` | queue building |
| Detector on a frame | `coffeecam/summary.py` — `detect_on_frame(frame, model, conf=)` → `(Detection|None, is_strong)` | prefill suggestions |
| Cached frame server | `coffeecam/server.py` — `_viewer_lock` / `_viewer_cache`, `_arg_int/_arg_float/_arg_bool`, `/viewer/frame/<i>.jpg` | copy the pattern |
| Frame size | all captures are **424×353 RGB** (`dataset/images/kahvi.png` is 512×456 — different, fine, YOLO normalizes) | — |

## 3. Core design decision — labels go to a sidecar, not straight to `dataset/`

Write annotations to **`captures/annotations.jsonl`**, one upsertable row per
frame:

```json
{"rel": "2026-08-31/161913_556.jpg", "boxes": [[171, 88, 249, 206]], "labeled_at": "2026-09-01T13:22:04", "note": ""}
```

- `rel` — path under the captures dir, the stable key (matches `FrameRef` and
  the `/viewer` manifest).
- `boxes` — list of `[x1, y1, x2, y2]` in **natural pixel space** (same contract
  as `annotate.py`'s CLI). Empty list `[]` = **explicit negative** (frame seen,
  no pot) — keep these, `TODO.md` wants negatives.
- Absent row = not yet labeled.

A separate **`promote`** step builds `dataset/images/`, `dataset/labels/`, and
the split manifests from the jsonl.

Why the indirection:

- `captures/` stays a clean raw archive; labeling never mutates it.
- The endpoint is a pure writer to one lockable file — easy to reason about,
  back up, diff.
- You curate *which* labeled frames enter the training set, and can rebuild
  `dataset/` deterministically as labels accumulate.
- Re-runnable: fix a bad box, re-promote, retrain.

## 4. Work items

### 4.1 `coffeecam/annotations.py` (new) — the label store

```python
DEFAULT_STORE = "captures/annotations.jsonl"

@dataclass(frozen=True)
class Annotation:
    rel: str
    boxes: list[tuple[int, int, int, int]]   # x1,y1,x2,y2 natural px
    labeled_at: str
    note: str = ""

def load(store: Path = ...) -> dict[str, Annotation]      # rel -> Annotation
def upsert(rel, boxes, *, note="", store=...) -> Annotation  # atomic: tmp file + os.replace, under a lock
def remove(rel, *, store=...) -> bool
```

- Atomic write: serialize the whole dict to `annotations.jsonl.tmp`, `os.replace`.
  The file is small (hundreds of lines); no need for append semantics.
- Validate boxes on the way in: ints, `x2 > x1`, `y2 > y1`, within frame bounds
  (pass the frame size in, or clamp with `dataset.clamp_bbox`). Reject otherwise
  with a 400.
- Caller (server) owns the lock; keep this module lock-free and pure so it is
  unit-testable without threads.

### 4.2 `coffeecam/annotate.py` — refactor for shared use

- Extract `write_example(image_path, boxes, *, images_dir, labels_dir, class_id=0)`
  that copies/normalizes the image into `dataset/images/` and writes the
  multi-box YOLO label into `dataset/labels/`. Current `annotate_image()` handles
  one box and also writes a preview — keep the preview for the CLI path, make it
  optional for `promote`.
- CLI stays working (`python -m coffeecam.annotate <img> x1 y1 x2 y2`).

### 4.3 `coffeecam/dataset.py` — `promote` + split manifest

- `promote(store=..., captures_dir=..., dataset_dir="dataset", *, val_frac=0.15, test_frac=0.15, seed=0, negatives=True)`:
  1. `annotations.load()`, skip rows with no image on disk.
  2. For each, `annotate.write_example(...)`. Naming: `cap_YYYYMMDD_HHMMSS[_fff].jpg`
     from `rel`, so dataset filenames are stable and collision-free.
  3. Deterministic split (hash of `rel` or seeded shuffle). **`kahvi.png` stays in
     train** (see `data.yaml` comment). Real frames only in val/test — never
     `augment_shift` copies.
  4. Rewrite `dataset/train.txt` / `val.txt` / `test.txt` (add `test.txt` to
     `data.yaml`). This also fills the second unchecked **Labeling** bullet in
     `TODO.md` (reproducible split manifests).
- `promote` is idempotent: re-running with more labels regenerates cleanly.
  Print a summary (`N train / M val / K test, P negatives`).
- CLI: `python -m coffeecam.dataset promote [--dry-run]`.

### 4.4 `coffeecam/server.py` — routes

Add `_annot_lock = threading.Lock()`. No long-lived cache needed (the store is
cheap to read); read it per request under the lock. Reuse `_arg_*` helpers.

| Route | Behavior |
|---|---|
| `GET /annotate` | the labeling page (`_ANNOTATE_PAGE`, static string, mimetype `text/html`) |
| `GET /annotate/queue.json` | `{frames: [{i, rel, ts, labeled, boxes}], counts: {total, labeled, remaining}}`. Params: `filter=unlabeled\|labeled\|all` (default `unlabeled`), `stride=N` (default 1 — see §6.2), `start=YYYY-MM-DD`. Built from `collect_frames()` + `annotations.load()`. |
| `GET /annotate/frame/<int:i>.jpg` | **raw** frame bytes at native res, **no annotation drawn** (unlike `/viewer/frame`). `Cache-Control: private, max-age=300`. `i` indexes the current queue ordering; 404 out of range. |
| `GET /annotate/suggest/<int:i>.json` | `{boxes: [[x1,y1,x2,y2]], source: "model"\|"none", conf: 0.42}` — run `detect_on_frame(frame, server._model)`. Lazy, not cached, only called when the user asks for a hint. 503 if `_model` is None. |
| `POST /annotate/label` | body `{rel, boxes}`. Validate, `annotations.upsert(...)` under `_annot_lock`, return the saved row. `boxes: []` allowed (negative). |
| `POST /annotate/label/delete` | body `{rel}` → `annotations.remove(...)`, return `{removed: bool}`. |
| `POST /annotate/promote` | optional; runs `dataset.promote()` and returns its summary. Gate behind `?confirm=1`. Can be left CLI-only for v1. |

Link `/annotate` from the dashboard index `<pre>` footer next to `/viewer`.

Route ordering note: register `queue.json` / `frame/<i>.jpg` etc. before any
catch-all; Flask's converter routes are fine here since paths are distinct.

### 4.5 The page — `_ANNOTATE_PAGE`

Single static HTML+CSS+JS string, same house style as `_VIEWER_PAGE` (dark,
system-ui, `#14161a` ground, `#6ab0ff` links). ~150 lines, drawing is the bulk.

Layout: image area (left / top) + a thin control column.

- **Canvas over the image.** `<img>` sized to fit the viewport with
  `max-width`/`max-height`; a `<canvas>` positioned exactly over it at the same
  CSS size. Draw the box(es) on the canvas.
- **Draw:** pointerdown starts a rubber-band rect, pointermove resizes,
  pointerup commits. After commit, 8 resize handles + drag-to-move on the body.
  One box is the norm; allow adding a second with a modifier or an "add box"
  button (rare, but multi-pot happens).
- **Coordinate mapping (the gotcha, §6.1):** store boxes in **natural pixel
  space**. On every pointer event convert with
  `natX = evt.offsetX * img.naturalWidth / img.clientWidth` (same for Y).
  Re-derive on `resize`. POST natural px.
- **Prefill:** on frame load, call `/annotate/suggest/<i>` and draw the result
  as a faint dashed box the user can accept (Enter) or drag. If the frame
  already has a saved label, show that instead (solid), pre-selected.
- **Controls / keys:**
  - `Space` or `Enter` — save current box(es) → `POST /label` → advance
  - `x` — save as **negative** (empty boxes) → advance
  - `d` / `Delete` — clear box on canvas; `Backspace` on a saved frame →
    `POST /label/delete`
  - `←` / `→` — prev / next without saving
  - arrow keys with a box selected — nudge 1px (Shift = 10px)
  - `z` — undo last box edit (local stack, not server)
- **Header readout:** `rel` path, timestamp, `37 / 250 labeled · 213 left`,
  and a class label (`coffee_pot`) — trivially fixed now, but show it so adding
  classes later is obvious.
- **After the last queued frame:** show a done state with a "run promote"
  button (if §4.4 promote route is built) or the CLI command to copy.

### 4.6 Tests — `tests/test_annotations.py`, extend `tests/test_server.py`

`annotations.py` (pure, no server):

- `upsert` creates the store, then updates in place (same `rel` → one row).
- `upsert` with `boxes=[]` round-trips as a negative.
- `load` returns `{rel: Annotation}`; ignores blank / malformed lines.
- `remove` returns False for an absent `rel`, True after an `upsert`.
- atomic write: no `.tmp` left behind; a concurrent `load` never sees a partial
  file (can assert the tmp+replace path exists rather than truly racing).
- box validation: `x2 <= x1`, out-of-bounds, non-int → `ValueError`.

`dataset.promote`:

- builds `dataset/images` + `labels` + 3 split files from a fixture store with
  3–4 fake capture frames; label contents match `bbox_pixel_to_yolo`.
- negative rows produce an **empty** `labels/<name>.txt` (YOLO convention) and
  the image still lands in `images/`.
- `kahvi.png` stays in `train.txt`.
- idempotent: run twice, same output, no dupes.
- `--dry-run` writes nothing.

server routes (reuse the `captures` fixture from `test_summary.py`; add
`server._annot_lock` reset — and any new global — to the autouse
`_reset_server_state` in `test_server.py` **and** `test_summary.py`, matching the
existing `_summary_cache` / `_viewer_cache` resets):

- `GET /annotate` → 200 `text/html`.
- `queue.json` default `filter=unlabeled` lists all frames; after a `POST /label`
  for one `rel`, that frame drops out and `counts.labeled == 1`.
- `filter=labeled` / `all` behave.
- `frame/<i>.jpg` → 200 `image/jpeg`, bytes equal the raw capture file (not
  re-encoded/annotated); out-of-range → 404.
- `POST /label` with a good box → 200, row in the store; bad box → 400.
- `POST /label` with `boxes: []` → 200, negative stored.
- `POST /label/delete` → row gone, frame reappears in the unlabeled queue.
- `suggest/<i>` → 503 when `server._model is None` (default in tests);
  with a `FakeModel` (copy from `test_summary.py`) → `{boxes: [...], source:
  "model"}`.
- no captures dir → `queue.json` 200 with empty list (not 404); `frame/1.jpg`
  → 404.

## 5. Data formats (reference)

**`captures/annotations.jsonl`** — one JSON object per line, `rel` unique:

```
{"rel": "2026-08-31/161913_556.jpg", "boxes": [[171, 88, 249, 206]], "labeled_at": "2026-09-01T13:22:04", "note": ""}
{"rel": "2026-08-31/155905_054.jpg", "boxes": [], "labeled_at": "2026-09-01T13:22:31", "note": "carafe removed"}
```

**`dataset/labels/<name>.txt`** — YOLO, unchanged: `0 cx cy w h` normalized,
one line per box, empty file for a negative.

**Split manifests** — `dataset/{train,val,test}.txt`, one image path per line
relative to `dataset/` (e.g. `./images/cap_20260831_161913.jpg`), as today.

## 6. Gotchas

### 6.1 Display → natural coordinates
The browser scales the 424×353 frame to fit. A box read in CSS pixels is wrong
in the label. Convert **every** pointer coordinate by
`naturalWidth / clientWidth` before storing, and recompute on window `resize`
and on image load. Store and POST natural px; the server converts to YOLO only
at `promote`. Add a visible check: draw the stored box back from natural px each
frame and confirm it lands where drawn.

### 6.2 Near-duplicate frames waste labeling effort
`captures/` is full of 5-minute heartbeat frames from static scenes (32 such
frames were pruned from `2026-08-31/` on 2026-09-01). Labeling all ~250 spends
the budget on identical shots. `queue.json` `stride=N` (take every Nth) is the
cheap fix; a better one is a "distinct only" filter using a perceptual/luma diff
between consecutive frames (there is already mean-luma code in
`coffeecam/image_summary.py`). Ship `stride` in v1, note the diff filter as a
follow-up.

### 6.3 Frame index vs. stable key
`frame/<i>.jpg` indexes the *current queue*, which changes as frames get labeled
(with `filter=unlabeled`) or as `stride` changes. The client must treat `i` as
ephemeral and always send `rel` (never `i`) on `POST /label`. `queue.json`
returns both.

### 6.4 Concurrency
Single user, but the capture loop writes `captures/` continuously and the
dashboard worker reads `_model`. The label store is separate
(`captures/annotations.jsonl`) so there is no contention with capture. Guard the
store with `_annot_lock` anyway — multiple browser tabs are a real case.

### 6.5 Don't block the worker
`suggest/<i>` runs the detector inline on the request thread. That is fine at
human pace (one click = one inference) but do not prefetch suggestions for the
whole queue.

## 7. Out of scope for v1

- Multi-class labeling UI (structure leaves room: `boxes` could become
  `[{cls, xyxy}]` later).
- Polygon / segmentation (fullness classifier may want it — separate effort).
- Auth. Homelab LAN, same posture as the rest of the dashboard.
- Undo across frames / an edit history on the server (local undo stack only).
- Auto-advance training after promote.

## 8. Suggested sequence

1. `annotations.py` + `tests/test_annotations.py` (pure, fast to nail down).
2. `annotate.py` refactor → shared `write_example`.
3. `dataset.py` `promote` + split manifests + tests. Now labels → trainable set
   works end to end from a hand-written jsonl.
4. Server routes + tests (no page yet — curl/pytest).
5. `_ANNOTATE_PAGE`: image + canvas + save/next first; then prefill; then
   handles/nudge/undo polish.
6. Label a first ~30 varied frames, `promote`, retrain, check val mAP moved.
   Iterate on the UI from real use.
7. Update `TODO.md` (check off the two **Labeling** bullets), add a short
   `docs/` note or fold into `PIPELINE.md`.
