# Annotation tooling: stale-index bug + full-dataset audit — HANDOFF

Two linked workstreams. **A (fix) and B (audit) both landed 2026-09-04** — see
"Results" at the bottom. Remaining human work: re-label the flagged rels through
the fixed UI, then re-run the audit.

Original framing (diagnosed 2026-09-04):

- **A. Fix** the stale positional-index bug in the `/annotate` and `/fullness`
  browser labelers — images shown don't match the header/`rel` the label is
  saved against.
- **B. Audit** all existing annotation data (`captures/annotations.jsonl`,
  `captures/fullness.jsonl`) for labels attached to the wrong frame, because the
  bug has likely been present for a while.

Unrelated work running in parallel: `imgsz960-sess` YOLO training on the
`coffeecam-train` pod (worker2). Don't disturb it. See
`docs/dataset-experiments-log.md`.

---

## Data state at handoff (2026-09-04 ~08:33)

`captures/annotations.jsonl` — 934 rows: **531 box, 48 negative, 355 skip/watched**
(counts drift as labeling continues). 946 captured jpgs; days 2026-08-31 →
2026-09-03 are 100 % covered, 2026-09-04 in progress.
`captures/fullness.jsonl` — fill-level labels, ~same era, also suspect.

---

## A. The bug

### Symptom
Annotate frame `07:11:18`, move on; the header then reads `07:16:18` but the
image on the canvas is a *different, later* frame. Reloading the whole page
realigns them ("frame without a hand" becomes "frame with a hand"). Boxes drawn
in that mismatched state are saved against the **header's `rel`**, not the frame
the user was actually looking at.

### Root cause — positional `i` used as a cross-request identifier
`coffeecam/server.py`:

- `_annot_queue()` (`server.py:428`) rebuilds the queue **per request**:
  `rows` (chronological) → `strided = rows[::stride]` → filter to
  `filter=unlabeled` → `picked`. Each frame's `i` is its **position in `picked`
  at that instant**.
- `/annotate/queue.json` (`:1353`) returns `frames[]` with those `i` values; the
  browser freezes them in `queue`.
- `show()` (`:743`): header, `rel`, prefilled `boxes` come from `queue[pos]`
  (**frozen**); the image is `GET /annotate/frame/<f.i>.jpg` (`:757`); the
  suggestion is `GET /annotate/suggest/<queue[pos].i>.json` (`:776`).
- `/annotate/frame/<i>` (`:1358`) and `/suggest/<i>` (`:1372`) each call
  `_annot_queue()` **again** and return `picked[i]` from the **current** list.

Between the `queue.json` fetch and an image fetch, `picked` shifts because:
1. **Every `save()` / `skipFrame()`** marks a frame labeled/skipped → it drops
   out of the `unlabeled` filter → every later index shifts down by one.
   `save()` (`:795`) does `f.labeled = true; pos++` **locally and never
   re-fetches `queue.json`**, so drift = number of saves since the last full
   page reload.
2. The **live capture loop** (`coffeecam-capture-loop.service`) appends new
   frames during the workday; with `stride > 1` this re-buckets `rows[::stride]`.

### Second, compounding defect — HTTP caching on an unstable URL
`/annotate/frame/<i>.jpg` returns `Cache-Control: private, max-age=300`
(`server.py:1369`; `/suggest` and the `/fullness/*` frame routes at `:1514`,
`:1526` do the same). `i` is a bare ordinal, so the browser will serve a **cached
image from a previous frame** for up to 5 min whenever an `i` repeats.

### `/fullness` has the identical pattern
`_fullness_queue()` (`:502`), `show()` (`:961`) uses `queue[pos].rel` for the
header but `f.i` for `/fullness/crop/<i>` and `/fullness/frame/<i>` (`:976-977`),
`label()` (`:989`) posts `f.rel` (frozen) + `pos++` with no reload. The crop and
frame are at least mutually consistent (both from `picked[i]`), but the fill
level is written against the stale header `rel`.
`viewer.py` is **not** affected — its manifest is built once and dumped; `f.i` is
stable within a build.

### Data-quality impact
Some fraction of box rows have coordinates for a frame N positions after the one
named in the row (N = saves since last reload; the user's trace suggests up to
~5). The scene changes slowly, so many are approximately right; the visible
misses are frames where a hand / person / moved pot appears. Fill-level labels
in `fullness.jsonl` are off the same way. Extent unknown until the audit (B).

### Fix — address frames by `rel`, not `i`
Contained to `coffeecam/server.py` (~15–20 lines) + tests.

1. `/annotate/frame` and `/annotate/suggest`: take `?rel=<rel>` instead of
   `<int:i>`. Resolve `path = (captures_dir / rel).resolve()`, **verify it is
   under `captures_dir.resolve()`** (path-traversal guard), 404 otherwise. Serve
   with `Cache-Control: no-store`.
2. Same for `/fullness/frame` and `/fullness/crop` (`:1503`, `:1517`). The crop
   needs the GT box — look it up from `annotations.load()` by `rel` rather than
   from `picked[i]`.
3. JS: `view.src = '/annotate/frame?rel=' + encodeURIComponent(f.rel) + '&' + qs()`;
   suggestion likewise; drop the `f.i` reads. `i` stays only as the on-screen
   cursor / for `queue.length` math.
4. Keep `_annot_queue()` returning `picked` for `queue.json`'s ordering, but
   nothing outside that request should index it.
5. Optional hardening: after `save()`/`label()`/`skip`, refresh `counts` from a
   light `queue.json?count_only=1` so the header stats stop drifting; not
   required for correctness once addressing is by `rel`.
6. Tests: `tests/test_server.py` — add cases that label frame k, then assert
   `/annotate/frame?rel=<k+1's rel>` still returns `k+1`'s bytes (today it
   shifts). Update existing `/annotate/frame/<i>` route tests.

Restart after patching: `sudo systemctl restart coffeecam-web.service`
(unit runs `python -m coffeecam.server`; dashboard on `192.168.50.16:8000`).

---

## B. Full-dataset annotation audit

### Can `viewer` do it? No.
`coffeecam/viewer.py` runs the **detector** and draws *its* box on a **160-frame
sample** (`_sample(refs, max_frames)`, `viewer.py:76`). It never draws the saved
human box, and it subsamples. Same for `summary.py` / `compare.py` — all draw
model output, not ground truth. Their helpers are the right base to reuse:
`summary.collect_frames()`, `summary._render_frame()` (drawing/label path),
`annotations.load()`, `dataset.bbox_*` conversions, `detect.load_model()`.

### Plan — new `coffeecam/annotation_audit.py` (+ optional `/annotate/audit` route)

**Layer 1 — automated triage** (don't eyeball 500+ frames):
For every non-skip box row in `annotations.jsonl`:
- load its frame, run the detector (`detect.detect_on_frame`, same path as
  `/annotate/suggest`), take the best detection box.
- compute `IoU(saved_box, best_det_box)`.
- bucket:
  - `IoU ≥ 0.5` → agrees with frame content → **OK**
  - `IoU < 0.2` **and** a detection exists with conf ≥ ~0.25 elsewhere in the
    frame → **LIKELY MISMATCH** (box belongs to a different frame)
  - no detection / low conf → **INCONCLUSIVE** → needs a human look
- emit `annotation_audit.csv`: `rel, saved_box, det_box, det_conf, iou, bucket`,
  sorted worst-first.

**Caveat:** the only checkpoint on hand is `trackB-v1` (~0.16 session mAP50-95 —
see `docs/dataset-experiments-log.md`). "No detection" will be common and the
triage is noisy. It reliably catches gross misses (pot absent, box in empty
space, hand-in-frame); it will not catch a box that's 40 px off. Use
`models/best-trackB-sess.pt` (0.487 session mAP50) instead if promoted, or wait
for `imgsz960-sess`.

**Layer 2 — visual contact sheet** for `LIKELY MISMATCH` + `INCONCLUSIVE`:
montage JPGs, ~20 frames/page, each = frame + **saved box (red)** + **detector
box (green)** + `rel` caption. Reuse `_render_frame`'s draw/label code. Write
`audit_contact/page_XX.jpg`. Page through, note rels to re-label.

**Layer 3 — fullness** (`fullness.jsonl`): can't IoU a scalar level. Cheapest
check: crop each labelled frame by its GT box (as `/fullness/crop` does),
montage grouped by assigned level, eyeball for crops that obviously don't match
their bin. Lower priority than boxes.

### Fixing what the audit finds
Re-labeling by hand via the (fixed) `/annotate` UI is simplest — filter to the
suspect rels. If the offset turns out to be **systematic** (every bad row is
exactly +k frames), a scripted shift of `rel` on the affected rows is possible,
but verify the constant first on 10+ hand-checked rows; don't assume.

### Deliverables
- `coffeecam/annotation_audit.py` + `tests/test_annotation_audit.py`
- `annotation_audit.csv` over all rows
- `audit_contact/` contact sheets for the flagged set
- a short results note appended to this file: how many rows in each bucket, and
  whether the offset is systematic or random

---

## Suggested order
1. Do the audit (B) **first** on the current data — establishes blast radius and
   whether earlier days are affected, before more labels pile on.
2. Fix the bug (A).
3. Re-label the flagged rels through the fixed UI.
4. Re-run the audit to confirm the flagged set shrank; record final numbers here.

---

## Results (2026-09-04)

### A — fix landed
`coffeecam/server.py`: `/annotate/frame/<i>.jpg` → `/annotate/frame.jpg?rel=`,
`/annotate/suggest/<i>.json` → `/annotate/suggest.json?rel=`, and the two
`/fullness/{crop,frame}/<i>.jpg` routes → `?rel=` likewise. New
`_safe_capture_path()` resolves `rel` under the captures dir and 400s on `..`
traversal. All four now send `Cache-Control: no-store` (was `max-age=300` on the
unstable ordinal URL). The `/fullness/crop.jpg` route re-reads the GT box from
`annotations.jsonl` by `rel` instead of trusting `picked[i]`. Embedded JS in
both `_ANNOTATE_PAGE` and `_FULLNESS_PAGE` now builds image/suggest URLs from
`f.rel` (`encodeURIComponent`) — `i` is only an on-screen cursor now.
Tests: `tests/test_server.py` — rewrote the `<i>` route tests, added
`test_annotate_frame_by_rel_is_stable_across_saves` (the regression: labeling
one frame no longer shifts the bytes another `rel` serves), plus no-store and
traversal-guard assertions. Full suite green (236 tests).
**Deploy:** `sudo systemctl restart coffeecam-web.service` on 192.168.50.16.
Not yet restarted at time of writing.

### B — audit run
New `coffeecam/annotation_audit.py` (+ `tests/test_annotation_audit.py`, 18
cases). Run over all 573 box/negative rows with `models/best-trackB-sess.pt`
(0.487 session mAP50 — better than the `trackB-v1` the CHECKPOINT points at):

| bucket | rows | % |
|---|---|---|
| OK (IoU ≥ 0.5 vs detector) | 379 | 66 % |
| LIKELY_MISMATCH (IoU < 0.2, detector conf ≥ 0.25) | 80 | 14 % |
| INCONCLUSIVE (no / weak detection, or 0.2–0.5 IoU) | 70 | 12 % |
| NEGATIVE_HAS_POT (row says "no pot", detector conf ≥ 0.5) | 44 | 8 % |

Artifacts: `annotation_audit.csv` (worst-first), `audit_contact/page_01..10.jpg`
(frame + saved box red + detector box green), `audit_contact/fullness_*.jpg`
(GT-box crops grouped by fill level).

**Flagged rate is far higher on the earliest days** — the bug has been present
from the start and was worst early:

| day | flagged (MISMATCH+NEG) / total |
|---|---|
| 2026-08-31 | 37 / 65  (57 %) |
| 2026-09-01 | 41 / 147 (28 %) |
| 2026-09-02 | 12 / 108 (11 %) |
| 2026-09-03 | 34 / 227 (15 %) |
| 2026-09-04 |  0 / 26   (0 %) |

**Offset is NOT systematic.** Probing each MISMATCH row's saved box against
detections on frames −2…+7 away: 47 of 80 have no confident (≥0.5) detection at
*any* offset (weak checkpoint); the rest scatter, with only a mild lean toward
+5…+7 (~21 rows). No single constant → **do not script a global `rel` shift**;
re-label by hand.

Visual spot-check of `page_01.jpg` confirms the LIKELY_MISMATCH bucket is real,
not model noise: red saved boxes sit in empty counter space (roughly where a
carafe would be if set down left of the machine) while the pot is clearly
present elsewhere and detected at conf 0.83–0.97. `fullness_low.jpg` shows the
same contamination reached `fullness.jsonl` — several "low" crops are a wrist, a
blurred hand, or bare counter.

### Still to do (human)
1. Page through `audit_contact/page_*.jpg`; for each bad frame note its `rel`.
2. In the fixed `/annotate` UI, filter to those days (esp. 2026-08-31) and
   re-label. Delete + redo the 44 `NEGATIVE_HAS_POT` rows first — cleanest signal.
3. Re-label fill levels for the same rels via `/fullness`.
4. Re-run `python -m coffeecam.annotation_audit --contact --fullness` and check
   the flagged counts dropped; append final numbers here.
