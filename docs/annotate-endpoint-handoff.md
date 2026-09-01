# `/annotate` endpoint — progress handoff

Date: 2026-09-01. Branch: **`feat/annotate-endpoint`** (1 commit, `4957ab8`, not pushed, no PR yet).
Plan: `docs/annotate-endpoint-plan.md`. Pipeline docs updated in `docs/PIPELINE.md`.

## TL;DR

The labeling tooling from the plan is **built, tested, deployed, and in use**. A first
labeling pass is ~40% done (117 frames). Nothing has been promoted to `dataset/` and the
detector has **not** been retrained yet. The UI has only been exercised by hand, not in an
automated browser test.

## What was built (steps 1–5 + 7 of the plan)

| File | What |
|---|---|
| `coffeecam/annotations.py` *(new)* | Sidecar label store `captures/annotations.jsonl`. `load` / `upsert` / `remove` / `validate_boxes`. Pure, lock-free, atomic tmp-file + `os.replace` writes, rows sorted by `rel`. `[]` boxes = explicit negative. |
| `coffeecam/annotate.py` | Extracted `write_example(image_path, boxes, *, images_dir, labels_dir, class_id=0, dest_name=None)` — the shared "copy image + write multi-box YOLO label" primitive. `annotate_image()` CLI (with preview PNG) still works, now delegates label writing to `write_label_lines`. |
| `coffeecam/dataset.py` | `promote(...)` — reads `annotations.jsonl`, copies frames to `dataset/images/` as `cap_YYYYMMDD_HHMMSS[_fff].jpg`, writes YOLO labels (empty file for a negative), regenerates `dataset/{train,val,test}.txt`, adds `test:` to `data.yaml`. Deterministic SHA1-of-`rel` split (`val_frac`/`test_frac` 0.15), synthetic `kahvi*` pinned to train, real frames the only val/test candidates. Idempotent. CLI: `python -m coffeecam.dataset promote [--dry-run] [--no-negatives]`. |
| `coffeecam/server.py` | `/annotate` page (`_ANNOTATE_PAGE`, fully inline canvas UI) + routes: `queue.json`, `frame/<i>.jpg` (raw, no box drawn), `suggest/<i>.json` (live-model prefill, 503 if no model), `POST label`, `POST label/delete`, `POST promote?confirm=1`. New global `_annot_lock`. New env `COFFEECAM_DATASET_DIR` (default `dataset/`). `/annotate` linked from the `/` index footer. |
| tests | `tests/test_annotations.py` (new, 19), promote tests in `test_dataset.py` (9), route tests in `test_server.py` (11). Full suite **113 passing**. |
| docs | `README.md` §1c, `docs/PIPELINE.md` "/annotate — browser labeling" section, `TODO.md` Labeling bullets checked off. |

The commit also folds in the **previously-uncommitted** capture-timelapse foundation this
builds on: `coffeecam/summary.py` (+ `image_summary.py`), `coffeecam/viewer.py`, their
`/summary` and `/viewer` routes and tests. `server.py` hunks for the two features aren't
cleanly separable, so they went in one commit — noted in the commit body.

## Design recap (from the plan)

- Labels go to a **sidecar** `captures/annotations.jsonl`, one upsertable row per frame
  keyed by its path under `captures/`. `captures/` is never mutated.
- `promote` is a separate, re-runnable step that builds a trainable `dataset/` from the
  sidecar. Curate labels → promote → retrain, repeat.
- Frame index `i` in the API is **ephemeral** (it indexes the current filtered/strided
  queue). Clients always send `rel`, never `i`, on writes.

## Live state

- `coffeecam-web.service` (user systemd unit) restarted 2026-09-01 ~14:45 UTC, running this
  branch's code. `http://<host>:8000/annotate`. `/healthz` 200.
- **Labeling pass in progress — 117 / 310 frames:**
  - 94 positive (has a box), 23 explicit negatives, 0 multi-box, 0 notes
  - by day: `2026-08-31` 47 (26 pos / 21 neg) · `2026-09-01` 70 (68 pos / 2 neg)
  - boxes are small (median ~5% of the 424×353 frame), pot appears in two horizontal
    zones (`cx` 0.09–0.82) — decent positional variety
  - almost all negatives are from 08-31; worth sanity-checking that day
- `promote --dry-run` today: **99 train / 22 val / 18 test, 23 negatives** (99 train = 22
  synthetic `kahvi*` + 77 real; 40 real frames held out).

## Known gaps / gotchas

1. **No "viewed but not labeled" state.** A frame you arrow past without `Space`/`x` records
   nothing and reappears in the default `filter=unlabeled` queue. ~59 near-duplicate
   heartbeat frames were deliberately skipped mid-pass and *will* come back. Mitigations
   available today: `?stride=N`, `?filter=all` (shows a "saved" badge). The real fix is the
   deferred **§6.2 "distinct-only" luma-diff filter** (~20 lines using `image_summary.py`
   mean-luma) — not built.
2. **UI has zero execution coverage.** `_ANNOTATE_PAGE` is only asserted to serve HTML with
   the right strings. The canvas coordinate mapping (§6.1, the flagged gotcha), pointer /
   corner-handle math, and key handlers have never run in a headless browser. Expect
   first-real-use bugs.
3. **`promote` default target is the real `dataset/`.** An early test run wrote into it;
   reverted, and the route now honours `COFFEECAM_DATASET_DIR`. The CLI still writes
   `dataset/` by default (intended) — be aware when experimenting.
4. Branch not pushed, no PR. The bundled summary/viewer work may warrant its own review.

## Next steps (plan §8.6–8.7 remainder)

1. Decide: keep labeling (then add the §6.2 distinct-only filter first — it pays for itself)
   or stop at the current batch.
2. `python -m coffeecam.dataset promote` → inspect `dataset/train.txt` / `val.txt` /
   `test.txt` and a few `dataset/labels/*.txt`.
3. Retrain from `yolov8n.pt` (`python -m coffeecam.train`), promote weights
   (`models/CHECKPOINT`), record **real** test-set mAP in the README — this replaces the
   long-standing "one image, overfits" caveat (README, `docs/PIPELINE.md` "Known
   limitations", `TODO.md` "Model / training").
4. If the retrained detector is good on live frames: delete the `normalize` (black-pad)
   pipeline stage and tighten `pipeline.DEFAULT_CONF` (`TODO.md` follow-ups).
5. Optional UI polish from real use: multi-box add button, 8-handle resize (only 4 corners
   now), headless-browser smoke test.
6. Push branch, open PR (consider splitting summary/viewer vs annotate for review).

## How to pick up

```bash
cd ~/claude-workspace/services/coffeecam
git checkout feat/annotate-endpoint
.venv/bin/python -m pytest -q                      # 113 passing

# label: open http://<host>:8000/annotate   (Space save+next, x negative, h hint, z undo)
systemctl --user status coffeecam-web.service      # already running this branch

# inspect labels
wc -l captures/annotations.jsonl
python -m coffeecam.dataset promote --dry-run

# build the training set (writes real dataset/)
python -m coffeecam.dataset promote
```
