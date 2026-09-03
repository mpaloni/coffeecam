# Handoff — pot-fullness classifier

**State as of 2026-09-03.** The full plan (`docs/fullness-plan.md`) is **built and
shipped** — steps 1–5 done, `fullness-v1` trained and live on `coffeecam-web`.
What remains is step 6: more data (distinct brew events, esp. a real `full`),
then retrain. `main` is at `97ae2c6`, pushed to Gitea. Labels have grown since
(262 labelled / 460 boxes) but no new *distinct* brew events, so no retrain — see
"Data reality".

## TL;DR for next session

- The fullness pipeline is done and running. `fullness.default_estimator()` →
  `ModelFullness` (yolov8n-cls) when `models/FULLNESS_CHECKPOINT` resolves, else
  `NullFullness`. `coffeecam-web` restarted on it — `/fullness.json` returns real
  class probs + a 0–1 fill score.
- **fullness-v1** (`runs/classify/fullness-v1`, gitignored): test balanced
  accuracy **0.667**, `has_coffee` recall 0.818, fill-score Spearman +0.68 on 34
  real held-out frames. Beats the old brightness heuristic (bal acc 0.208,
  Spearman +0.05). Full matrix + comparison: `docs/training-status.md`.
- **Next lever is data, not model.** 34-frame test set, mostly heartbeat dupes
  from a few brew events; `full` has 5 labels total, 0 in test. Seed more brew
  events (see "What's left"), rerun `python -m coffeecam.fullness_train`,
  eyeball `/fullness/compare.gif`.
- Test suite: `.venv/bin/python -m pytest -q` → **217 passing**.

## What each piece is

| module | role |
|---|---|
| `coffeecam/fullness_crop.py` | `prepare_crop(frame, box)` + `DEFAULT_POT_BOX (291,113,373,205)` + `CROP_SIZE 96`. Expand box → fixed 83:92 aspect → clamp → letterbox → resize. Never None. The one crop transform, train and inference. |
| `coffeecam/fullness_labels.py` | Sidecar store `captures/fullness.jsonl` (mirrors `annotations.py`). `LABELS = LEVELS + ("absent",)` = `empty/low/half/high/full/absent`. `upsert/remove/skip/skip_many/load/counts/positive_box_rels`; CLI `stats`, `skip-unlabeled`. |
| `coffeecam/fullness_dataset.py` | Builds `fullness_dataset/{train,val,test}/{class}/*.jpg` of `prepare_crop` outputs. Split = `coffeecam.dataset._split_bucket` (**same hash+seed as the detector** → no cross-task leakage). `--merge {none,coarse,binary}`. `--balance` oversamples *train* to ~1:1:1 with `jitter_box`'d variants (scale ×0.85–1.20, each side ±12 px), cap 8/frame; val/test untouched. Tree wiped+rebuilt each run (guarded). |
| `coffeecam/fullness_train.py` | Rebuilds tree (`--merge coarse --balance` default) → fine-tunes `yolov8n-cls.pt` imgsz 96 → writes `models/FULLNESS_CHECKPOINT`. `--no-build` to reuse the tree. |
| `coffeecam/fullness.py` | `ModelFullness` (lazy YOLO), `resolve_fullness_weights()` (None on a fresh clone, not an error), `default_estimator()`. `score` = prob-weighted fill scalar via `_FILL_SCALAR`, `None` when argmax is `absent`. `BrightnessFullness` kept reference-only. |
| `coffeecam/fullness_compare.py` | `build_test_gif()` → `(gif_bytes, scoreboard)`: fullness-v1 vs brightness over the test split, per-class probs as a legend. |

### server routes (`coffeecam/server.py`)

- `GET /fullness` + `/fullness/{queue.json, crop/<i>.jpg, frame/<i>.jpg, label,
  label/delete, skip, skip-queue}` — labeling UI, mirrors `/annotate`. Queue =
  frames with a positive box in `annotations.jsonl`. Crop endpoint serves
  `prepare_crop(frame, GT_box)`. Hotkeys `1`–`5` (fill), `w` (`absent` = no pot),
  `s` (skip/watched), filter + stride.
- `GET /fullness/compare.gif` + `.json` — the model-vs-heuristic walkthrough.
- `GET /artifacts` + `/artifacts/<path>` — read-only gallery of
  `COFFEECAM_ARTIFACTS_DIR` (default `scratchpad/`). Drop a GIF/PNG/report in,
  refresh, no restart. Images + `.json/.txt/.csv/.md/.log`; traversal 404s,
  other suffixes 415.
- The pipeline worker builds `default_estimator()` **once** and passes it to
  every `run_pipeline` call.

## Decisions log

- **5-level scale, not the planned 3-way.** `LEVELS` reuses
  `coffeecam.fullness.LEVELS` (`empty/low/half/high/full`), shown as 1–5. User
  asked for the finer scale; matches the enum `FullnessResult` already returns.
- **`coarse` merge for v1 training** — `empty` / `some` (low+half) / `lots`
  (high+full) / `absent`. `none` leaves `full` with 0 test frames; `binary`
  loses "how full". Collapse further at `fullness_dataset` time if `some`/`lots`
  stay unlearnable — don't re-label.
- **`absent` is its own label**, not a `skip`. Key `w`, "no pot in frame".
  Carafe genuinely off the warmer. `skip`/watched stays for glare/blur/
  unlabelable or a wrong GT box you don't want to fix now.
- **Train/serve skew handled by augmentation** — `--balance` bakes box-jittered
  crops into the train tree rather than a custom dataset class.

## What's left

1. **Seed more brew events, esp. `full`.** Right after someone brews:
   `.venv/bin/python -m coffeecam.capture --mode stream --interval 5` for ~15 min,
   a few times across lighting. `empty` accumulates for free. Target ~40–60
   *distinct* brew events (not frames — heartbeat dupes don't add signal). Then
   label via `/fullness`, `python -m coffeecam.fullness_train`, compare.
2. **Back up the labels.** `captures/` is fully gitignored, so
   `captures/fullness.jsonl` (262 labels) + `captures/annotations.jsonl` (460
   boxes) are unbacked-up manual work. Add `captures/*` + `!captures/fullness.jsonl`
   + `!captures/annotations.jsonl` to `.gitignore` (blanket `captures/` can't be
   un-ignored file-by-file — needs the `/*` form). Left undone pending a nod:
   it changes repo-wide tracking.
3. ✅ `docs/training-status.md` "Current state" block reconciled (2026-09-03) —
   now says the detector + fullness work is on `main`, 217 tests passing, no open
   PR. The detector annotation breakdown in that block (457 rows / 243 positive)
   is still the pre-relabel snapshot; the store is now 797 frames / 460 positive
   boxes — refresh it next time detector training is touched.
4. Optional: `detect.resolve_weights()` fallback to `models/*.pt` so a fresh
   clone runs the detector without `runs/` (unrelated to fullness).

## Labeling guidance

The `/fullness` crop is `prepare_crop(frame, GT_box)` where `GT_box` is the
**hand-drawn** box from `/annotate`, not the live detector — so the crop is only
wrong when that box was drawn sloppily.

- **Crop loose / clipped but still on the carafe** → label from whichever image
  reads clearer (full frame shown alongside). A correct level on a slightly-off
  crop is still good signal — box jitter covers it.
- **Crop on the wrong thing entirely** (wall, mug, bare counter — mis-drawn/stale
  box): **don't label it.** The crop *is* the training input. Re-draw the box in
  `/annotate`, or `skip`. Not `absent`.
- **`skip`/watched** when *neither* image lets you judge, or for a wrong-box
  frame you won't re-annotate now.
- **`w`/`absent`** when the carafe isn't on the warmer at all.

## Data reality

- 460 frames with a hand-drawn `coffee_pot` box (was 373 at first handoff); box
  size stable (`w` 83±10, `h` 92±8), position mostly stable (median
  `291,113→373,205`), nudged tail.
- Fullness labels (2026-09-03, updated): 262 labelled + 198 watched, all 460
  positive-box frames resolved. `empty 58 · low 76 · half 48 · high 33 · full 5
  · absent 42`. **`full` still only 5** — no new distinct brew events captured
  yet, so fullness-v1 is unchanged (`runs/classify/fullness-v1`, weights from
  13:29). Retrain is still gated on step 1 below.
- Non-empty frames cluster into a handful of brew events with many near-duplicate
  heartbeat frames — count *distinct brew events*, not frames.

## Not versioned

- `captures/` (frames + both `.jsonl` label stores), `runs/` (all weights incl.
  `runs/classify/fullness-v1`), `fullness_dataset/` (derived, rebuilt on demand),
  `scratchpad/`. `models/FULLNESS_CHECKPOINT` and `models/CHECKPOINT` (pointers)
  *are* committed; they point into `runs/`, so a fresh clone falls back to
  `NullFullness` / newest-by-mtime.

## Env / commands

```bash
cd services/coffeecam
.venv/bin/python -m pytest -q                              # 217 passing
.venv/bin/python -m coffeecam.fullness_dataset --merge coarse --balance
.venv/bin/python -m coffeecam.fullness_train               # rebuild + train + promote
.venv/bin/python -m coffeecam.fullness_compare             # -> scratchpad/fullness-v1-test.gif
systemctl --user restart coffeecam-web.service             # dashboard on :8000
```

Branch note: `services/coffeecam` clones can sit on a stale feature branch —
`git branch --show-current` before editing. Currently on `main`.
