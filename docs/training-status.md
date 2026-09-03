# Training status & next steps

Living status of the coffee-pot detector: what's live, what's training, what to do next.
How the labeling → dataset → train → compare loop works is in `dataset-and-training.md`.

Last updated: 2026-09-03 (fullness-v1 trained; detector "Current state" reconciled — work
is on `main`, 217 tests passing).

---

## Current state

- **Live model:** `models/CHECKPOINT` → `runs/detect/runs/trackB-v1` (`weights/best.pt`,
  md5-identical to the loose `models/best-trackB-v1.pt`). `coffeecam-web.service` **restarted
  on it 2026-09-02** — `/compare.json?set=test` live shows checkpoint 36/3/4, matching
  offline. Live smoke: conf 0.827 on a real test frame.
  - **Held-out 43-frame test split: mAP50 0.603 / mAP50-95 0.438 / P 0.690 / R 0.556;
    strong/weak/none 36/3/4.** Early-stopped epoch 48 (best epoch 33, val mAP50 0.725 /
    mAP50-95 0.527), 3.2 h on k8s worker2, `yolov8n` imgsz 640 / batch 16 / `mosaic=0`.
  - Prior live model `balanced-v1` re-scored on this same split: **mAP50 0.323 / mAP50-95
    0.246** / P 0.667 / R 0.361; 26/9/8. trackB-v1 beats it on every metric
    (Δ +0.280 mAP50 / +0.192 mAP50-95) — biggest jump in the series. Confirms real scene
    count (117 → 243 labels), not augmentation, is the lever.
  - `nomosaic-2` (2026-08 baseline) on this split: 0.216 / 0.065.
  - **`detect.DEFAULT_IMGSZ` 320 → 640** (trackB-v1's train size; 320 lost small-pot
    recall). `test_pipeline.py` assertion updated.
- **Labeling:** queue empty. `captures/annotations.jsonl` = 457 rows — **243 positive**, 45
  negative, 169 watched. 0 unlabeled.
- **Dataset on disk:** the Track B 1202-frame set (147 real positives + 1026 synthetic + 29
  negatives; val 40 / test 43, real-only). Built at ~214 positives — a fresh rebuild at 243
  would be slightly bigger; do that before the *next* train.
- **Compare artifact:** `scratchpad/compare/compare-trackB-test.gif` (+ `.json`).
- **Committed & merged to `main`:** the `feat/annotate-endpoint` work (`58f3624`), the
  `trackB-v1` weights commit (`35b51b0`), and all fullness steps (1–5 + compare/artifacts,
  through `97ae2c6`). No open PR — landed directly on `main`.
- Test suite: **217 passing** (`-p no:randomly`; one `test_augment_shift` case is flaky
  under random ordering — pre-existing test-isolation pollution, passes in isolation).

---

## Next steps, in order

1. ✅ **Retrieved `trackB-v1`** → `runs/detect/runs/trackB-v1/` + `models/best-trackB-v1.pt`.
2. ✅ **Scored on the 43-frame test split** — 0.603 / 0.438 (see Current state).
3. ✅ **Promoted.** `models/CHECKPOINT` → `runs/detect/runs/trackB-v1`; `coffeecam-web.service`
   restarted; `DEFAULT_IMGSZ` → 640. Still to do: re-run `compare --set captures` after a
   day of fresh frames to confirm live behaviour.
4. ✅ **Cleanup.** `coffeecam-train` pod deleted (freed its 8 CPU / 10Gi worker2
   reservation + 209M emptyDir). **Namespace `coffeecam-train` kept, empty** — next retrain
   skips `create ns`, just `kubectl apply -f scratchpad/coffeecam-train-worker2.yaml`.
   worker2 healthy: no taints, DiskPressure `False`.
5. ✅ **Caveat rewritten** — `README.md` (train section), `docs/PIPELINE.md` (Weights +
   Known limitations), `TODO.md` "Model / training" bullet. All now cite test mAP50 0.603 /
   mAP50-95 0.438 on 43 held-out frames and keep the "bottleneck is ~243 real positives,
   not augmentation" point. *(These edits pre-date the split into `dataset-and-training.md`;
   reconcile if that doc now owns the same material.)*
6. ✅ **Landed on `main`** (no PR — committed directly). Carried: `/annotate` +
   `dataset.promote`, the bundled summary/viewer work, `augment_shift` rotate/occlude/
   balanced, `compare.py` + `/compare` + `/summary?set=`, the skip/watched state, the Track B
   dataset, trackB-v1 promotion (`CHECKPOINT` + `DEFAULT_IMGSZ`), and these docs.

7. Then consider dropping the `normalize` (black-pad) pipeline stage and tightening
   `pipeline.DEFAULT_CONF` (`TODO.md` follow-ups).

### The longer game

Synthetic augmentation is at its ceiling for this eval (trackA proved it). The next real
gain is more **real** scenes: keep `/annotate` going toward ≥ 300–400 positives, build the
§6.2 luma-diff distinct-only filter so the queue stops surfacing heartbeat dups, let
val/test grow, then re-augment from the bigger real base.

---

## Retrain runbook — k8s worker2

worker2 (10 CPU / 15.8 GiB allocatable) is the node for imgsz-640 runs. The coffeecam box
OOMs above imgsz 320; worker1 OOMs above imgsz 416.

```bash
kubectl create ns coffeecam-train   # skip — the empty ns is kept between runs
kubectl apply -f scratchpad/coffeecam-train-worker2.yaml   # plain Pod, nodeSelector worker2,
kubectl wait --for=condition=Ready pod/coffeecam-train -n coffeecam-train --timeout=180s  # emptyDir /work, sleep infinity, req 8 CPU / 10Gi

# ship code + data + base weights — NOT dataset/previews (QA only, ~140 MB). ~200 MB, ~1 min.
tar czf - coffeecam yolov8n.pt requirements.txt pytest.ini \
    dataset/images dataset/labels dataset/train.txt dataset/val.txt dataset/test.txt dataset/data.yaml \
  | kubectl exec -i coffeecam-train -n coffeecam-train -- tar xzf - -C /work
kubectl exec coffeecam-train -n coffeecam-train -- pip -q install -e /work 2>/dev/null || true

kubectl exec coffeecam-train -n coffeecam-train -- bash -lc '
  cd /work && nohup yolo detect train \
    model=yolov8n.pt data=dataset/data.yaml \
    epochs=60 imgsz=640 batch=16 mosaic=0 close_mosaic=10 \
    cache=ram patience=15 project=runs name=trackB-v1 \
    > /work/trackB-v1.log 2>&1 &'

# monitor
kubectl exec coffeecam-train -n coffeecam-train -- bash -lc '
  tail -c 300 /work/trackB-v1.log | tr "\r" "\n" | grep -E "^\s+[0-9]+/60" | tail -1
  awk -F, "NR>1{printf \"%s mAP50=%.3f mAP50-95=%.3f\n\",\$1,\$8,\$9}" \
    /ultralytics/runs/detect/runs/trackB-v1/results.csv | tail -10
  grep -aE "epochs completed|early stopping|Results saved to" /work/trackB-v1.log'
```

**worker1 fallback** (smaller box): `imgsz=416 batch=4 cache=disk`, mem limit 5500Mi. That
is how `balanced-v1` actually ran. **coffeecam box fallback:** `imgsz=320 batch=8 mosaic=0
--cache ram`, ~1 h; stop `coffeecam-web.service` + pause capture first to free RAM; imgsz
640 OOM-kills it.

### Rebuild the dataset from scratch

From `captures/annotations.jsonl`. Keep real:synth near ~1:7 (trackA showed ~1:24 hurts
mAP50-95).

```bash
rm -f dataset/images/*_shift_x* dataset/labels/*_shift_x* dataset/previews/*_shift_x* \
      dataset/labels.cache dataset/images/kahvi* dataset/labels/kahvi* dataset/previews/kahvi*
echo '[]' > dataset/augmentations.json
.venv/bin/python -m coffeecam.dataset promote --drop-kahvi
.venv/bin/python -m coffeecam.augment_shift --generate-from dataset/train.txt --balanced \
      --seed 0 --max-shift 100 --occ-min-cover 0.05
.venv/bin/python -m coffeecam.dataset promote --drop-kahvi
```

---

## Gotchas

- **Mutating `kubectl`** (`apply` / `exec` / `cp` / `delete pod`) is **classifier-blocked in
  some sessions**, allowed in others. `kubectl delete namespace` and `kubectl debug node/…`
  have been blocked. If blocked, the operator runs them by hand.
- **`pytorch/pytorch:2.4.1-cpu` does not exist** — the pod uses
  `ultralytics/ultralytics:latest-cpu` (torch + ultralytics + cv2 preinstalled, no pip step).
- **ultralytics writes to `/ultralytics/runs/detect/runs/<name>/` inside the pod**, not
  `/work/runs/...` — `kubectl cp` from the former.
- **Don't `kubectl cp` `dataset/previews/`** — QA only, ~140 MB.
- **worker2 DiskPressure:** the 648 MB image pull once auto-tainted worker2
  `node.kubernetes.io/disk-pressure:NoSchedule`. Fix as root on the node
  (`sudo crictl rmi --prune`) — do **not** just add a toleration; a disk-pressured node
  evicts the pod mid-run. It self-recovered within ~1 h last time.
- **Cluster headroom is thin.** 2026-09-01: a 62-day-broken `semantic-router` deploy spawned
  ~4900 failed pods → ~6500 dead pod objects → apiserver OOM-loop on the 3.3 GiB master.
  Recovered (reboot, drain, delete Failed/Succeeded pods, `etcdctl defrag`, taint master
  control-plane-only, `--terminated-pod-gc-threshold=1000`). `semantic-router` left scaled
  to 0 — leave it. Nothing coffeecam-related caused it, but prefer the box for small runs.

---

## Fullness classifier (`yolov8n-cls`)

Separate model from the detector — reads the `prepare_crop` output, predicts fill
level. Full design: `fullness-plan.md`. Pointer: `models/FULLNESS_CHECKPOINT`
(mirrors `models/CHECKPOINT`); pipeline picks it up via
`fullness.default_estimator()`, falls back to `NullFullness` on a fresh clone.

**fullness-v1 (2026-09-03)** — `runs/classify/fullness-v1`, `--merge coarse`
(`empty` / `some` = low+half / `lots` = high+full / `absent`), 80 epochs, imgsz
96. Train oversampled to ~1:1:1 with box-jittered crops (361 imgs); val/test at
natural prevalence (34 / 34 real held-out frames, same hash-split as the
detector).

Test split (34 frames — **never report raw accuracy**, majority class ≈ 53 %):

| metric | value |
|---|---|
| balanced accuracy | **0.667** |
| recall `empty` | 0.50 |
| recall `some` | 0.667 |
| recall `lots` | 0.75 |
| recall `absent` | 0.75 |
| recall `has_coffee` (some+lots merged) | 0.818 |
| raw accuracy | 0.647 (22/34) |

Confusion (rows = true, cols = pred):

```
         absent  empty   lots   some
absent      3      0      0      1
 empty      1      4      0      3
  lots      0      0      3      1
  some      2      2      2     12
```

Read: `empty`↔`some` is the main confusion (early-morning dark frames), and the
34-frame test set is mostly heartbeat near-dupes from a handful of brew events,
so treat ±0.1 as noise. **Next lever is more distinct brew events** (esp. a real
`full`), not a bigger model. Re-run: `python -m coffeecam.fullness_train`.

**vs the retired `BrightnessFullness` heuristic** on the same 34 frames: balanced
acc 0.208, fill-score Spearman +0.05 (i.e. uncalibrated noise — it predicts
`empty` zero times and is blind to `absent`). fullness-v1: balanced acc 0.667,
Spearman +0.68. Confirms `fullness-plan.md`'s premise that lighting dominates the
brightness signal. Per-frame walkthrough GIF: `/fullness/compare.gif` (or
`python -m coffeecam.fullness_compare`), JSON scoreboard at `/fullness/compare.json`.

---

## History

**2026-09-02 capture-backlog purge** (one-off, not a feature): deleted 240 stale
un-annotated frames from `captures/` — reviewed and passed over during a labeling pass.
Backup: `scratchpad/seen-frames-backup-20260902.tgz` (`tar xzf … -C captures` to restore).
`/summary`, `/viewer`, `compare --set captures` lose that slice of (mostly heartbeat-dup)
history. Going forward the `s` key / "skip rest of queue" / `skip-unlabeled` CLI are the
non-destructive way to keep the queue clear.
