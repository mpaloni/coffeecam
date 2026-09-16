# Capturing frames from the break-room camera

`coffeecam/capture.py` pulls frames off the office coffeecam and writes them,
privacy-cropped, for the dataset build. It replaces taking screenshots by hand.

## Camera / tunnel

The camera lives on the office network and is **not routable from the homelab
LAN**. It is exposed at `http://192.168.50.10:8888` (the real address lives in
`hosts.env`, gitignored — see `hosts.env.example`; `coffeecam/hosts.py` loads
it into `COFFEECAM_SOURCE_URL`) by an SSH tunnel documented in homelab-docs
`docs/runbooks/coffeecam-tunnel.md` (make it the persistent systemd-user
variant on `.10` before relying on any of this):

| URL | Content |
|---|---|
| `…:8888/snapshot` | one JPEG, `1280x720`, ~120 KB |
| `…:8888/stream` | MJPEG `multipart/x-mixed-replace` |

**Raw frames are upside-down and unmasked.** The 180° rotation and the privacy
crop (`inset(18.1% 43.4% 32.9% 23.4%)`) exist only in the CSS of the viewer page.
`capture.py` re-applies both before writing to disk (`--raw` opts out — don't,
except for debugging). Cropped + rotated is `424x353`, ~27 KB at quality 85.

## Usage

```bash
# one frame now
.venv/bin/python -m coffeecam.capture --once

# loop: one frame/min, Mon-Fri 07:00-18:59, cropped, no dedup
.venv/bin/python -m coffeecam.capture --interval 60 --days mon-fri --hours 7 19

# loop with change detection: sample the stream every 2s, keep only distinct
# frames + one forced frame every 5 min
.venv/bin/python -m coffeecam.capture --mode stream --interval 2 \
    --dedup --dedup-threshold 8 --heartbeat-secs 300 --days mon-fri --hours 7 19
```

Output layout:

```
captures/
  2026-08-30/
    070001.jpg
    070101.jpg
    ...
    index.jsonl        # one line per frame, kept or skipped
```

`index.jsonl` rows: `{t, kept, file?, bytes?, hash, dist, heartbeat?, src}`.
Skipped (deduped) frames are logged too, so the file doubles as an event log
(`dist` = perceptual-hash distance from the last kept frame). On restart the loop
reseeds its "last kept" hash from today's `index.jsonl`.

`captures/` is git-ignored — copy the frames you actually want to label into
`dataset/images/` and annotate them there.

## Running it as a service

Two user units in `deploy/systemd/` — enable **one**:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/coffeecam-capture.{service,timer} ~/.config/systemd/user/    # 1/min baseline
#   or
cp deploy/systemd/coffeecam-capture-loop.service   ~/.config/systemd/user/     # dedup stream

systemctl --user daemon-reload
systemctl --user enable --now coffeecam-capture.timer         # (baseline)
#   or
systemctl --user enable --now coffeecam-capture-loop.service  # (dedup)

sudo loginctl enable-linger "$USER"   # keep it running with no login session
```

Both assume the repo at `~/claude-workspace/services/coffeecam` with its `.venv`.

## Storage estimates

Workday = Mon–Fri (~22/month, ~261/year). "12 h" = a 07:00–19:00 window.
Cropped frame ≈ 27 KB (the default on disk); raw ≈ 120 KB (`--raw`, ~4.4×).

### Frame counts

| interval | /hour | /12 h day | /24 h day |
|---|---|---|---|
| 1 / min | 60 | 720 | 1,440 |
| 1 / sec | 3,600 | 43,200 | 86,400 |

### Cropped @ 27 KB (default)

| interval / window | per workday | per work-month | per work-year |
|---|---|---|---|
| 1/min, 12 h | 19 MB | 425 MB | **5.0 GB** |
| 1/min, 24 h | 39 MB | 850 MB | 10 GB |
| 1/sec, 12 h | 1.2 GB | 26 GB | 305 GB |
| 1/sec, 24 h | 2.3 GB | 51 GB | 610 GB |

### Raw @ 120 KB (`--raw`)

| interval / window | per workday | per work-month | per work-year |
|---|---|---|---|
| 1/min, 12 h | 84 MB | 1.9 GB | 22 GB |
| 1/sec, 12 h | 5.2 GB | 114 GB | 1.3 TB |

### With `--dedup` (sample 1/sec, keep only distinct + 5-min heartbeat)

A static break-room scene is unchanged 95–99 % of the time. Expected keep-rate
1–5 %:

| keep rate | effective frames / 12 h | cropped/day | cropped/work-year |
|---|---|---|---|
| 2 % | ~860 | 23 MB | 6 GB |
| 5 % | ~2,160 | 58 MB | 15 GB |

→ dedup gives 1/sec temporal resolution on the interesting moments (pour, refill,
someone walking through, lights changing) for roughly the storage of the plain
1/min plan.

## Recommended starting point

`coffeecam-capture.timer` — **1/min, cropped, 12 h workday window** (~5 GB/yr).
It closes the dataset's lighting / fill-level / day-to-day gaps within a couple of
weeks. Switch to the dedup loop later if you want motion/event frames.
