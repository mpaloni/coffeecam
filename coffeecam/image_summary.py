"""Inventory every image in the coffeecam tree, bucketed by what it *is*.

The repo accumulates images from several unrelated sources — the one real seed
frame, its `augment_shift` copies, labelled-box previews, the live capture loop's
unlabelled frames, dashboard `COFFEECAM_HARVEST` output, detector crops, training
run artifacts. "How many images do we have?" only makes sense per category, so
this walks the tree, classifies each file by path, and reports counts, sizes,
dimensions and mean luminance per bucket.

    .venv/bin/python -m coffeecam.image_summary                 # text report
    .venv/bin/python -m coffeecam.image_summary --html sheet.html   # + contact sheet
    .venv/bin/python -m coffeecam.image_summary --json > images.json

Mean luminance is included because the fullness heuristic keys off exactly that —
a category's luma range is a quick sanity check on what the detector/crop stages
are actually seeing.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageStat

IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
PRUNE_DIRS = {".venv", ".git", "__pycache__", "node_modules", ".pytest_cache", "graphify-out"}

# Ordered: first matcher to claim a path wins. (key, title, blurb, matcher)
# matcher(relposix, name, stem) -> bool
CATEGORIES: list[tuple[str, str, str, object]] = [
    ("dataset:seed", "Dataset · seed frames",
     "Real distinct frames — the actual detector training signal.",
     lambda p, n, s: p.startswith("dataset/images/") and "_shift_" not in n),
    ("dataset:augmented", "Dataset · shift-augmented",
     "augment_shift translations of the seed frame(s); cheap geometry augmentation.",
     lambda p, n, s: p.startswith("dataset/images/") and "_shift_" in n),
    ("dataset:previews", "Dataset · label previews",
     "Boxed previews for eyeballing labels. QA only — never fed to training.",
     lambda p, n, s: p.startswith("dataset/previews/")),
    ("capture:stream", "Captures · live stream (unlabelled)",
     "Frames kept by the capture loop's change-detection dedup, by day. Future training set.",
     lambda p, n, s: p.startswith("captures/") and not p.startswith("captures/pipeline/")),
    ("harvest:frame", "Harvest · pipeline frames",
     "Dashboard COFFEECAM_HARVEST=1 — full privacy-cropped frames the pipeline ran on.",
     lambda p, n, s: p.startswith("captures/pipeline/") and s.endswith("_frame")),
    ("harvest:crop", "Harvest · pipeline crops",
     "Dashboard COFFEECAM_HARVEST=1 — the detector crop for each harvested frame.",
     lambda p, n, s: p.startswith("captures/pipeline/") and s.endswith("_crop")),
    ("detector:bounded", "Detector output · bounded",
     "detect.py output: the frame with the predicted box drawn.",
     lambda p, n, s: p.startswith("crops/") and s.endswith("_bounded")),
    ("detector:crop", "Detector output · crops",
     "detect.py output: the cropped pot region.",
     lambda p, n, s: p.startswith("crops/") and s.endswith("_crop")),
    ("training:runs", "Training run artifacts",
     "Ultralytics run output — batch mosaics, PR/confusion curves, val previews.",
     lambda p, n, s: p.startswith("runs/")),
    ("misc", "Uncategorised",
     "Everything else (stray images, GIFs, one-offs).",
     lambda p, n, s: True),
]
CAT_ORDER = [c[0] for c in CATEGORIES]
CAT_META = {c[0]: (c[1], c[2]) for c in CATEGORIES}


@dataclass
class Item:
    path: str          # posix, relative to root
    category: str
    sub: str            # day / run name / "" — a within-category grouping
    w: int | None = None
    h: int | None = None
    nbytes: int = 0
    luma: float | None = None
    split: str = ""      # train / val / "" (dataset images only)
    boxes: int | None = None  # label box count (dataset images only)
    error: str = ""


@dataclass
class Bucket:
    items: list[Item] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(i.nbytes for i in self.items)

    def dims(self) -> dict[str, int]:
        d: dict[str, int] = defaultdict(int)
        for i in self.items:
            d[f"{i.w}x{i.h}" if i.w else "?"] += 1
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))

    def luma_range(self) -> tuple[float, float, float] | None:
        vals = [i.luma for i in self.items if i.luma is not None]
        if not vals:
            return None
        return min(vals), sum(vals) / len(vals), max(vals)


# ---------------------------------------------------------------------------

def find_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    here = Path.cwd()
    for cand in (here, *here.parents):
        if (cand / "dataset" / "data.yaml").exists() and (cand / "coffeecam").is_dir():
            return cand
    return here


def classify(relposix: str) -> tuple[str, str]:
    name = relposix.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    for key, _title, _blurb, matcher in CATEGORIES:
        if matcher(relposix, name, stem):
            return key, _subgroup(key, relposix)
    return "misc", ""


def _subgroup(key: str, relposix: str) -> str:
    parts = relposix.split("/")
    if key == "capture:stream" and len(parts) >= 3:
        return parts[1]                       # captures/<day>/file
    if key in ("harvest:frame", "harvest:crop") and len(parts) >= 4:
        return parts[2]                       # captures/pipeline/<date>/file
    if key == "training:runs" and len(parts) >= 2:
        return parts[-2]                      # .../<run>/file
    return ""


def load_splits(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        f = root / "dataset" / f"{split}.txt"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                out[Path(line).name] = split
    return out


def count_boxes(root: Path, stem: str) -> int | None:
    lbl = root / "dataset" / "labels" / f"{stem}.txt"
    if not lbl.exists():
        return None
    return sum(1 for ln in lbl.read_text().splitlines() if ln.strip())


def measure(path: Path, want_luma: bool) -> tuple[int | None, int | None, float | None, str]:
    try:
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            luma = None
            if want_luma:
                small = im.convert("L")
                small.thumbnail((128, 128))
                luma = round(ImageStat.Stat(small).mean[0], 1)
            return w, h, luma, ""
    except Exception as exc:  # noqa: BLE001 — a corrupt file shouldn't kill the report
        return None, None, None, str(exc)


def collect(root: Path, want_luma: bool = True) -> list[Item]:
    splits = load_splits(root)
    items: list[Item] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for fn in filenames:
            if Path(fn).suffix.lower() not in IMG_EXT:
                continue
            abspath = Path(dirpath) / fn
            rel = abspath.relative_to(root).as_posix()
            cat, sub = classify(rel)
            w, h, luma, err = measure(abspath, want_luma)
            it = Item(
                path=rel, category=cat, sub=sub,
                nbytes=abspath.stat().st_size,
                w=w, h=h, luma=luma, error=err,
            )
            if cat.startswith("dataset:"):
                stem = fn.rsplit(".", 1)[0]
                it.split = splits.get(fn, "")
                it.boxes = count_boxes(root, stem)
            items.append(it)
    return items


def bucketise(items: list[Item]) -> dict[str, Bucket]:
    buckets: dict[str, Bucket] = {k: Bucket() for k in CAT_ORDER}
    for it in items:
        buckets[it.category].items.append(it)
    return {k: v for k, v in buckets.items() if v.items}


# ---------------------------------------------------------------------------

def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} GB"


def print_report(root: Path, buckets: dict[str, Bucket]) -> None:
    total_n = sum(len(b.items) for b in buckets.values())
    total_b = sum(b.nbytes for b in buckets.values())
    print(f"\ncoffeecam image inventory   root={root}")
    print(f"{'=' * 64}")
    for key in CAT_ORDER:
        b = buckets.get(key)
        if not b:
            continue
        title, blurb = CAT_META[key]
        print(f"\n{title}   [{key}]")
        print(f"  {blurb}")
        subs = defaultdict(list)
        for it in b.items:
            subs[it.sub].append(it)
        if len(subs) > 1 or (len(subs) == 1 and "" not in subs):
            for sub in sorted(subs):
                sb = Bucket(subs[sub])
                lr = sb.luma_range()
                luma = f" · luma {lr[0]:.0f}-{lr[2]:.0f}" if lr else ""
                dims = ", ".join(f"{k}x{v}" for k, v in sb.dims().items())
                label = sub or "(root)"
                print(f"    {label:<14} {len(sb.items):>4} files · {dims} · {human(sb.nbytes)}{luma}")
        dims = ", ".join(f"{k} x{v}" for k, v in b.dims().items())
        lr = b.luma_range()
        luma = f" · luma {lr[0]:.0f}-{lr[2]:.0f} (mean {lr[1]:.0f})" if lr else ""
        print(f"  --> {len(b.items)} files · {dims} · {human(b.nbytes)}{luma}")
        if key.startswith("dataset:"):
            sp = defaultdict(int)
            for it in b.items:
                sp[it.split or "unassigned"] += 1
            labeled = sum(1 for it in b.items if it.boxes)
            sp_str = " · ".join(f"{k} {v}" for k, v in sorted(sp.items()))
            print(f"      split: {sp_str} · labelled {labeled}/{len(b.items)}")
        errs = [it for it in b.items if it.error]
        if errs:
            print(f"      !! {len(errs)} unreadable: {errs[0].path} ({errs[0].error})")
    print(f"\n{'=' * 64}")
    print(f"GRAND TOTAL: {total_n} image files · {human(total_b)}\n")


def to_json(root: Path, buckets: dict[str, Bucket]) -> str:
    out = {"root": str(root), "categories": {}}
    for key, b in buckets.items():
        title, blurb = CAT_META[key]
        lr = b.luma_range()
        out["categories"][key] = {
            "title": title, "blurb": blurb,
            "count": len(b.items), "bytes": b.nbytes,
            "dims": b.dims(),
            "luma": {"min": lr[0], "mean": round(lr[1], 1), "max": lr[2]} if lr else None,
            "items": [
                {k: v for k, v in vars(it).items() if v not in ("", None)}
                for it in b.items
            ],
        }
    out["total"] = {
        "count": sum(len(b.items) for b in buckets.values()),
        "bytes": sum(b.nbytes for b in buckets.values()),
    }
    return json.dumps(out, indent=2)


def thumb_uri(path: Path, box: int = 190) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((box, box))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=74)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def write_html(root: Path, buckets: dict[str, Bucket], out: Path, limit: int) -> None:
    total_n = sum(len(b.items) for b in buckets.values())
    ncat = len(buckets)
    parts = [f"""<!doctype html><meta charset=utf-8>
<title>coffeecam image inventory</title>
<style>
 :root {{ color-scheme: light dark;
   --bg:#f6f4ef; --card:#fff; --ink:#232019; --soft:#6b6355; --line:#dcd5c7; --accent:#a85d16; }}
 @media (prefers-color-scheme:dark) {{ :root {{
   --bg:#16130e; --card:#201b14; --ink:#ece4d5; --soft:#a89d89; --line:#3b3324; --accent:#e19345; }} }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--ink);
   font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
 header {{ padding:22px 26px 6px; }}
 h1 {{ font-size:17px; margin:0 0 3px; }}
 .meta {{ color:var(--soft); font-size:12px; }}
 section {{ padding:18px 26px; border-top:1px solid var(--line); }}
 h2 {{ font-size:13px; margin:0 0 2px; letter-spacing:.02em; }}
 .blurb {{ color:var(--soft); font-size:12px; margin:0 0 12px; }}
 .tally {{ font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--soft); margin:0 0 12px; }}
 .subhead {{ font:11px ui-monospace,Menlo,monospace; color:var(--accent); margin:14px 0 8px; letter-spacing:.05em; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:10px; }}
 figure {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
 img {{ display:block; width:100%; height:132px; object-fit:contain; background:#0002; }}
 figcaption {{ padding:5px 7px; font:10px/1.35 ui-monospace,Menlo,monospace; color:var(--soft);
   white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
 figcaption b {{ color:var(--ink); font-weight:600; }}
 .more {{ color:var(--soft); font-size:12px; margin-top:10px; }}
</style>
<header>
 <h1>coffeecam image inventory</h1>
 <div class=meta>{root} &middot; {total_n} image files &middot; {ncat} categories</div>
</header>
"""]

    for key in CAT_ORDER:
        b = buckets.get(key)
        if not b:
            continue
        title, blurb = CAT_META[key]
        lr = b.luma_range()
        luma = f" &middot; luma {lr[0]:.0f}&ndash;{lr[2]:.0f}" if lr else ""
        dims = ", ".join(f"{k}&times;{v}" for k, v in b.dims().items())
        parts.append(f"<section><h2>{title}</h2><p class=blurb>{blurb}</p>")
        parts.append(f"<p class=tally>{len(b.items)} files &middot; {dims} &middot; {human(b.nbytes)}{luma}</p>")

        subs: dict[str, list[Item]] = defaultdict(list)
        for it in b.items:
            subs[it.sub].append(it)
        shown = 0
        for sub in sorted(subs):
            group = subs[sub]
            if sub:
                parts.append(f"<div class=subhead>{sub} &nbsp;({len(group)})</div>")
            parts.append("<div class=grid>")
            for it in sorted(group, key=lambda x: x.path):
                if shown >= limit:
                    break
                tags = []
                if it.split:
                    tags.append(it.split)
                if it.boxes is not None:
                    tags.append(f"{it.boxes}box" if it.boxes else "nolabel")
                if it.luma is not None:
                    tags.append(f"L{it.luma:.0f}")
                cap = f"<b>{it.path.rsplit('/', 1)[-1]}</b><br>{it.w}&times;{it.h} &middot; {human(it.nbytes)}"
                if tags:
                    cap += "<br>" + " &middot; ".join(tags)
                try:
                    uri = thumb_uri(root / it.path)
                    parts.append(f"<figure><img loading=lazy src='{uri}'><figcaption>{cap}</figcaption></figure>")
                    shown += 1
                except Exception as exc:  # noqa: BLE001
                    parts.append(f"<figure><figcaption>{it.path}<br>unreadable: {exc}</figcaption></figure>")
            parts.append("</div>")
        if len(b.items) > shown:
            parts.append(f"<p class=more>&hellip; {len(b.items) - shown} more not shown (raise --limit)</p>")
        parts.append("</section>")

    out.write_text("".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="repo root (default: auto-detect from CWD)")
    ap.add_argument("--html", type=Path, default=None, metavar="PATH", help="also write a contact-sheet HTML")
    ap.add_argument("--json", action="store_true", help="emit JSON to stdout instead of the text report")
    ap.add_argument("--limit", type=int, default=200, help="max thumbnails per category in --html (default 200)")
    ap.add_argument("--no-luma", action="store_true", help="skip mean-luminance (faster)")
    args = ap.parse_args(argv)

    root = find_root(args.root)
    if not root.exists():
        ap.error(f"root {root} does not exist")

    items = collect(root, want_luma=not args.no_luma)
    buckets = bucketise(items)

    if args.json:
        print(to_json(root, buckets))
    else:
        print_report(root, buckets)

    if args.html:
        write_html(root, buckets, args.html, args.limit)
        if not args.json:
            print(f"contact sheet -> {args.html}  ({human(args.html.stat().st_size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
