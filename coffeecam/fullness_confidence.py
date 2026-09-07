"""How confident is the fullness classifier over the captured frames?

    .venv/bin/python -m coffeecam.fullness_confidence [--low 0.6] [--limit 40]
                                                      [--date 2026-09-03 ...] [--csv out.csv]

Reads the state-history logs (``captures/pipeline/state-*.jsonl``, written by the
worker / ``coffeecam.backfill_history``) and summarises the classifier's own
top-1 probability (``p`` in each row):

  * overall distribution + a coarse histogram
  * per-level mean confidence and low-confidence share
  * detector misses (no box -> the classifier ran on the static DEFAULT_POT_BOX
    crop, a common source of garbage calls)
  * the lowest-confidence frames, by path, as bad-image candidates

Rows predate this field if written before ``p`` was added — re-run the backfill
(``--force``) to refresh them.
"""

from __future__ import annotations

import argparse
import csv as _csv
import sys
from pathlib import Path

from coffeecam import state_history
from coffeecam.summary import DEFAULT_CAPTURES_DIR

_BINS = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0001]


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _fmt(x: float) -> str:
    return "  nan" if x != x else f"{x:.3f}"


def _load(captures_dir: Path, dates: list[str] | None) -> list[dict]:
    dates = dates or state_history.available_dates(captures_dir)
    rows: list[dict] = []
    for d in dates:
        for r in state_history.load_rows(captures_dir, date=d):
            r["_date"] = d
            rows.append(r)
    return rows


def summarise(rows: list[dict], *, low: float, limit: int) -> str:
    out: list[str] = []
    n = len(rows)
    with_p = [r for r in rows if isinstance(r.get("p"), (int, float))]
    ps = [float(r["p"]) for r in with_p]
    out.append(f"rows                {n}")
    out.append(f"  with classifier p {len(with_p)}")
    out.append(f"  no p (Null/old)   {n - len(with_p)}")
    errs = [r for r in rows if r.get("errors")]
    if errs:
        out.append(f"  rows with errors  {len(errs)}")
    if not ps:
        out.append("\nno classifier probabilities in range — nothing to summarise")
        return "\n".join(out)

    out.append("\ntop-1 probability (classifier confidence)")
    out.append(f"  min {_fmt(min(ps))}  p10 {_fmt(_pct(ps, .1))}  p25 {_fmt(_pct(ps, .25))}  "
               f"median {_fmt(_pct(ps, .5))}  mean {_fmt(sum(ps) / len(ps))}  "
               f"p75 {_fmt(_pct(ps, .75))}  max {_fmt(max(ps))}")
    out.append("\n  histogram")
    for lo, hi in zip(_BINS, _BINS[1:]):
        c = sum(1 for p in ps if lo <= p < hi)
        bar = "#" * round(40 * c / len(ps))
        out.append(f"    {lo:.2f}-{min(hi, 1.0):.2f}  {c:5d}  {c / len(ps) * 100:5.1f}%  {bar}")
    below = sum(1 for p in ps if p < low)
    out.append(f"\n  below {low:.2f}: {below}  ({below / len(ps) * 100:.1f}%)")

    out.append("\nper level              n     mean p   <%.2f" % low)
    by_level: dict[str, list[float]] = {}
    for r in with_p:
        by_level.setdefault(r.get("level", "?"), []).append(float(r["p"]))
    for lvl, xs in sorted(by_level.items(), key=lambda kv: -len(kv[1])):
        lo_share = sum(1 for x in xs if x < low) / len(xs) * 100
        out.append(f"  {lvl:<18} {len(xs):5d}   {sum(xs) / len(xs):.3f}    {lo_share:5.1f}%")

    det_miss = [r for r in rows if r.get("conf") is None]
    det_conf = [float(r["conf"]) for r in rows if isinstance(r.get("conf"), (int, float))]
    out.append(f"\ndetector: {len(det_miss)} misses (no box -> static crop), "
               f"{len(det_conf)} hits")
    if det_conf:
        out.append(f"  box conf  min {_fmt(min(det_conf))}  median {_fmt(_pct(det_conf, .5))}  "
                   f"max {_fmt(max(det_conf))}")
    if det_miss:
        miss_p = [float(r["p"]) for r in det_miss if isinstance(r.get("p"), (int, float))]
        if miss_p:
            out.append(f"  classifier p on misses: mean {_fmt(sum(miss_p) / len(miss_p))} "
                       f"(vs {_fmt(sum(ps) / len(ps))} overall)")

    ranked = sorted(with_p, key=lambda r: float(r["p"]))[:limit]
    out.append(f"\nlowest-confidence frames (<= {limit})")
    out.append("   p      conf   level   when              frame")
    for r in ranked:
        conf = r.get("conf")
        out.append(
            f"  {float(r['p']):.3f}  {('  -  ' if conf is None else f'{conf:.3f}')}  "
            f"{r.get('level', '?'):<6}  {r['_date']} {r.get('ts', '')[11:19]}  "
            f"{r.get('frame', '(no path; pre-backfill row)')}"
        )
    return "\n".join(out)


def _write_csv(rows: list[dict], path: Path) -> None:
    cols = ["date", "ts", "level", "p", "conf", "score", "frame", "artifact", "errors"]
    with path.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: (r["_date"], r.get("ts", ""))):
            w.writerow([
                r["_date"], r.get("ts", ""), r.get("level", ""), r.get("p", ""),
                r.get("conf", ""), r.get("score", ""), r.get("frame", ""),
                r.get("artifact", ""), " | ".join(r.get("errors", [])),
            ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures-dir", type=Path, default=DEFAULT_CAPTURES_DIR)
    ap.add_argument("--date", action="append", help="restrict to this day (repeatable)")
    ap.add_argument("--low", type=float, default=0.6, help="low-confidence threshold")
    ap.add_argument("--limit", type=int, default=40, help="how many worst frames to list")
    ap.add_argument("--csv", type=Path, help="also dump every row to this CSV")
    args = ap.parse_args(argv)

    rows = _load(args.captures_dir, args.date)
    if not rows:
        print("no state-history rows found", file=sys.stderr)
        return 1
    print(summarise(rows, low=args.low, limit=args.limit))
    if args.csv:
        _write_csv(rows, args.csv)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
