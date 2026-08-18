#!/usr/bin/env python3
"""Read a run.jsonl produced by odometry/utils/RunLog.

The whole point is compression. A 4500-frame replay writes ~35k records; the
default `summary` turns that into about a screenful, and runs the health checks
that encode the bugs this pipeline has actually hit. Raw records are available
via `trace` / `grep`, but you should rarely need them.

  logreader.py summary run.jsonl
  logreader.py health  run.jsonl          # just the checks, exit 1 if any fail
  logreader.py trace   run.jsonl --frame 2884
  logreader.py grep    run.jsonl --type vins
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(path):
    """Returns (meta, records_by_type). Tolerates a truncated final line, which
    is what a SIGKILLed run leaves behind."""
    meta, by_type = {}, defaultdict(list)
    bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            t = r.get("type", "?")
            if t in ("run.meta", "run.vins", "perf.summary"):
                meta.update(r)
            else:
                by_type[t].append(r)
    if bad:
        print(f"note: skipped {bad} unparseable line(s) (truncated run?)\n", file=sys.stderr)
    return meta, by_type


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[i]


def corr(a, b):
    """Pearson correlation; nan when either series is constant."""
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


def path_len(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


# --------------------------------------------------------------------------
# health checks -- each one encodes a bug this pipeline actually hit
# --------------------------------------------------------------------------

def health(meta, by):
    """Returns [(status, name, detail)] with status in OK / WARN / FAIL."""
    out = []
    feed_ok = [r for r in by.get("feed", []) if r.get("accepted")]
    states = by.get("state", [])
    inited = [r for r in states if r.get("solver_flag") == "NON_LINEAR"]
    gts = by.get("gt", [])

    # -- initialisation --------------------------------------------------
    vins = by.get("vins", [])
    fails = Counter()
    for r in vins:
        m = r.get("msg", "")
        for key in ("IMU excitation not enouth",
                    "Not enough features or parallax",
                    "misalign visual structure",
                    "failure detection"):
            if key in m:
                fails[key] += 1
    if not inited:
        out.append(("FAIL", "initialisation",
                    "never reached NON_LINEAR" +
                    (f"; reasons: {dict(fails)}" if fails else "")))
    else:
        f0 = inited[0].get("frame")
        out.append(("OK", "initialisation", f"first solved frame {f0}" +
                    (f"; {sum(fails.values())} pre-init rejections {dict(fails)}" if fails else "")))
    if fails.get("failure detection"):
        out.append(("WARN", "failure detection",
                    f"VINS reset {fails['failure detection']}x mid-run"))

    # -- feed rate vs configured freq (the 'Not enough parallax' bug) -----
    dts = [r["dt"] for r in feed_ok if r.get("dt", 0) > 0]
    if dts:
        hz = 1.0 / (sum(dts) / len(dts))
        out.append(("OK", "feed rate", f"{hz:.1f} Hz to the backend "
                                       f"({len(feed_ok)} accepted, "
                                       f"{len(by.get('feed', [])) - len(feed_ok)} throttled)"))

    # -- feature count vs VINS's design point (the 10x bug) --------------
    nf = [r["n_feat"] for r in feed_ok if "n_feat" in r]
    if nf:
        med = pct(nf, 50)
        status = "WARN" if med > 400 else "OK"
        out.append((status, "features/frame",
                    f"median {med:.0f}"
                    + ("  -- VINS is tuned for ~150; 10x that was a real bug" if status == "WARN" else "")))

    # -- track carry-over: low means tracks die and init will starve ------
    carry = [r["carry"] for r in feed_ok if "carry" in r]
    if carry:
        med = pct(carry, 50)
        status = "FAIL" if med < 0.4 else ("WARN" if med < 0.7 else "OK")
        out.append((status, "track carry-over", f"median {med * 100:.0f}% of ids survive a frame"))

    # -- per-axis estimate vs GT correlation (the trajectory_flip bug) ----
    if inited and gts:
        gt_by_frame = {r["frame"]: r["p"] for r in gts if "p" in r}
        est, ref = [], []
        for r in inited:
            g = gt_by_frame.get(r.get("frame"))
            if g and "p" in r:
                est.append(r["p"])
                ref.append(g)
        if len(est) > 10:
            axes = "XYZ"
            bad = []
            detail = []
            for k in range(3):
                c = corr([p[k] for p in est], [p[k] for p in ref])
                detail.append(f"{axes[k]}={c:+.2f}")
                if not math.isnan(c) and c < -0.3:
                    bad.append(axes[k])
            # X/Y are only comparable up to the unobservable yaw, so a poor
            # correlation there is expected; Z is gravity-aligned in both frames
            # and a NEGATIVE Z correlation means the altitude is inverted.
            zc = corr([p[2] for p in est], [p[2] for p in ref])
            status = "FAIL" if (not math.isnan(zc) and zc < -0.3) else "OK"
            msg = " ".join(detail)
            if status == "FAIL":
                msg += "  -- Z inverted: frame convention bug (cf. trajectory_flip)"
            else:
                msg += "  (X/Y free up to the unobservable yaw; Z must be positive)"
            out.append((status, "estimate vs GT", msg))

        # -- scale --------------------------------------------------------
        if len(est) > 10:
            le, lg = path_len(est), path_len(ref)
            if lg > 0:
                ratio = le / lg
                status = "WARN" if not (0.8 <= ratio <= 1.25) else "OK"
                out.append((status, "scale",
                            f"estimated path {le:.0f} m vs GT {lg:.0f} m ({ratio * 100:.0f}%)"))

    # -- state sanity -----------------------------------------------------
    if inited:
        vmax = max(r.get("v_norm", 0) for r in inited)
        bamax = max(r.get("ba_norm", 0) for r in inited)
        out.append((("WARN" if vmax > 100 else "OK"), "velocity", f"max |v| {vmax:.1f} m/s"))
        out.append((("WARN" if bamax > 2.0 else "OK"), "accel bias", f"max |ba| {bamax:.2f} m/s^2"))

    # -- perf: what dominates, and whether anything is unmeasured ---------
    if "total_mean" in meta and meta["total_mean"] > 0:
        tot = meta["total_mean"]
        stages = ["bag_read", "decode", "undistort", "gt", "frontend",
                  "reject", "backend", "imu", "log", "gui"]
        acc = sum(meta.get(f"{st}_mean", 0.0) for st in stages)
        top = max(stages, key=lambda st: meta.get(f"{st}_mean", 0.0))
        out.append(("OK", "perf hotspot",
                    f"{top} is {100.0 * meta.get(top + '_mean', 0.0) / tot:.0f}% of "
                    f"{tot:.1f} ms/frame"))
        un = (tot - acc) / tot
        if un > 0.10:
            out.append(("WARN", "perf unattributed",
                        f"{un * 100:.0f}% of loop time is claimed by no stage"))

    # -- solver: saturation and determinism -------------------------------
    if meta:
        mst = meta.get("max_solver_time")
        if isinstance(mst, (int, float)) and 0 < mst < 1.0:
            out.append(("WARN", "determinism",
                        f"max_solver_time={mst}s is a WALL-CLOCK budget; results "
                        f"vary with machine load"))
    return out


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_summary(args):
    meta, by = load(args.log)
    W = 78
    print("=" * W)
    print("RUN")
    print("=" * W)
    for k in ("git", "built", "config", "config_hash", "vins_config",
              "vins_config_hash", "bag", "start_time", "end_time", "frontend",
              "max_corners", "min_distance", "bucketing",
              "max_solver_time", "max_num_iterations", "max_cnt", "td",
              "estimate_td", "multiple_thread", "feed_hz"):
        if k in meta:
            print(f"  {k:<18} {meta[k]}")

    print()
    print("=" * W)
    print("VOLUME")
    print("=" * W)
    for t in sorted(by):
        print(f"  {t:<18} {len(by[t])}")

    feed_ok = [r for r in by.get("feed", []) if r.get("accepted")]
    states = by.get("state", [])
    if states:
        print()
        print("=" * W)
        print("TIMING (ms)")
        print("=" * W)
        solve = [r["solve_ms"] for r in states if "solve_ms" in r]
        if solve:
            print(f"  backend solve      p50 {pct(solve,50):7.1f}   "
                  f"p90 {pct(solve,90):7.1f}   max {max(solve):7.1f}")

    # -- perf breakdown, ranked by share of the loop --------------------
    stages = ["bag_read", "decode", "undistort", "gt", "frontend",
              "reject", "backend", "imu", "log", "gui"]
    recs = by.get("perf", [])
    if "total_mean" in meta and recs:
        tot = meta["total_mean"]
        print()
        print("=" * W)
        print(f"PERF  ({meta.get('frames', len(recs))} frames, ms/frame)")
        print("=" * W)
        print(f"  {'stage':<11} {'mean':>7} {'p50':>7} {'p90':>7} {'max':>7}  share"
              f"   {'active':>7} {'on':>6}")
        rows = [(meta.get(f"{st}_mean", 0.0), st) for st in stages]
        rows.sort(reverse=True)
        acc = 0.0
        for mean, st in rows:
            if mean <= 0.0:
                continue
            acc += mean
            share = 100.0 * mean / tot if tot > 0 else 0.0
            # A stage that only runs on some frames (the backend is skipped by
            # the feed throttle; gui is off in headless runs) has a per-frame
            # mean far below its real cost. "active" is the mean over the frames
            # where it actually ran, and "on" is how often that was -- without
            # them a bimodal stage reads as uniformly cheap.
            # Threshold scales with the stage's own p90 rather than a fixed
            # epsilon: a throttled backend frame still costs ~0.08 ms on the
            # early-return path, which a 0.01 ms cutoff counts as "active" and
            # which then drags the active mean down (29.7 ms over 68% of frames
            # instead of the true 40.4 ms over 50%).
            thresh = max(0.05, 0.05 * meta.get(f"{st}_p90", 0.0))
            act = [r.get(st, 0.0) for r in recs if r.get(st, 0.0) > thresh]
            amean = sum(act) / len(act) if act else 0.0
            on = 100.0 * len(act) / len(recs) if recs else 0.0
            bimodal = on < 95.0
            print(f"  {st:<11} {mean:7.2f} {meta.get(st+'_p50',0):7.2f} "
                  f"{meta.get(st+'_p90',0):7.2f} {meta.get(st+'_max',0):7.2f} "
                  f"{share:5.1f}%   "
                  + (f"{amean:7.2f} {on:5.0f}%" if bimodal else f"{'':>7} {'':>6}"))
        print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  -----")
        print(f"  {'total':<11} {tot:7.2f} {meta.get('total_p50',0):7.2f} "
              f"{meta.get('total_p90',0):7.2f} {meta.get('total_max',0):7.2f}  100.0%")
        un = tot - acc
        if tot > 0 and un / tot > 0.10:
            print(f"  {'UNATTRIBUTED':<11} {un:7.2f} {'':>7} {'':>7} {'':>7}  "
                  f"{100.0*un/tot:5.1f}%   <- loop time no stage claims")
        wall = tot * float(meta.get("frames", len(recs))) / 1000.0
        print(f"\n  wall clock in the loop: {wall:.0f} s "
              f"({(meta.get('frames', len(recs)) / wall) if wall > 0 else 0:.1f} frames/s)")
        print("  'active'/'on' shown only for stages that skip frames.")

    vins = by.get("vins", [])
    if vins:
        print()
        print("=" * W)
        print("VINS MESSAGES (deduplicated)")
        print("=" * W)
        c = Counter(r.get("msg", "").split(":")[0][:58] for r in vins)
        for msg, n in c.most_common(12):
            first = next(r.get("frame") for r in vins if r.get("msg", "").startswith(msg[:20]))
            # No frame context yet == emitted during construction, before the
            # first image was processed.
            where = "startup" if first is None else f"frame {first}"
            print(f"  {n:>5}x  (first @ {where})  {msg}")

    print()
    print("=" * W)
    print("HEALTH")
    print("=" * W)
    bad = 0
    for status, name, detail in health(meta, by):
        mark = {"OK": "  ok ", "WARN": " WARN", "FAIL": " FAIL"}[status]
        if status != "OK":
            bad += 1
        print(f"{mark}  {name:<20} {detail}")
    print()
    print(f"{bad} issue(s) flagged." if bad else "All checks passed.")
    return 1 if any(s == "FAIL" for s, _, _ in health(meta, by)) else 0


def cmd_health(args):
    meta, by = load(args.log)
    rows = health(meta, by)
    for status, name, detail in rows:
        print(f"{status:<5} {name:<20} {detail}")
    return 1 if any(s == "FAIL" for s, _, _ in rows) else 0


def cmd_trace(args):
    with open(args.log) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("frame") == args.frame:
                print(json.dumps(r, sort_keys=True))
    return 0


def cmd_grep(args):
    n = 0
    with open(args.log) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.type and r.get("type") != args.type:
                continue
            if args.contains and args.contains not in line:
                continue
            print(json.dumps(r, sort_keys=True))
            n += 1
            if args.limit and n >= args.limit:
                break
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("summary"); p.add_argument("log"); p.set_defaults(fn=cmd_summary)
    p = sub.add_parser("health");  p.add_argument("log"); p.set_defaults(fn=cmd_health)
    p = sub.add_parser("trace");   p.add_argument("log"); p.add_argument("--frame", type=int, required=True); p.set_defaults(fn=cmd_trace)
    p = sub.add_parser("grep");    p.add_argument("log"); p.add_argument("--type"); p.add_argument("--contains"); p.add_argument("--limit", type=int, default=40); p.set_defaults(fn=cmd_grep)

    args = ap.parse_args()
    if not getattr(args, "fn", None):
        ap.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
