#!/usr/bin/env python3
"""Compare RosbagEvaluator result CSV(s) against the FRL ground truth WITHOUT
any spatial alignment (the pipeline's auto-aligner or heading seed already
handles the yaw).

Pairing is by TIMESTAMP. Both sides are put on the same absolute clock -- the
FRL's GPS time converted to UTC by subtracting 18 leap seconds, exactly as
main_rosbag's loadPpkFile does -- and the VO is linearly interpolated to each
FRL sample time. The FRL is 5 Hz, so that sets the comparison rate.

This replaces a parametric resample by INDEX, which assumed the VO track spanned
the whole config window at a uniform rate. It does not: the trajectory only
begins once the estimator initialises (about 7 s into this bag), so index-based
pairing compared VO against GT shifted by that much and quietly reported a
different number than the timestamp-matched evo path on the same data.

Both tracks are re-origined at the first matched sample, so this is the same
start-anchored convention as scripts/evaluate_mun3_frl.sh. No Umeyama /
rotation / scale fitting is applied.

Usage:
  scripts/compare_frl.py <result.csv> [more.csv ...]
  scripts/compare_frl.py build/queue/results          # whole directory

Config is derived from each result name:
  mun3_sift_flannmatch__gyro.csv -> configs/mun/mun3_sift_flannmatch.yaml
(override the search dir with --configs <dir>).
"""
import sys, os, re, csv, glob, math, argparse
from datetime import datetime, timezone, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EARTH_R = 6378137.0
DEG = math.pi / 180.0


def parse_config(path):
    """Pull rosbag.start_time / end_time / ppk_path from an OpenCV-YAML config
    by hand (yaml.safe_load chokes on %YAML:1.0 and !!opencv-matrix)."""
    start, end, ppk = 0.0, -1.0, None
    in_rosbag = False
    for line in open(path):
        if line[:1] not in (' ', '\t', '\n', '#'):       # top-level key
            in_rosbag = line.strip().startswith('rosbag:')
        if not in_rosbag:
            continue
        if (m := re.search(r'start_time:\s*([-\d.]+)', line)): start = float(m.group(1))
        if (m := re.search(r'end_time:\s*([-\d.]+)', line)):   end = float(m.group(1))
        if (m := re.search(r'ppk_path:\s*(\S+)', line)):       ppk = m.group(1)
    return start, end, ppk


def load_frl(path):
    """Returns list of (epoch_seconds_utc, lat, lon, alt).

    GPS time minus 18 leap seconds (valid 2017+), matching main_rosbag's
    loadPpkFile and prepare_evo.py, so FRL and VO share one absolute clock."""
    rows = []
    for line in open(path):
        if not line.strip() or line.startswith('%'):
            continue
        p = line.split()
        if len(p) < 5:
            continue
        try:
            t = datetime.strptime(p[0] + ' ' + p[1], '%Y/%m/%d %H:%M:%S.%f')
            t = t.replace(tzinfo=timezone.utc) - timedelta(seconds=18)
            lat, lon, alt = float(p[2]), float(p[3]), float(p[4])
        except ValueError:
            continue
        rows.append((t.timestamp(), lat, lon, alt))
    rows.sort()
    return rows


def frl_window_enu(frl, t_lo, t_hi):
    """Trim to the absolute-time span [t_lo, t_hi] and convert to ENU about the
    first in-window sample. Returns (times, positions[Nx3])."""
    win = [r for r in frl if t_lo <= r[0] <= t_hi]
    if len(win) < 2:
        return [], []
    _, o_lat, o_lon, o_alt = win[0]
    times, pos = [], []
    for t, lat, lon, alt in win:
        e = EARTH_R * (lon - o_lon) * DEG * math.cos(o_lat * DEG)
        n = EARTH_R * (lat - o_lat) * DEG
        u = alt - o_alt
        times.append(t)
        pos.append((e, n, u))
    return times, pos


def load_vo(path):
    """Returns (times[N], positions[Nx3]). Column 0 must be an absolute
    timestamp; a frame counter (the pre-2026 CSV format) is rejected loudly
    rather than silently mis-paired."""
    T, P = [], []
    for r in list(csv.reader(open(path)))[1:]:
        if not r or len(r) < 4:
            continue
        T.append(float(r[0]))
        P.append((float(r[1]), float(r[2]), float(r[3])))
    if T and T[0] < 1e8:
        raise ValueError(f"{path}: column 0 looks like a frame index, not a "
                         f"timestamp (first value {T[0]}). Re-run: the CSV "
                         f"schema is Timestamp,Pred_X,...")
    return T, P


def interp_vo(vo_t, vo_p, times):
    """Linear interpolation of the VO track at each requested time. Times
    outside the VO span are dropped, and the surviving indices are returned so
    the GT can be trimmed to match."""
    out, keep = [], []
    n = len(vo_t)
    j = 0
    for i, t in enumerate(times):
        if t < vo_t[0] or t > vo_t[-1]:
            continue
        while j + 1 < n - 1 and vo_t[j + 1] < t:
            j += 1
        t0, t1 = vo_t[j], vo_t[min(j + 1, n - 1)]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        p0, p1 = vo_p[j], vo_p[min(j + 1, n - 1)]
        out.append(tuple(p0[k] * (1 - a) + p1[k] * a for k in range(3)))
        keep.append(i)
    return out, keep


def reorigin(P):
    """Shift a track so it starts at the origin -- the start-anchored
    convention both scorers use."""
    if not P:
        return P
    o = P[0]
    return [(p[0] - o[0], p[1] - o[1], p[2] - o[2]) for p in P]


def resample(P, m):
    """Parametric linear resample of P (Nx3) to m points over the same span."""
    n = len(P)
    out = []
    for j in range(m):
        x = (j / (m - 1)) * (n - 1) if m > 1 else 0.0
        i0 = int(math.floor(x)); i1 = min(i0 + 1, n - 1); a = x - i0
        out.append(tuple(P[i0][k] * (1 - a) + P[i1][k] * a for k in range(3)))
    return out


def rmse(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else float('nan')


def compare(vo, gt):
    """No-alignment errors between resampled VO and the GT (both Nx3, ENU)."""
    e3 = [math.dist(vo[i], gt[i]) for i in range(len(gt))]
    e2 = [math.hypot(vo[i][0] - gt[i][0], vo[i][1] - gt[i][1]) for i in range(len(gt))]
    eu = [abs(vo[i][2] - gt[i][2]) for i in range(len(gt))]
    return {
        'n': len(gt),
        'ate3d': rmse(e3), 'ate2d': rmse(e2), 'atez': rmse(eu),
        'mean3d': sum(e3) / len(e3), 'max3d': max(e3), 'final3d': e3[-1],
    }


def plot(name, gt, vo, metrics, out_png):
    # Run this script with ~/venv/bin/python: matplotlib lives there, not in
    # the system interpreter (and not in the evo venv either).
    try:
        import matplotlib
    except ImportError:
        raise SystemExit("matplotlib not found -- run with ~/venv/bin/python")
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([p[0] for p in gt], [p[1] for p in gt], '--', color='#555555',
            lw=1.4, label='FRL ground truth')
    ax.plot([p[0] for p in vo], [p[1] for p in vo], '-', color='tab:red',
            lw=1.4, label='estimate')
    ax.plot([gt[0][0]], [gt[0][1]], 'o', color='black', ms=6, label='start')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    ax.set_title(f"{name}  --  timestamp-matched, no alignment")
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    txt = (f"ATE3D {metrics['ate3d']:.1f} m\n"
           f"ATE2D {metrics['ate2d']:.1f} m\n"
           f"ATEz  {metrics['atez']:.1f} m\n"
           f"mean  {metrics['mean3d']:.1f} m\n"
           f"max   {metrics['max3d']:.1f} m\n"
           f"final {metrics['final3d']:.1f} m\n"
           f"n     {metrics['n']}")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, va='bottom', ha='left',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', fc='white', ec='#bbbbbb', alpha=0.9))
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='run folder(s), a parent of them, or result CSV(s)')
    ap.add_argument('--configs', default=os.path.join(REPO, 'configs', 'mun'))
    ap.add_argument('--plot', action='store_true',
                    help='also write <run>/eval/trajectory.png')
    args = ap.parse_args()

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            # results/ is one folder per run (see results/README.md), so accept
            # either a run folder or the parent of several.
            here = sorted(glob.glob(os.path.join(p, '*.csv')))
            nested = sorted(glob.glob(os.path.join(p, '*', 'trajectory.csv')))
            files += here + nested
        else:
            files.append(p)

    frl_cache = {}
    print(f"{'result':40} {'n':>5} {'ATE3D':>8} {'ATE2D':>8} {'ATEz':>7} {'mean':>8} {'max':>8} {'final':>8}")
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        # A run folder holds its result as <run>/trajectory.csv, so the run name
        # -- and hence the config -- comes from the folder, not the file.
        if name == 'trajectory':
            name = os.path.basename(os.path.dirname(os.path.abspath(f)))
        combo = name.split('__')[0]
        cfg = os.path.join(args.configs, combo + '.yaml')
        if not os.path.exists(cfg):
            print(f"{name:40}  no config ({cfg})"); continue
        _, _, ppk = parse_config(cfg)
        if not ppk:
            print(f"{name:40}  no ppk_path in config"); continue
        frl_path = os.path.normpath(os.path.join(REPO, 'build', ppk))
        if not os.path.exists(frl_path):
            print(f"{name:40}  FRL not found ({frl_path})"); continue
        if frl_path not in frl_cache:
            frl_cache[frl_path] = load_frl(frl_path)

        try:
            vo_t, vo_p = load_vo(f)
        except ValueError as e:
            print(f"{name:40}  {e}"); continue
        if len(vo_t) < 2:
            print(f"{name:40}  too few VO points ({len(vo_t)})"); continue

        # The comparison window is the VO's own span: the estimator only starts
        # producing poses once it initialises, and scoring it against GT from
        # before that would be comparing against nothing.
        times, gt = frl_window_enu(frl_cache[frl_path], vo_t[0], vo_t[-1])
        if len(gt) < 2:
            print(f"{name:40}  no FRL samples in the VO span"); continue

        vo, keep = interp_vo(vo_t, vo_p, times)
        gt = [gt[i] for i in keep]
        if len(gt) < 2:
            print(f"{name:40}  no overlap after interpolation"); continue

        gt, vo = reorigin(gt), reorigin(vo)
        r = compare(vo, gt)
        print(f"{name:40} {r['n']:5d} {r['ate3d']:8.2f} {r['ate2d']:8.2f} "
              f"{r['atez']:7.2f} {r['mean3d']:8.2f} {r['max3d']:8.2f} {r['final3d']:8.2f}")

        if args.plot:
            d = os.path.join(os.path.dirname(os.path.abspath(f)), 'eval')
            os.makedirs(d, exist_ok=True)
            png = os.path.join(d, 'trajectory.png')
            plot(name, gt, vo, r, png)
            print(f"{'':40}  -> {png}")


if __name__ == '__main__':
    main()
