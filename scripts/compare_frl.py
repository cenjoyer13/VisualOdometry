#!/usr/bin/env python3
"""Compare RosbagEvaluator result CSV(s) against the FRL ground truth WITHOUT
any spatial alignment (the pipeline's auto-aligner already handles the yaw).

The FRL (.pos, 5 Hz) is cut to the config's [start_time, end_time] window and
converted to a local ENU frame the same way main_rosbag does. Because the FRL
is only 5 Hz, it sets the comparison rate: the higher-rate VO trajectory is
sampled at each FRL point's relative position in the window, so we compare at
5 Hz. No Umeyama / rotation / scale fitting is applied.

Usage:
  scripts/compare_frl.py <result.csv> [more.csv ...]
  scripts/compare_frl.py build/queue/results          # whole directory

Config is derived from each result name:
  mun3_sift_flannmatch__gyro.csv -> configs/mun/mun3_sift_flannmatch.yaml
(override the search dir with --configs <dir>).
"""
import sys, os, re, csv, glob, math, argparse
from datetime import datetime

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
    """Returns list of (rel_t_seconds, lat, lon, alt)."""
    rows = []
    t0 = None
    for line in open(path):
        if not line.strip() or line.startswith('%'):
            continue
        p = line.split()
        if len(p) < 5:
            continue
        try:
            t = datetime.strptime(p[0] + ' ' + p[1], '%Y/%m/%d %H:%M:%S.%f')
            lat, lon, alt = float(p[2]), float(p[3]), float(p[4])
        except ValueError:
            continue
        if t0 is None:
            t0 = t
        rows.append(((t - t0).total_seconds(), lat, lon, alt))
    return rows


def frl_window_enu(frl, start, end):
    """Trim to [start, end] (relative seconds) and convert to ENU about the
    first in-window sample. Returns (times, positions[Nx3])."""
    hi = end if end >= 0 else frl[-1][0]
    win = [r for r in frl if start <= r[0] <= hi]
    if len(win) < 2:
        return [], []
    o_t, o_lat, o_lon, o_alt = win[0]
    times, pos = [], []
    for t, lat, lon, alt in win:
        e = EARTH_R * (lon - o_lon) * DEG * math.cos(o_lat * DEG)
        n = EARTH_R * (lat - o_lat) * DEG
        u = alt - o_alt
        times.append(t)
        pos.append((e, n, u))
    return times, pos


def load_vo(path):
    P = []
    for r in list(csv.reader(open(path)))[1:]:
        if not r:
            continue
        P.append((float(r[1]), float(r[2]), float(r[3])))
    return P


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='result CSV(s) or a directory')
    ap.add_argument('--configs', default=os.path.join(REPO, 'configs', 'mun'))
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
        start, end, ppk = parse_config(cfg)
        if not ppk:
            print(f"{name:40}  no ppk_path in config"); continue
        frl_path = os.path.normpath(os.path.join(REPO, 'build', ppk))
        if not os.path.exists(frl_path):
            print(f"{name:40}  FRL not found ({frl_path})"); continue
        if frl_path not in frl_cache:
            frl_cache[frl_path] = load_frl(frl_path)
        _, gt = frl_window_enu(frl_cache[frl_path], start, end)
        vo = load_vo(f)
        if len(gt) < 2 or len(vo) < 2:
            print(f"{name:40}  too few points (gt={len(gt)} vo={len(vo)})"); continue
        r = compare(resample(vo, len(gt)), gt)
        print(f"{name:40} {r['n']:5d} {r['ate3d']:8.2f} {r['ate2d']:8.2f} {r['atez']:7.2f} "
              f"{r['mean3d']:8.2f} {r['max3d']:8.2f} {r['final3d']:8.2f}")


if __name__ == '__main__':
    main()
