#!/usr/bin/env python3
"""Score + plot a single OdomLogEvaluator result CSV against its inline GT.

Unlike the mun/rosbag configs (FRL ground truth lives in a separate .pos file,
scored by compare_frl.py / plot_frl.py), OdomLogEvaluator bakes GT straight
into the result CSV as GT_X/Y/Z (ENU, origin at the first processed frame,
already interpolated to each frame's timestamp -- see main_odomlog.cpp). So
there's no separate GT track to load or resample: VO and GT are already
paired row-for-row.

For one result CSV this:
  - computes the same no-alignment ATE family as compare_frl.compare()
    (ate3d/ate2d/atez, mean3d, max3d, final3d),
  - plots VO vs GT (top-down East-North) with the metrics in a corner box,
  - appends one row to a summary CSV (creating it with a header if absent).

Meant to be called once per run, right after that run's OdomLogEvaluator
process exits -- so the summary and graph are up to date after every run
rather than only once at the end of a whole queue.

Usage:
  scripts/eval_odomlog.py <result.csv> <out.png> <summary.csv> [--log <run.log>]
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_run(path):
    required = {'Frame', 'Pred_X', 'Pred_Y', 'Pred_Z', 'GT_X', 'GT_Y', 'GT_Z'}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            sys.exit(f"{path}: missing columns {sorted(missing)}")
        rows = [r for r in reader if all(r.get(c) not in (None, '') for c in required)]
    if not rows:
        sys.exit(f"{path}: no data rows")
    vo = np.array([[float(r['Pred_X']), float(r['Pred_Y']), float(r['Pred_Z'])] for r in rows])
    gt = np.array([[float(r['GT_X']), float(r['GT_Y']), float(r['GT_Z'])] for r in rows])
    return vo, gt


def compare(vo, gt):
    e3 = np.linalg.norm(vo - gt, axis=1)
    e2 = np.linalg.norm(vo[:, :2] - gt[:, :2], axis=1)
    eu = np.abs(vo[:, 2] - gt[:, 2])
    return {
        'n': len(gt),
        'ate3d': float(np.sqrt(np.mean(e3 ** 2))),
        'ate2d': float(np.sqrt(np.mean(e2 ** 2))),
        'atez': float(np.sqrt(np.mean(eu ** 2))),
        'mean3d': float(np.mean(e3)),
        'max3d': float(np.max(e3)),
        'final3d': float(e3[-1]),
    }


def avg_fps(log_path):
    if not log_path or not os.path.exists(log_path):
        return ''
    vals = [float(x) for x in re.findall(r'FPS:\s*([0-9.]+)', open(log_path, errors='ignore').read())]
    vals = [v for v in vals if v > 0]
    return round(sum(vals) / len(vals), 2) if vals else ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result_csv')
    ap.add_argument('out_png')
    ap.add_argument('summary_csv')
    ap.add_argument('--log', default=None, help='run log, for the avg FPS column')
    ap.add_argument('--name', default=None, help='row label (default: result_csv basename)')
    args = ap.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.result_csv))[0]
    vo, gt = load_run(args.result_csv)
    r = compare(vo, gt)
    fps = avg_fps(args.log)
    path_len = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())

    # ----- plot -----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt[:, 0], gt[:, 1], 'k-', lw=2.2, label='GT', zorder=1)
    ax.plot(gt[0, 0], gt[0, 1], 'ko', ms=7, zorder=5, label='start')
    ax.plot(vo[:, 0], vo[:, 1], color='tab:blue', lw=1.1, alpha=0.9, label='VO', zorder=2)

    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    ax.set_title(name)
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)

    box = [
        f"ATE3D {r['ate3d']:.1f} m   drift {100 * r['ate3d'] / path_len:.1f}%" if path_len > 0 else f"ATE3D {r['ate3d']:.1f} m",
        f"MAE   {r['mean3d']:.1f} m   FDE {r['final3d']:.1f} m",
        f"path  {path_len:.1f} m   n={r['n']}",
    ]
    if fps != '':
        box.append(f"FPS   {fps}")
    ax.text(0.02, 0.98, '\n'.join(box), transform=ax.transAxes, va='top',
            ha='left', family='monospace', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.85))

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_png) or '.', exist_ok=True)
    fig.savefig(args.out_png, dpi=130)
    plt.close(fig)
    print(f"wrote {args.out_png}")

    # ----- append summary row -----
    cols = ['name', 'n', 'ate3d', 'ate2d', 'atez', 'mean3d', 'max3d', 'final3d',
            'path_len', 'drift_pct', 'avg_fps']
    row = {
        'name': name, 'n': r['n'],
        'ate3d': round(r['ate3d'], 3), 'ate2d': round(r['ate2d'], 3), 'atez': round(r['atez'], 3),
        'mean3d': round(r['mean3d'], 3), 'max3d': round(r['max3d'], 3), 'final3d': round(r['final3d'], 3),
        'path_len': round(path_len, 3),
        'drift_pct': round(100 * r['ate3d'] / path_len, 3) if path_len > 0 else '',
        'avg_fps': fps,
    }
    os.makedirs(os.path.dirname(args.summary_csv) or '.', exist_ok=True)
    write_header = not os.path.exists(args.summary_csv)
    existing = []
    if not write_header:
        with open(args.summary_csv) as f:
            existing = list(csv.DictReader(f))
        existing = [r2 for r2 in existing if r2['name'] != name]
    with open(args.summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(existing)
        w.writerow(row)
    print(f"appended {name} to {args.summary_csv}")


if __name__ == '__main__':
    main()
