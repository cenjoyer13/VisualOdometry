#!/usr/bin/env python3
"""For each detector/matcher combo, plot the 4 variant VO trajectories (raw,
bucket, both, gyro) against the FRL ground truth on one XY graph with the ATEs
in the corner, and write a per-sequence summary CSV of all ATEs.

No spatial alignment (same convention as compare_frl.py): the FRL is cut to the
config window, converted to ENU, and the VO is compared at the FRL's 5 Hz rate.

Outputs (under build/queue/results/):
  mun3/<combo>.png , mun3/summary_ate.csv
  mun4/<combo>.png , mun4/summary_ate.csv

Usage: scripts/plot_frl.py [results_dir]   (default build/queue/results)
"""
import sys, os, csv, glob, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_frl as cf

REPO = cf.REPO
VARIANTS = ['raw', 'bucket', 'both', 'gyro']
COLORS = {'raw': '#888888', 'bucket': 'tab:blue', 'both': 'tab:green', 'gyro': 'tab:red'}


def avg_fps(log_path):
    """Mean per-frame FPS from the run log (excludes the 0.0 bootstrap frame)."""
    if not os.path.exists(log_path):
        return ''
    vals = [float(x) for x in re.findall(r'FPS:([0-9.]+)', open(log_path, errors='ignore').read())]
    vals = [v for v in vals if v > 0]
    return round(sum(vals) / len(vals), 2) if vals else ''


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, 'build', 'queue', 'results')
    combos = sorted(set(os.path.basename(f).split('__')[0]
                        for f in glob.glob(os.path.join(results_dir, '*.csv'))))
    logs_dir = os.path.join(os.path.dirname(results_dir), 'logs')
    frl_cache = {}
    summaries = {}  # seq -> list of rows

    for combo in combos:
        seq = combo.split('_')[0]              # mun3 / mun4
        cfg = os.path.join(REPO, 'configs', 'mun', combo + '.yaml')
        if not os.path.exists(cfg):
            print(f"skip {combo}: no config"); continue
        start, end, ppk = cf.parse_config(cfg)
        frl_path = os.path.normpath(os.path.join(REPO, 'build', ppk)) if ppk else None
        if not frl_path or not os.path.exists(frl_path):
            print(f"skip {combo}: FRL not found"); continue
        if frl_path not in frl_cache:
            frl_cache[frl_path] = cf.load_frl(frl_path)
        _, gt = cf.frl_window_enu(frl_cache[frl_path], start, end)
        if len(gt) < 2:
            print(f"skip {combo}: empty GT window"); continue

        out_dir = os.path.join(results_dir, seq)
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot([p[0] for p in gt], [p[1] for p in gt], 'k-', lw=2.2, label='GT', zorder=1)
        ax.plot(gt[0][0], gt[0][1], 'ko', ms=7, zorder=5)   # start

        ate_lines = ['ATE 3D RMSE [m]']
        for v in VARIANTS:
            f = os.path.join(results_dir, f"{combo}__{v}.csv")
            if not os.path.exists(f):
                continue
            vo = cf.load_vo(f)
            if len(vo) < 2:
                continue
            ax.plot([p[0] for p in vo], [p[1] for p in vo],
                    color=COLORS[v], lw=1.1, alpha=0.9, label=v, zorder=2)
            r = cf.compare(cf.resample(vo, len(gt)), gt)
            fps = avg_fps(os.path.join(logs_dir, f"{combo}__{v}.log"))
            ate_lines.append(f"{v:6} {r['ate3d']:7.1f}  {fps:>5} fps" if fps != '' else f"{v:6} {r['ate3d']:7.1f}")
            summaries.setdefault(seq, []).append({
                'combo': combo, 'variant': v, 'n': r['n'],
                'ate3d': round(r['ate3d'], 3), 'ate2d': round(r['ate2d'], 3),
                'atez': round(r['atez'], 3), 'mean3d': round(r['mean3d'], 3),
                'max3d': round(r['max3d'], 3), 'final3d': round(r['final3d'], 3),
                'avg_fps': fps,
            })

        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
        ax.set_title(combo)
        ax.grid(True, ls=':', alpha=0.5)
        ax.legend(loc='lower right', fontsize=9)
        ax.text(0.02, 0.98, '\n'.join(ate_lines), transform=ax.transAxes,
                va='top', ha='left', family='monospace', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.85))
        fig.tight_layout()
        png = os.path.join(out_dir, f"{combo}.png")
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"wrote {png}")

    # Per-sequence summary CSVs.
    cols = ['combo', 'variant', 'n', 'ate3d', 'ate2d', 'atez', 'mean3d', 'max3d', 'final3d', 'avg_fps']
    for seq, rows in summaries.items():
        path = os.path.join(results_dir, seq, 'summary_ate.csv')
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}  ({len(rows)} rows)")


if __name__ == '__main__':
    main()
