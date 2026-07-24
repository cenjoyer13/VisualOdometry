#!/usr/bin/env python3
"""Top-down (X-Z) VO trajectory plot for a bare, GT-less OdomLogEvaluator run.

Mirrors plot_kitti.py's style (same figure size, equal-aspect X-Z plane,
corner text box) but with no ground-truth overlay -- odom_log has none.
scale_estimator.type: Unit means axis values are arbitrary units, not
meters; the corner box says so explicitly.

Usage: scripts/plot_odomlog.py <trajectory.csv> <output.png>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_vo(path):
    import pandas as pd
    df = pd.read_csv(path)
    return df['Frame'].to_numpy(dtype=int), df[['Pred_X', 'Pred_Y', 'Pred_Z']].to_numpy(dtype=float)


def main():
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <trajectory.csv> <output.png>")
    csv_path, png_path = sys.argv[1], sys.argv[2]

    frames, pos = load_vo(csv_path)
    path_len = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(pos[:, 0], pos[:, 2], color='tab:blue', lw=1.3, zorder=2, label='VO')
    ax.plot(pos[0, 0], pos[0, 2], 'ko', ms=7, zorder=5, label='start')
    ax.plot(pos[-1, 0], pos[-1, 2], 'k^', ms=7, zorder=5, label='end')

    ax.set_aspect('equal', 'datalim')
    ax.set_xlabel('X [arb. units]')
    ax.set_ylabel('Z [arb. units]')
    ax.set_title('odom_log VO trajectory (no ground truth)')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)

    box = [
        f"frames  {frames[0]}-{frames[-1]} ({len(frames)})",
        f"path len  {path_len:8.1f} (arb. units)",
        "scale: Unit (no GT) -- shape only",
    ]
    ax.text(0.02, 0.98, '\n'.join(box), transform=ax.transAxes, va='top',
            ha='left', family='monospace', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.85))

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"wrote {png_path}")


if __name__ == '__main__':
    main()
