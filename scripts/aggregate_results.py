"""Aggregate the 14 AirSim playback logs into 28 plots + a summary table.

Re-uses the start-locked alignment logic from analyze_playback.py so the per-
config RMSE is computed identically to what you'd get from running the
analyzer by hand on a single log.

Usage::

    python scripts/aggregate_results.py              # default subdir
    python scripts/aggregate_results.py --subdir airsim_eval_difficult

Outputs (under results/<subdir>/):
  plots/<config>_2d.png
  plots/<config>_3d.png
  summary.csv     <- machine-readable: config,frames,rmse_3d,rmse_uniform,mean_fps
  summary.md      <- human-readable table sorted by RMSE @ 10 Hz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless

# Make analyze_playback importable as a sibling module.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_playback import (  # noqa: E402
    load_active_segment,
    start_lock,
    plot_2d,
    plot_3d,
    DEFAULT_HEADING_SAMPLES,
)

import matplotlib.pyplot as plt  # noqa: E402  (imported after backend lock)


PROJECT_ROOT = HERE.parent

# Module-level paths are bound at runtime by configure_paths(); we keep names
# here so the rest of the script can refer to them uniformly.
LOGS_DIR: Path
PLOTS_DIR: Path
SUMMARY_CSV: Path
SUMMARY_MD: Path


def configure_paths(subdir: str) -> None:
    global LOGS_DIR, PLOTS_DIR, SUMMARY_CSV, SUMMARY_MD
    root = PROJECT_ROOT / "results" / subdir
    LOGS_DIR    = root / "logs"
    PLOTS_DIR   = root / "plots"
    SUMMARY_CSV = root / "summary.csv"
    SUMMARY_MD  = root / "summary.md"

# Uniform resampling rate for the "fair" RMSE column. 10 Hz matches the
# density of the recorded flight_path.csv (10 Hz on capture), so the
# resampled trajectories share the GT's native time resolution rather than
# being shaped by whichever VO config happened to log fastest.
UNIFORM_RATE_HZ = 10.0


def rmse_uniform(df: pd.DataFrame) -> float:
    """RMSE after resampling both GT and VO to a 10 Hz time grid.

    Steps:
      1. Drop bootstrap rows (vo all-zero) — already applied by the caller.
      2. Build a uniform time grid t_uniform = arange(t_min, t_max, 1/10).
      3. Linearly interpolate gt_xyz and vo_xyz onto that grid (np.interp).
      4. Apply the SAME start-locked alignment we use for the per-frame RMSE
         (camera-frame -> ENU basis swap, translate to GT[0], yaw-align via
         the first-K-resampled-samples heading).
      5. RMSE = sqrt(mean(||gt - vo_aligned||^2)) on the resampled data.

    With ~94 s logs at 10 Hz, every config contributes ~940 samples, so the
    average is no longer dragged by how often a slow config managed to log
    relative to a fast one.
    """
    t = df["time_s"].to_numpy(dtype=float)
    gt = df[["gt_x", "gt_y", "gt_z"]].to_numpy(dtype=float)
    vo = df[["vo_x", "vo_y", "vo_z"]].to_numpy(dtype=float)

    step = 1.0 / UNIFORM_RATE_HZ
    t_grid = np.arange(t[0], t[-1], step)
    if t_grid.size < 10:
        return float("nan")

    gt_u = np.column_stack([np.interp(t_grid, t, gt[:, i]) for i in range(3)])
    vo_u = np.column_stack([np.interp(t_grid, t, vo[:, i]) for i in range(3)])

    vo_locked = start_lock(vo_u, gt_u, k=DEFAULT_HEADING_SAMPLES)
    err = vo_locked - gt_u
    return float(np.sqrt((err ** 2).sum(axis=1).mean()))


def evaluate_one(csv_path: Path) -> dict:
    """Run start-locked alignment + RMSE on a single log and emit two PNGs.

    Returns a dict of per-config metrics for the summary table.
    """
    df = load_active_segment(str(csv_path))
    if len(df) < 10:
        return {
            "config":   csv_path.stem,
            "frames":   len(df),
            "rmse_3d":  float("nan"),
            "mean_fps": float("nan"),
            "note":     "too few tracked frames",
        }

    gt = df[["gt_x", "gt_y", "gt_z"]].to_numpy(dtype=float)
    vo = df[["vo_x", "vo_y", "vo_z"]].to_numpy(dtype=float)

    vo_locked = start_lock(vo, gt, k=DEFAULT_HEADING_SAMPLES)
    err = vo_locked - gt
    rmse_3d = float(np.sqrt((err ** 2).sum(axis=1).mean()))

    # "Fair" RMSE: interpolate both trajectories to a uniform 10 Hz grid
    # before computing the residual, so fast configs don't get a denser
    # sampling vote than slow ones over the same physical flight.
    rmse_uniform_val = rmse_uniform(df)

    # PathPlayer logs the instantaneous fps per frame; the 0th frame can be 0
    # before the timer warms up, so filter zeros before averaging.
    fps_col = df["fps"].to_numpy(dtype=float) if "fps" in df.columns else np.array([])
    fps_col = fps_col[fps_col > 0]
    mean_fps = float(np.mean(fps_col)) if fps_col.size else float("nan")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    p2d = PLOTS_DIR / f"{csv_path.stem}_2d.png"
    p3d = PLOTS_DIR / f"{csv_path.stem}_3d.png"
    fig2 = plot_2d(gt, vo_locked, rmse_3d, save_to=str(p2d))
    fig3 = plot_3d(gt, vo_locked, rmse_3d, save_to=str(p3d))
    plt.close(fig2)
    plt.close(fig3)

    return {
        "config":       csv_path.stem,
        "frames":       len(df),
        "rmse_3d":      rmse_3d,
        "rmse_uniform": rmse_uniform_val,
        "mean_fps":     mean_fps,
        "note":         "",
    }


def write_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    # Sort by the fair (uniformly-resampled) RMSE so configs are compared on
    # equal time-grid footing rather than on whoever happened to log fastest.
    df = df.sort_values(["rmse_uniform", "config"],
                        na_position="last").reset_index(drop=True)
    df.to_csv(SUMMARY_CSV, index=False, float_format="%.4f")

    lines = []
    lines.append("# AirSim playback sweep — summary")
    lines.append("")
    lines.append(f"Logs: `{LOGS_DIR.relative_to(PROJECT_ROOT)}`  "
                 f"|  Plots: `{PLOTS_DIR.relative_to(PROJECT_ROOT)}`")
    lines.append("")
    lines.append(f"`RMSE raw` is the per-frame Euclidean error after start-locked "
                 f"alignment. `RMSE @ {UNIFORM_RATE_HZ:g} Hz` resamples GT and VO "
                 f"to a common {UNIFORM_RATE_HZ:g} Hz time grid (linear interp) "
                 f"before computing the error, so configs with very different "
                 f"frame rates contribute equally many samples to the average.")
    lines.append("")
    lines.append("Sorted by `RMSE @ 10 Hz` (lower is better).")
    lines.append("")
    lines.append(f"| # | Config | Frames | RMSE raw [m] | RMSE @ {UNIFORM_RATE_HZ:g} Hz [m] | Mean FPS | Notes |")
    lines.append("|---|--------|-------:|-------------:|---------------:|---------:|-------|")
    for i, row in enumerate(df.itertuples(index=False), 1):
        rmse_r = "—" if pd.isna(row.rmse_3d)      else f"{row.rmse_3d:.3f}"
        rmse_u = "—" if pd.isna(row.rmse_uniform) else f"{row.rmse_uniform:.3f}"
        fps    = "—" if pd.isna(row.mean_fps)     else f"{row.mean_fps:.1f}"
        note   = row.note if row.note else ""
        lines.append(f"| {i} | `{row.config}` | {row.frames} | {rmse_r} | {rmse_u} | {fps} | {note} |")
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--subdir", default="airsim_eval",
                   help="Folder name under results/ to read logs from and "
                        "write plots/summary into (default: airsim_eval).")
    args = p.parse_args()
    configure_paths(args.subdir)

    if not LOGS_DIR.is_dir():
        print(f"error: {LOGS_DIR} not found — did the sweep run?", file=sys.stderr)
        return 1

    csvs = sorted(LOGS_DIR.glob("*.csv"))
    if not csvs:
        print(f"error: no CSVs in {LOGS_DIR}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for i, csv in enumerate(csvs, 1):
        print(f"[{i:>2}/{len(csvs)}] {csv.stem} ...", end=" ", flush=True)
        try:
            row = evaluate_one(csv)
        except Exception as exc:  # noqa: BLE001
            row = {"config": csv.stem, "frames": 0, "rmse_3d": float("nan"),
                   "mean_fps": float("nan"), "note": f"error: {exc}"}
        rows.append(row)
        if row["note"]:
            print(f"SKIPPED ({row['note']})")
        else:
            print(f"RMSE raw={row['rmse_3d']:.3f} m  "
                  f"uniform={row['rmse_uniform']:.3f} m  "
                  f"FPS={row['mean_fps']:.1f}")

    write_summary(rows)
    print()
    print(f"summary CSV : {SUMMARY_CSV}")
    print(f"summary MD  : {SUMMARY_MD}")
    print(f"28 plots in : {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
