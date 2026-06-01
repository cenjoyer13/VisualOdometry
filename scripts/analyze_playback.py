#!/usr/bin/env python3
"""Lock the main_player VO trajectory to the GT start pose, plot, report RMSE.

The VO pipeline accumulates poses in the OpenCV camera frame: +X right,
+Y down, +Z forward, so the horizontal plane is X-Z and the altitude axis
is -Y. AirSim ground truth lives in an ENU world frame: +X east, +Y north,
+Z up. Direct distance between the two number triples is meaningless until
both live in the same basis.

Start-locked alignment, in order:

  1. Basis swap (fixed): map (X_cam, Y_cam, Z_cam) -> (X_cam, Z_cam, -Y_cam).
     After this VO is in an ENU-style local frame with X right, Y forward,
     Z up.
  2. Translate VO so VO[0] == GT[0].
  3. Yaw-rotate VO about the world Z axis by the angle that aligns its
     initial direction of travel with GT's initial direction of travel
     (averaged over the first K samples to suppress per-frame noise).

No global least-squares fit is applied, so any divergence further along the
trajectory is genuine drift, not absorbed by the alignment.

Input CSV header (produced by main_player.cpp):
    time_s, fps,
    gt_x, gt_y, gt_z, gt_pitch, gt_roll, gt_yaw,
    vo_x, vo_y, vo_z
"""

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


# Number of leading samples used to estimate the initial heading. Larger is
# more robust to per-frame noise; too large eats into the drift signal.
DEFAULT_HEADING_SAMPLES = 30


# Fixed camera-frame to ENU-style local-frame basis swap.
#   X_local = +X_cam    (right    -> east-equivalent before yaw)
#   Y_local = +Z_cam    (forward  -> north-equivalent before yaw)
#   Z_local = -Y_cam    (down     -> up)
CAM_TO_ENU = np.array([
    [1.0, 0.0,  0.0],
    [0.0, 0.0,  1.0],
    [0.0, -1.0, 0.0],
])


def load_active_segment(csv_path: str) -> pd.DataFrame:
    """Read the CSV and drop the leading frames before VO tracking activated.

    Tracking-inactive rows are written as vo_xyz = (0, 0, 0).
    """
    df = pd.read_csv(csv_path)
    required = {"gt_x", "gt_y", "gt_z", "vo_x", "vo_y", "vo_z"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"CSV is missing columns: {sorted(missing)}")

    inactive = (df[["vo_x", "vo_y", "vo_z"]].abs().sum(axis=1) <= 1e-9)
    if inactive.any():
        first_active = int((~inactive).idxmax())
        df = df.iloc[first_active:].reset_index(drop=True)
    return df


def initial_heading(points: np.ndarray, k: int) -> np.ndarray:
    """Return a unit vector approximating the trajectory's heading in the X-Y
    plane over the first k samples. Falls back to NaN if the segment is too
    short or stationary (caller decides what to do)."""
    k = min(k, len(points) - 1)
    if k < 1:
        return np.array([np.nan, np.nan])
    delta = points[k] - points[0]
    horiz = delta[:2]
    norm = np.linalg.norm(horiz)
    if norm < 1e-6:
        return np.array([np.nan, np.nan])
    return horiz / norm


def start_lock(vo_cam: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    """Camera-frame VO -> ENU-locked-at-GT-start.

    Steps:
      1. Apply the fixed camera-to-ENU basis swap.
      2. Translate so the swapped VO starts at GT[0].
      3. Yaw-rotate about world Z so the first-K-samples horizontal heading
         of VO matches the first-K-samples horizontal heading of GT.

    Returns the start-locked VO trajectory in the GT (ENU) frame.
    """
    vo = (CAM_TO_ENU @ vo_cam.T).T
    vo = vo - vo[0] + gt[0]

    h_vo = initial_heading(vo - gt[0], k)
    h_gt = initial_heading(gt - gt[0], k)
    if np.isnan(h_vo).any() or np.isnan(h_gt).any():
        print("[warn] Could not estimate initial heading (segment too short or "
              "stationary at start); skipping yaw alignment.", file=sys.stderr)
        return vo

    # Signed angle from VO heading to GT heading about the +Z axis.
    cross_z = h_vo[0] * h_gt[1] - h_vo[1] * h_gt[0]
    dot = float(np.clip(h_vo @ h_gt, -1.0, 1.0))
    theta = float(np.arctan2(cross_z, dot))

    c, s = np.cos(theta), np.sin(theta)
    R_yaw = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]])

    centered = vo - gt[0]
    return (R_yaw @ centered.T).T + gt[0]


def plot_2d(gt, vo, rmse_3d, save_to=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt[:, 0], gt[:, 1], color="C3", linewidth=1.6, label="GT")
    ax.plot(vo[:, 0], vo[:, 1], color="C2", linewidth=1.1, label="VO (start-locked)")
    ax.scatter(gt[0, 0], gt[0, 1], color="black", marker="o", s=40,
               zorder=5, label="start")
    ax.scatter(gt[-1, 0], gt[-1, 1], color="black", marker="x", s=40,
               zorder=5, label="GT end")
    ax.scatter(vo[-1, 0], vo[-1, 1], color="C2", marker="x", s=40,
               zorder=5, label="VO end")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (East) [m]")
    ax.set_ylabel("Y (North) [m]")
    ax.set_title(f"Top-down trajectory   |   RMSE 3D = {rmse_3d:.3f} m")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=130, bbox_inches="tight")
        print(f"Saved 2D plot: {save_to}")
    return fig


def plot_3d(gt, vo, rmse_3d, save_to=None):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color="C3", linewidth=1.6, label="GT")
    ax.plot(vo[:, 0], vo[:, 1], vo[:, 2], color="C2", linewidth=1.1,
            label="VO (start-locked)")
    ax.scatter(*gt[0], color="black", marker="o", s=40, label="start")
    ax.scatter(*gt[-1], color="black", marker="x", s=40, label="GT end")
    ax.scatter(*vo[-1], color="C2", marker="x", s=40, label="VO end")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"3D trajectory   |   RMSE 3D = {rmse_3d:.3f} m")
    ax.legend(loc="best")
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=130, bbox_inches="tight")
        print(f"Saved 3D plot: {save_to}")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Path to playback_log.csv produced by PathPlayer.")
    ap.add_argument("--heading-samples", type=int, default=DEFAULT_HEADING_SAMPLES,
                    help=f"Number of leading samples used to estimate the "
                         f"initial heading for yaw lock (default: "
                         f"{DEFAULT_HEADING_SAMPLES}).")
    ap.add_argument("--save", default=None,
                    help="Prefix for output PNGs (<prefix>_2d.png, "
                         "<prefix>_3d.png). When omitted, the plots are shown "
                         "interactively.")
    args = ap.parse_args()

    df = load_active_segment(args.csv)
    if len(df) < 10:
        sys.exit("Not enough tracked frames after dropping inactive prefix.")

    gt = df[["gt_x", "gt_y", "gt_z"]].to_numpy(dtype=float)
    vo = df[["vo_x", "vo_y", "vo_z"]].to_numpy(dtype=float)

    vo_locked = start_lock(vo, gt, k=args.heading_samples)

    err = vo_locked - gt
    rmse_3d = float(np.sqrt((err ** 2).sum(axis=1).mean()))

    print(f"Frames used:  {len(df)}")
    print(f"RMSE (3D):    {rmse_3d:.4f} m")

    save_2d = f"{args.save}_2d.png" if args.save else None
    save_3d = f"{args.save}_3d.png" if args.save else None
    plot_2d(gt, vo_locked, rmse_3d, save_to=save_2d)
    plot_3d(gt, vo_locked, rmse_3d, save_to=save_3d)

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
