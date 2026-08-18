# VisualOdometryVINS

Monocular visual-inertial odometry: this project's **frontend** (detectors,
matchers, optical flow, camera models) on top of **VINS-Fusion's estimator
math** as the backend.

## Why this fork exists

The parent project could not estimate metric scale. Its pose estimator returned
a normalised translation and the magnitude came from `GtScaleEstimator`
(the norm of a ground-truth position delta) or from an altimeter via the
Homography path. Every attempt to add an inertial component to that stack
failed, in three separate architectures, because the IMU's one irreplaceable
contribution — scale — was already being supplied by a cheat, and the pose
estimator discarded the metric content before any fusion could see it.

VINS-Fusion solves exactly that, and does so on this project's own data:
**68.7 m ATE over a 4321 m flight (1.6 % drift) on `bell412_dataset3`, with
scale recovered from the IMU alone.** Replicating that estimator from scratch
is a large piece of work; integrating it is not. So the backend is imported and
the frontend stays ours, which is the half worth experimenting with.

**The value of `vins/` is the estimator math** — IMU preintegration, the
sliding-window solve, marginalisation, and the visual-inertial initialisation
that makes scale observable. Everything that merely wrapped that math for a ROS
node has been deleted rather than shimmed. This tree is **not** kept diffable
against upstream VINS-Fusion; it is a controlled integration.

## Architecture

```
rosbag ─► image ─► ICameraModel ─► IFrontend ─► OutlierRejector ─► VinsBackend
                   (undistort or   (KLT /       (essential /       (VINS
                    lift points)    descriptors) fundamental / H)   estimator)
                                                                      │
IMU ──────────────────────────────────────────────────────────────────┤
                                                                      ▼
                                                        TrajectoryAligner ─► CSV
```

- `odometry/frontend/` — the experimental surface. `OpticalFlowFrontend`
  (Shi-Tomasi + FB-LK, persistent track ids) works today; `DescriptorFrontend`
  needs a track manager first (see TODO).
- `odometry/camera/` — Pinhole and Kannala-Brandt. Two undistortion modes,
  `camera.undistort: image | points`.
- `odometry/vins/VinsBackend` — the adapter. Feeds
  `Estimator::inputFeature()`, which bypasses VINS's own tracker entirely.
- `vins/` — the vendored estimator, 35 files. No ROS, no camera models, no
  feature tracker.

Only `RosbagEvaluator` and `MatchDebugger` are built. `main_odomlog.cpp` is
kept in the tree but unbuilt (see TODO).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DROS_ROOT=/home/lysenko/miniforge3/envs/ros_env \
      -DONNXRUNTIME_DIR=/home/lysenko/VO-testing/claude-sandbox/onnxruntime-1.18.1
cmake --build build -j$(nproc)
```

Ceres comes from `ROS_ROOT` unless `-DCERES_DIR=` says otherwise, and is found
by path rather than `find_package` (the conda `CeresConfig.cmake` chains
`find_dependency` through SuiteSparse → METIS → OpenMP and does not resolve
against the system compiler).

## Run

```bash
cd build
conda activate ros_env
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libglib-2.0.so.0 \
            /usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0 \
            /usr/lib/x86_64-linux-gnu/libgio-2.0.so.0" \
  ./RosbagEvaluator ../configs/mun/mun3_optical_imagefull.yaml --no-gui \
    --out ../results/mun3_optical_imagefull/trajectory.csv
```

The `LD_PRELOAD` is environment drift, not a project requirement: the conda env
ships a `libgio` older than the system `libglib` that OpenCV's GTK backend links
against. It affects the parent project's binaries too.

Long runs should be launched with `setsid ... & disown` — a plain `nohup` child
gets reaped when its parent shell exits.

## Evaluate

```bash
~/venv/bin/python scripts/compare_frl.py results --plot   # project scorer + graphs
scripts/evaluate_mun3_frl.sh results/<run>/trajectory.csv # evo, yaw-only
python3 scripts/logreader.py summary results/<run>/run.jsonl
```

Both scorers are start-anchored with yaw-only alignment and no Umeyama fit —
deliberately, because a global 6-dof fit redistributes error and hides drift.
They differ in resampling, so **compare within one scorer, never across**.

`scripts/logreader.py` compresses a ~35k-record run log to a screenful and runs
health checks that encode bugs this pipeline has actually hit (feed rate,
feature density, track carry-over, estimate-vs-GT axis correlation, scale,
solver determinism). Read the JSONL through it, never raw.

## Current results — `bell412_dataset3`, optical flow, no GT scale, no altimeter

| run | undistort | res | ATE (evo) | drift | notes |
| --- | --- | --- | --- | --- | --- |
| `mun3_optical_ransac`    | image  | 1440×1080 | — | — | + essential/MAGSAC |
| `mun3_optical_imagefull` | image  | 1440×1080 | 230.98 m | 5.35 % | best |
| `mun3_optical`           | image  | 720×540   | 247.44 m | 5.73 % | |
| `mun3_optical_points`    | points | 1440×1080 | 254.08 m | 5.88 % | |
| VINS-Fusion reference    | — | — | 68.7 m | 1.6 % | target |

Runs are reproducible bit-for-bit. The remaining ~3.4× gap to VINS is frontend.

**Read that table with a caveat.** Yaw-fit error currently exceeds trajectory
error: true shape error is ~84 m, and the 158–231 m reported depends on which
yaw estimate the scorer used. Differences smaller than ~20 m are inside the
alignment noise floor.

## TODO

Roughly in order of expected payoff.

1. **Tighten the yaw alignment.** It dominates the metric. The 10 m trigger
   fires 10.9 s after tracking starts at 0.93 m/s horizontal while the
   helicopter climbs, so the fit comes from near-hover wander. `aligner.min_speed`
   exists for this but is unset. Also report a yaw-optimal ATE alongside the
   start-anchored one, so estimator quality and heading quality stop being
   confounded.
2. **Feature distribution sweep.** `max_corners` / `min_distance` at 1440×1080
   were never tuned; the perf table now prices each setting.
3. **Real feature velocities.** Zeroed today — exact while `estimate_td: 0`,
   but required if online td estimation is ever enabled.
4. **TrackManager**, so descriptor frontends can be fed. VINS keys landmarks and
   its keyframe decision on persistent ids; descriptor matching against a
   keyframe anchor breaks them at every promotion. This unlocks the detector /
   matcher comparison the fork exists for — and the `points` undistort mode,
   whose effect on descriptors should be much larger than on optical flow.
5. **Revive `OdomLogEvaluator`.** Still on the pre-VINS pipeline API. Its
   captures carry ~20 Hz IMU, too slow to preintegrate well, so it needs a
   faster IMU log before it is worth much.
6. **Calibrate `heading.mount_yaw_offset` properly** — once, on the ground,
   against a known bearing. Never fitted to a flight.
7. **Stereo / multi-camera.** `VinsBackend` already accepts a second camera_id
   under the same feature id, which is how VINS recognises a stereo pair.
