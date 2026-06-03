# IMU (gyro) + altimeter-scale plan

Status: **Phase 0 + Phase 1 (gyro) done & verified. Full VIO dropped (see
decision record). Remaining: Phase 2 — altimeter-from-homography metric scale.**

Scope is now deliberately narrow: use the IMU **gyro** for rotation, and resolve
the monocular **scale** geometrically from an **altimeter + homography**, not
from the accelerometer. No velocity/bias/gravity states, no `CombinedImuFactor`.

Reference for the gyro preintegration: VINS-Fusion-gpu
(`~/VINS-melodic/src/VINS-Fusion-gpu`); we use GTSAM's `gtsam::Rot3` for the
integration. Calibration: [[reference-mun-frl-imu-calib]].

## Decisions (locked)
- **IMU = gyro only.** `imu.mode: off | gyro` (the `vio` mode is removed).
  - `off` (default) — identical to today, byte-for-byte.
  - `gyro` — preintegrated gyro → rotation-only `BetweenFactor<Pose3>` between
    consecutive LBA keyframes (tight on rotation, free on translation).
- **No full VIO.** The accelerometer is not integrated. Rationale in the
  decision record below: on this nadir/low-excitation/rotor-vibration setup the
  accel can *actively hurt* accuracy and would compete with a better scale
  source.
- **Scale from altimeter + homography (Phase 2).** For a nadir camera the
  ground plane gives `decomposeHomographyMat` translation as `t/d` and the
  altimeter gives `d = AGL`, so `t_metric = (t/d)·AGL`. Drift-free and better
  conditioned than accel scale.
- **Pipeline flexibility.** The pipeline can accept an optional **altitude
  (AGL)** reading per frame. When an altitude is present **and** the homography
  estimator is active, scale is taken from the altitude and the homography
  translation is **not normalized**. Otherwise behavior is unchanged
  (translation normalized + the existing `IScaleEstimator`).
- **Frame/time sync (done, verified).** Gyro is rotated into the camera frame
  via `R_cam_imu = rotation(body_T_cam0)^T`, and the camera-IMU offset `imu.td`
  (`image_clock + td = imu_clock`) is applied. Gyro vs VO rotation agree to
  ~0.07 deg.

## Hard constraint
The gyro factor lives in the GTSAM LBA graph, so `mode: gyro` requires
`local_bundle_adjustment.enabled: 1`. With LBA off or `mode: off`, no IMU
effect. The altimeter scale (Phase 2) is independent of the LBA and works with
or without it.

## Altimeter-from-homography scale (Phase 2 design)
Geometry: the ground plane is horizontal; the camera-to-plane perpendicular
distance equals the altitude, so `d = AGL` regardless of tilt.
`decomposeHomographyMat(H, K)` returns translation normalized by `d`, i.e.
`t/d`; multiplying by `AGL` recovers metric translation. Essential-matrix `t`
has no `t/d` meaning (pure direction) and must stay normalized — so this is
**homography-only**.

Implementation sketch (small, contained):
1. **Altitude channel.** Add `altitude` (AGL, m; NaN = absent) to
   `GroundTruthData`. `main_rosbag` fills it from the already-computed
   `cur_alt - baseline_agl`. Other mains leave it NaN.
2. **Stop normalizing in the cheirality base** — return the raw winning `best_t`
   (essential's is already unit from `decomposeEssentialMat`; homography's is
   `t/d`). Move the unit-normalization into the pipeline's fallback branch so
   existing behavior is preserved.
3. **Pipeline per keyframe:**
   - if `altitude` valid **and** `pose_estimator == Homography`:
     `scale = AGL`, integrate the raw `t/d` → metric.
   - else: normalize `t` (guard `norm > eps`; stationary stays zero),
     `scale = scale_estimator.updateScale(...)` (existing path).
4. **`AltimeterScaleEstimator`** (`IScaleEstimator`) returning `gt.altitude`, or
   the pipeline computes it inline — TBD at implementation.

Validate empirically that the raw `t/d` magnitude is stable at the keyframe
baselines (decomposeHomographyMat scale can be noisy at very low parallax); if
jumpy, fall back to using only the direction and deriving magnitude from
`AGL`-based baseline geometry.

## Config schema (additive; absent ⇒ disabled / unchanged)
```yaml
imu:
  mode: off | gyro            # default off
  topic: /imu/data
  gyr_n: 0.004                # dataset value
  gyro_rot_sigma: 0.002       # rotation-prior sigma (tighter = trust gyro more)
  td: -0.0297405129318        # camera-IMU time offset
# acc_n / acc_w / gyr_w / g_norm are read but unused while accel is out.
```
Extrinsic reused from top-level `body_T_cam0`. Phase-2 altitude scale: driven by
the altitude input + `pose_estimator: Homography` (config knob TBD).

## Phases
- **Phase 0 — plumbing. DONE.** `imu:` block + `ImuParams`,
  `odometry/inertial/ImuTypes.h`, `addImu` buffer + timestamped `processFrame`
  overload.
- **Phase 1 — `gyro`. DONE.** `ImuPreintegrator::integrateGyro` (→ `Rot3`,
  rotated by `R_cam_imu`); per-keyframe IMU slice in `BAFrame::imu_samples`;
  rotation-only `BetweenFactor<Pose3>` in the LBA; `main_rosbag` feeds
  `/imu/data` + timestamps + `td`. Verified.
- **Phase 2 — altimeter-from-homography scale.** As designed above. Replaces the
  dropped `vio` phase.
- ~~Phase (vio)~~ — **dropped.**

## Files
New (`odometry/inertial/`): `ImuTypes.h`, `ImuPreintegrator.{h,cpp}` (done).
Phase 2 touches: `OdometryTypes.h` (`GroundTruthData.altitude`),
`HomographyCheiralityPoseEstimator` / `CheiralityPoseEstimator` (no
normalization), `OdometryPipeline` (scale branch), maybe a new
`AltimeterScaleEstimator`, `main_rosbag` (fill altitude). CMake: none.

## Backwards-compatibility guarantees
- No `imu:` block ⇒ outputs identical to today; `mode: off` is byte-for-byte.
- No altitude fed (NaN) ⇒ scale path unchanged (normalize + existing estimator).
- The pose-only LBA graph is unchanged when IMU off.
- 2-arg `processFrame` preserved; KittiEvaluator/PathPlayer untouched.

## Decision record — why no full VIO
The accelerometer can *actively hurt* accuracy on this dataset:
- **Scale competition.** Accel scale would be jointly estimated with the
  altimeter; in steady cruise (low linear acceleration) accel scale is poorly
  observable, so the worse source drags the near-perfect (<0.1 %, drift-free)
  altimeter scale off.
- **Rotor vibration.** Periodic vibration aliases into accel preintegration as
  spurious Δv/Δp (gyro is largely immune — hence its 0.07 deg agreement).
- **Bias/gravity coupling under low excitation** can inject systematic
  tilt/scale error; **td/init sensitivity** is higher for translation than
  rotation.
The only genuine VIO benefit here is dead-reckoning across vision dropouts,
which is rare for high-altitude nadir over textured terrain. If such dropouts
show up, revisit VIO with accel **subordinate** to the altimeter (robust loss,
conservative weight, rotor-frequency notch filter) — not as a free add-on.
