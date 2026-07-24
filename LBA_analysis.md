# Local Bundle Adjustment: implementation analysis vs. state of the art

Analysis of `odometry/LocalBundleAdjustment.{h,cpp}` in the VisualOdometry
pipeline, compared against published windowed-BA / VIO systems.

## 1. What the current LBA actually is

A **structureless, fixed-window, batch GTSAM bundle adjustment running as a
decoupled output smoother**:

| Aspect | Current implementation |
|---|---|
| Solver | GTSAM **batch Levenberg-Marquardt**, re-solved from scratch every `opt_stride` (2) frames |
| Structure | **Structureless** - `SmartProjectionPoseFactor<Cal3_S2>` with HESSIAN linearization; landmarks Schur-eliminated, never variables |
| Window | Fixed **temporal** window of 10 keyframes; oldest frame **erased** on overflow (`pruneOutdatedTracks` only decrements track counts) |
| "Marginalization" | **None.** A dropped frame's information is lost. Continuity comes only from writing optimized poses back into the window and re-adding a tight prior on the new oldest pose |
| Gauge / drift cap | Tight anchor prior on oldest pose + **loose prior on the newest pose at its integrator snapshot value** |
| Odometry | `BetweenFactor<Pose3>` chain from frontend `(R,t)`; locks scale; stationary frames -> tight identity |
| IMU | Optional **gyro-only** rotation `BetweenFactor`; translation sigma 1e6 (free). No accelerometer, no preintegrated metric factor |
| Robustness | `ZERO_ON_DEGENERACY`, dynamic outlier rejection, bbox-parallax filter, min-observations. **No Huber/Cauchy** (incompatible with HESSIAN Schur) |
| Coupling | **Decoupled** smoother: never corrects the live integrator; a pass is published only if it clears heuristic gates (min smart-factor count, translation/rotation caps) |
| Scale | Monocular; scale supplied externally (altimeter / GT) |

It is **not naive** - structureless smart-factor windows are a legitimate SOTA
family (MSCKF, Kimera-VIO). It occupies a specific corner of the design space.

## 2. State-of-the-art landscape

| System | Window mgmt | Structure | Solver | Robust | IMU | Consistency |
|---|---|---|---|---|---|---|
| **VINS-Mono/Fusion** | Sliding window + **Schur marginalization prior** | explicit landmarks (inverse depth) | Ceres LM | Huber | full preintegration | - |
| **ORB-SLAM3** | **Covisibility** local BA; non-window KFs fixed | explicit landmarks | g2o LM | **Huber** | full preintegration (VI) | - |
| **DSO** | ~7 KF + **dynamic marginalization** | explicit (inverse depth, photometric) | custom GN | photometric robust | (VI variant) | **First-Estimate Jacobians** |
| **OKVIS / BASALT** | Sliding window + marginalization (**sqrt** in BASALT) | explicit | GN/LM | Huber | full preintegration | FEJ / sqrt prior |
| **iSAM2 (GTSAM)** | **Incremental** Bayes tree, fluid relinearization | explicit | incremental GN/LM | Huber | full | exact incremental |
| **This LBA** | Fixed temporal window, **drop + re-prior** | **structureless** smart factors | batch LM | dynamic-outlier only | gyro-rotation only | none |

Two near-universal SOTA ideas the current code does not implement: (1) a
**marginalization prior** that folds a departing state's information (Schur
complement, or its numerically superior square-root form) into a dense prior on
the survivors; (2) **per-observation robust kernels** (Huber).

## 3. Gap analysis

**a) No marginalization is the central structural gap.** The loose prior on the
newest pose is a heuristic stand-in: it re-injects the integrator's own
(un-optimized) estimate as a pseudo-measurement with a hand-tuned sigma, rather
than carrying the Hessian/covariance of marginalized states. The in-code comment
"the optimizer then partly undoes its own work each cycle" is the exact symptom a
proper marginalization prior prevents.

**b) The decoupled + gated correction path is a symptom, not a feature.** SOTA
estimators are tightly coupled - the optimized window *is* the state. Here the
window is optimized, then filtered through caps and a min-factor gate. The
project's own notes record why: adaptive LBA helps mun4 but regresses
well-behaved sequences because "the harm originates upstream
(directionally-biased relative-pose estimates) and is invisible to the BA
window." A short window of *relative* BetweenFactors + structureless reprojection
with no absolute/marginal prior cannot catch a directional bias - the gauge
freedom hides it.

**c) Structureless forecloses Huber.** Eliminating landmarks keeps the state tiny
and fast (genuinely SOTA), but HESSIAN Schur elimination is incompatible with a
per-observation robust kernel, so robustness falls to dynamic outlier rejection +
degeneracy zeroing - blunter than the Huber loss ORB-SLAM3/VINS use, exactly when
features starve (the mun4 failure mode).

**d) Temporal vs covisibility window.** A fixed 10-frame temporal window cannot
exploit high-parallax non-adjacent keyframes or re-observations the way
ORB-SLAM3's covisibility BA does. On near-planar aerial footage parallax is the
binding constraint, so this matters (though it edges into SLAM territory).

**e) Gyro-only IMU leaves the biggest lever unused.** The dataset has a full IMU,
but only gyro rotation is used and **scale is still the altimeter/GT cheat**.
Every VI SOTA system makes scale and gravity observable from a full preintegrated
IMU factor.

## 4. Conclusions

1. **Architecturally sound, not naive.** A structureless sliding-window BA in the
   MSCKF/Kimera smart-factor tradition on GTSAM batch LM. The robustness machinery
   and the deterministic synchronous mode are good engineering.

2. **A proper marginalization / fixed-lag rework is NOT the fix - it was already
   tried and reverted.** A prior session implemented the `fixedlag` /
   `SlidingWindowSmoother` backend with Schur/marginalization priors (still
   described in `odometry/CLAUDE.md` and the absent `SMOOTHER_REWORK.md`) and
   reverted it because it made LBA **dead weight that improved nothing**. That
   outcome is itself the finding: a *correct* marginalization smoother yielding no
   gain confirms the limiting factor is upstream - biased relative poses and no
   absolute scale constraint on near-planar, low-parallax aerial scenes - not the
   LBA backend. The BA window holds little exploitable information beyond what the
   frontend already provides.

3. **The decoupled, gate-heavy correction path is a workaround for that same
   upstream bias.** Tightening the coupling only helps *together with* an absolute
   (IMU) constraint; without it, it reproduces the mun4 regression.

4. **Full IMU preintegration (`CombinedImuFactor`) is the single most impactful,
   dataset-supported upgrade.** It makes metric scale and gravity observable,
   removes the altimeter/GT scale dependency, and aligns the system with every VI
   SOTA method. This attacks the actual root cause from conclusions #2-#3.

5. **Lower priority:** FEJ only matters once a marginalization prior is kept (moot
   here, see #2). Huber is unavailable under HESSIAN - accept the current outlier
   rejection or switch smart factors to an implicit/Jacobian-Q mode.

6. **Fix the docs.** `odometry/CLAUDE.md` still describes the reverted `fixedlag`
   backend, `IWindowSmoother`, `getSmoothedTrajectory()`, and `SMOOTHER_REWORK.md`
   that are not in the tree. Correct the documentation to match the legacy-only
   reality.

**Tie-back to the queue experiments:** LBA (`both`) clearly helps feature-rich
bell3 but regresses feature-starved bell4 - the expected signature of a
structureless short window with no marginalization/absolute prior and no Huber. It
sharpens good geometry but cannot reject directionally-biased relative poses when
features collapse. Conclusion #4 (full IMU) is the lever that addresses it.

## Sources
- Demmel et al., *Square-Root Marginalization for Sliding-Window BA*, 2021 - https://arxiv.org/abs/2109.02182
- *Square Root Bundle Adjustment for Large-Scale Reconstruction* - https://arxiv.org/pdf/2103.01843
- *ORB-SLAM3* (Campos et al., 2021) - https://arxiv.org/pdf/2007.11898
- *ORB-SLAM* (Mur-Artal et al., 2015) - https://ar5iv.labs.arxiv.org/html/1502.00956
- *Direct Sparse VIO using Dynamic Marginalization* (DSO/DSVIO) - https://ar5iv.labs.arxiv.org/html/1804.05625
- *iSAM2* (Kaess et al., IJRR 2012) - https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf
- GTSAM `SmartProjectionFactor` - http://docs.ros.org/en/kinetic/api/gtsam/html/SmartProjectionFactor_8h_source.html
