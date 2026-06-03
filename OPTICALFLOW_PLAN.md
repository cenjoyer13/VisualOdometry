# Optical-flow frontend plan (Shi-Tomasi + KLT via `IFrontend`)

Status: **Design only. Sequenced AFTER the IMU work** (`IMU_PLAN.md`: Phase 0/1
done, Phase 2 altimeter scale pending). Do not start until that is fully
finished — the two are independent (frontend vs backend), but sequencing avoids
churn on the same `OdometryPipeline` file.

## Goal
Add Shi-Tomasi corners + Lucas-Kanade (KLT) optical flow as a first-class
alternative to descriptor matching, **without duplicating the backend**
(pose / scale / integrator / keyframing / LBA). Achieved by extracting the
correspondence step behind an `IFrontend` interface with two implementations.

## Why not the existing detector/matcher interfaces
KLT does not fit `IFeatureDetector`/`IFeatureMatcher`:
1. **Needs the images** — `match(prev_desc, curr_desc, prev_kpts, curr_kpts)`
   gets no images; `calcOpticalFlowPyrLK` needs both pyramids.
2. **Produces tracked locations, not matches into a detected set** — the
   pipeline does `pts_curr = curr_keypoints[match.trainIdx]`, assuming current
   points came from `detect(curr)`. KLT tracks to sub-pixel locations not in any
   `curr_keypoints` (which is const, can't be overwritten).
3. **Stateful/track-based** vs the stateless detect+match.
Only the *frontend* differs; `estimatePose`, the integrator, the keyframe-with-
cap logic, the IMU slice, and the LBA are identical regardless of how
correspondences are produced. That is the seam to cut.

## `IFrontend` design
A frontend owns the keyframe anchor and produces, per frame, the matched point
pairs (for pose) plus the LBA observations. It does **not** decide keyframes —
the pipeline still does that from pose stationarity + caps and tells the
frontend when to advance the anchor.

```cpp
struct FrontendResult {
    std::vector<cv::Point2f> pts_prev;   // matched in the keyframe anchor
    std::vector<cv::Point2f> pts_curr;   // matched in the current frame
    // For LBA:
    std::vector<cv::Point2f> points2D;        // all current observations
    std::vector<int>         matched_prev_idx;// curr idx -> prev idx, -1 if new
    std::vector<int64_t>     track_ids;       // persistent IDs (optional; OF-native)
    cv::Mat                  debug_overlay;    // optional tracks viz
};

class IFrontend {
public:
    virtual ~IFrontend() = default;
    virtual void initialize(DeviceBuffer& frame) = 0;        // first frame: set anchor
    virtual FrontendResult process(DeviceBuffer& frame) = 0; // vs current anchor; no advance
    virtual void promoteKeyframe(DeviceBuffer& frame) = 0;   // pipeline-driven anchor advance
};
```

Implementations:
- **`DescriptorFrontend`** — owns the existing `IFeatureDetector` +
  `IFeatureMatcher` (built via the current factories). `process` =
  detect(curr) + match(anchor, curr); `promoteKeyframe` stores curr
  kpts/descriptors. Reproduces today's behavior byte-for-byte.
- **`OpticalFlowFrontend`** — Shi-Tomasi (`goodFeaturesToTrack`) + KLT
  (`calcOpticalFlowPyrLK`):
  - Tracks **frame-to-frame** (robust short baseline), keeping persistent track
    IDs; reports **anchor→current** correspondences via track continuity, so the
    keyframe/parallax logic is unchanged.
  - **Forward-backward error** + status mask for outlier rejection.
  - **Replenish** corners (Shi-Tomasi, bucketed) when live track count drops
    below a floor or on `promoteKeyframe`.
  - State: `prev_image` (last frame, for f2f KLT) + live tracks
    `{id, anchor_pos, current_pos}` + keyframe image.

## Pipeline refactor (`OdometryPipeline`)
`processFrame` frontend section collapses to:
1. first frame → `frontend_->initialize(frame)`; seed `gt_prev`; return.
2. `auto r = frontend_->process(frame);`
3. `estimatePose(r.pts_prev, r.pts_curr) -> R, t`.
4. keyframe decision (stationary / `keyframe_min_matches` / `keyframe_max_skip`)
   — **unchanged**.
5. on keyframe: integrate, `frontend_->promoteKeyframe(frame)`, advance
   `gt_prev`, IMU slice, LBA push (using `r.points2D` / `r.matched_prev_idx` /
   `r.track_ids`).
6. else: hold (no promote, no integrate) — anchor stays; parallax accumulates.

The pipeline stops holding `prev_image`/`prev_keypoints`/`prev_descriptors`
(now frontend-owned) and the `detector`/`matcher` members (now inside
`DescriptorFrontend`). The IMU buffer/slice, keyframe caps, scale, integrator,
and LBA are untouched.

## LBA synergy (optional follow-on)
Optical flow yields genuinely persistent track IDs across many frames — exactly
what `SmartProjectionPoseFactor` wants. Initially feed `matched_prev_idx` and
let the LBA keep `assignTrackIDs` as-is (no LBA change). Later, route
`r.track_ids` straight into the LBA to get longer, cleaner tracks (likely helps
on the nadir/low-parallax scene).

## Config schema (additive)
```yaml
frontend: descriptor | optical_flow   # default descriptor (current behavior)

optical_flow:
  max_corners: 1000
  quality_level: 0.01
  min_distance: 10
  win_size: 21
  max_level: 3
  fb_error_threshold: 1.0     # forward-backward px
  min_tracks: 150             # replenish below this
```
`frontend: descriptor` uses the existing `detector:`/`matcher:` blocks.

## Files
New `odometry/frontend/`: `IFrontend.h`, `DescriptorFrontend.{h,cpp}`,
`OpticalFlowFrontend.{h,cpp}`, `FrontendFactory.{h,cpp}`.
Touched: `OdometryTypes.h` (`frontend_type` + optical-flow params;
`FrontendResult` lives in `IFrontend.h`), `ConfigLoader.{h,cpp}` (read
`frontend` + `optical_flow`), `OdometryPipeline.{h,cpp}` (own `IFrontend`
instead of detector/matcher/anchor state), `OdometryPipeline::build`
(construct via `FrontendFactory`). CMake/presets: none (new dir auto-globs;
OpenCV `video`/`imgproc` already linked for KLT/goodFeaturesToTrack).

## Backwards-compatibility
- `frontend` absent or `descriptor` ⇒ identical to today (DescriptorFrontend
  wraps the same detector+matcher). KITTI/AirSim/MUN configs unchanged.
- The keyframe-with-cap logic, IMU gyro path, and pose-only/IMU LBA are
  unaffected — they live in the backend.

## Phasing
- **OF-0 — extract.** Introduce `IFrontend` + `DescriptorFrontend` +
  `FrontendFactory`; move detect+match and anchor state out of the pipeline.
  Pure refactor. **Acceptance: byte-for-byte identical CSVs** on a KITTI/MUN
  baseline vs the pre-refactor binary.
- **OF-1 — optical flow.** `OpticalFlowFrontend` (Shi-Tomasi + KLT + FB error +
  replenish), config-selectable.
- **OF-2 (optional) — track IDs to LBA.** Route native OF track IDs into the
  LBA, bypassing `assignTrackIDs` for the OF path.

## Risk / validation
- Main risk is the refactor moving anchor state cleanly — mitigated by the OF-0
  byte-for-byte gate before any KLT is added.
- Validate KLT on the nadir aerial data: tune `win_size`/`max_level` for the
  720x540 undistorted frames and large inter-frame motion (helicopter 13-30
  m/s); forward-backward error is the key outlier guard at low parallax.
- KLT pairs naturally with the homography pose estimator + altimeter scale
  (`IMU_PLAN.md` Phase 2) — both are point-based and frame-convention agnostic.
