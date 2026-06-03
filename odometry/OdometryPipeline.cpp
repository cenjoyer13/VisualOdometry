#include "OdometryPipeline.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <utility>

#include "frontend/FrontendFactory.h"
#include "pose_estimators/PoseEstimatorFactory.h"
#include "scale_estimators/ScaleEstimatorFactory.h"
#include "integrators/IntegratorFactory.h"

std::unique_ptr<OdometryPipeline> OdometryPipeline::build(const OdometryConfig& config) {
    std::cout << "[OdometryPipeline] Initiating build sequence..." << std::endl;
    return std::make_unique<OdometryPipeline>(
        config,
        FrontendFactory::create(config),
        PoseEstimatorFactory::create(config),
        ScaleEstimatorFactory::create(config),
        IntegratorFactory::create(config)
    );
}

OdometryPipeline::OdometryPipeline(const OdometryConfig& cfg,
                                   std::unique_ptr<IFrontend> f,
                                   std::unique_ptr<IPoseEstimator> p,
                                   std::unique_ptr<IScaleEstimator> s,
                                   std::unique_ptr<ITrajectoryIntegrator> i)
    : config(cfg),
      frontend(std::move(f)),
      pose_estimator(std::move(p)),
      scale_estimator(std::move(s)),
      integrator(std::move(i)),
      is_first_frame(true)
{
    if (config.use_local_ba) {
        cv::Mat K = (cv::Mat_<double>(3, 3) <<
            config.intrinsics.fx, 0.0, config.intrinsics.cx,
            0.0, config.intrinsics.fy, config.intrinsics.cy,
            0.0, 0.0, 1.0);

        lba_ = std::make_unique<LocalBundleAdjustment>(K, config.lba_params, config.imu_params, config.verbose);
        lba_->start();
    }

    std::cout << "[OdometryPipeline] Pipeline successfully assembled and ready." << std::endl;
}

void OdometryPipeline::addImu(double t, const cv::Vec3d& acc, const cv::Vec3d& gyr) {
    // Phase 0: buffer only. Preintegration consumes this in a later phase.
    imu_buffer_.push_back(ImuSample{t, acc, gyr});
}

void OdometryPipeline::processFrame(DeviceBuffer& frame, const GroundTruthData& current_gt) {
    // Legacy vision-only path: synthesize a monotonic timestamp so IMU-aware
    // callers and vision-only callers share one implementation.
    processFrame(frame, current_gt, static_cast<double>(synthetic_frame_counter_++));
}

void OdometryPipeline::processFrame(DeviceBuffer& frame, const GroundTruthData& current_gt, double timestamp) {
    last_frame_time_ = timestamp;  // Phase 0: stored for upcoming IMU preintegration
    auto t_start_total = std::chrono::high_resolution_clock::now();

    // Bootstrap: first frame has no predecessor; the frontend stores the anchor.
    if (is_first_frame) {
        frontend->initialize(frame);
        gt_prev = current_gt;
        is_first_frame = false;
        // Debug frame for frame 0 is the raw input (no tracks to draw yet).
        debug_frame = frame.getAsCPU().clone();
        return;
    }

    // Frontend: correspondences of the current frame against the keyframe anchor
    // (detect+match for the descriptor frontend, KLT for optical flow).
    FrontendResult fr = frontend->process(frame);
    metrics.time_detect_ms = fr.detect_ms;
    metrics.time_match_ms = fr.match_ms;
    debug_frame = fr.debug_overlay;

    // Pose recovery.
    auto t0 = std::chrono::high_resolution_clock::now();
    cv::Mat R, t;
    bool pose_success = pose_estimator->estimatePose(fr.pts_prev, fr.pts_curr, config.intrinsics, R, t);
    auto t1 = std::chrono::high_resolution_clock::now();
    metrics.time_pose_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (pose_success) {
        // The estimator returns a zero translation when it cannot recover
        // motion (parallax below min_disparity, too few matches, or a
        // degenerate decomposition).
        const double norm_t = cv::norm(t);
        const bool stationary = (norm_t <= 1e-6);
        const int num_matches = static_cast<int>(fr.pts_prev.size());

        // Keyframing decision:
        //  - real keyframe: motion recovered (enough parallax) -> integrate.
        //  - forced keyframe: parallax still too low, but holding the anchor
        //    longer risks matching collapse, so cut a keyframe to keep tracking
        //    alive (matches near the floor, or held for too many frames).
        //  - otherwise hold the anchor: the next frame keeps matching against
        //    this keyframe, so both parallax and the GT baseline that sets the
        //    metric scale accumulate until a usable motion appears. This is the
        //    fix for high-FPS scale loss: advancing on every sub-threshold
        //    frame would integrate zero while moving gt_prev forward, dropping
        //    the GT motion across the gap.
        const bool forced = stationary &&
            (num_matches < config.keyframe_min_matches ||
             frames_since_keyframe_ >= config.keyframe_max_skip);

        if (!stationary || forced) {
            // Scale source:
            //  - altimeter: when an altitude (AGL) is provided and the homography
            //    estimator is active, its translation is the raw t/d, so the
            //    metric scale is the plane distance d = AGL and t is left as-is.
            //  - otherwise: normalize t to a unit direction and take the metric
            //    distance from the scale estimator (the existing path; scale
            //    spans the whole keyframe-to-current baseline since gt_prev was
            //    held across any skipped frames). A forced keyframe has t == 0,
            //    so it integrates nothing either way.
            double scale;
            const bool altimeter = config.pose_estimator_type == "Homography" &&
                                   current_gt.altitude > 0.0f;
            if (altimeter) {
                scale = current_gt.altitude;
            } else {
                const double nt = cv::norm(t);
                if (nt > 1e-6) t = t / nt;
                scale = scale_estimator->updateScale(gt_prev, current_gt);
            }
            integrator->integrate(R, t, scale);

            // Advance the frontend anchor (the new keyframe) before pushing to
            // LBA so the next match's indices line up with what was just pushed.
            frontend->promoteKeyframe(frame);
            gt_prev = current_gt;
            frames_since_keyframe_ = 0;

            // Slice the IMU buffer for this keyframe interval (gyro/vio modes).
            // Drained up to this frame's time; empty when no IMU is active.
            std::vector<ImuSample> kf_imu;
            if (config.imu_params.enabled()) {
                for (const auto& s : imu_buffer_)
                    if (s.t > last_keyframe_time_ && s.t <= last_frame_time_)
                        kf_imu.push_back(s);
                imu_buffer_.erase(
                    std::remove_if(imu_buffer_.begin(), imu_buffer_.end(),
                        [&](const ImuSample& s) { return s.t <= last_frame_time_; }),
                    imu_buffer_.end());
                last_keyframe_time_ = last_frame_time_;
            }

            if (lba_ && !R.empty() && !t.empty()) {
                // Snapshot the integrator's world pose AFTER integrate() so the
                // snapshot reflects this frame's pose in the world. The LBA
                // correction is later computed as a world-frame delta against
                // this exact snapshot. Forced keyframes push with is_stationary
                // so LBA holds them as a tight identity rather than fitting the
                // unobserved motion, and the track chain stays contiguous.
                const uint64_t frame_id = static_cast<uint64_t>(current_frame_id_);
                integrator->snapshotPose(frame_id);

                BAFrame new_frame;
                new_frame.frame_id = frame_id;
                new_frame.R = R.clone();
                new_frame.t = t.clone() * scale;  // metric-scaled
                new_frame.is_stationary = stationary;
                cv::Mat snap;
                integrator->getSnapshot(frame_id, snap);
                new_frame.T_world = snap;

                new_frame.points2D = fr.points2D;
                new_frame.matched_prev_idx = fr.matched_prev_idx;
                new_frame.imu_samples = std::move(kf_imu);

                lba_->pushFrame(new_frame);
            }

            if (lba_) {
                BACorrection corr;
                if (lba_->getCorrection(corr)) {
                    integrator->applyCorrection(corr.frame_id, corr.T_world_optimized);
                }
            }

            current_frame_id_++;
        } else {
            // Hold the keyframe; anchor and gt_prev unchanged so parallax and
            // the scale baseline keep accumulating.
            frames_since_keyframe_++;
        }
    }

    // Wall-clock totals and rolling FPS.
    auto t_end_total = std::chrono::high_resolution_clock::now();
    metrics.time_total_ms = std::chrono::duration<double, std::milli>(t_end_total - t_start_total).count();
    metrics.fps = 1000.0 / metrics.time_total_ms;
}

cv::Mat OdometryPipeline::getGlobalTransformVO() const {
    return integrator->getGlobalTransformVO();
}

bool OdometryPipeline::isTrackingActive() const {
    return !is_first_frame; 
}

OdometryPipeline::~OdometryPipeline() {
    if (lba_) {
        lba_->stop();
    }
}
