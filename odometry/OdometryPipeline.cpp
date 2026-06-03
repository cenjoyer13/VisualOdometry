#include "OdometryPipeline.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <utility>

#include "detectors/DetectorFactory.h"
#include "matchers/MatcherFactory.h"
#include "pose_estimators/PoseEstimatorFactory.h"
#include "scale_estimators/ScaleEstimatorFactory.h"
#include "integrators/IntegratorFactory.h"

std::unique_ptr<OdometryPipeline> OdometryPipeline::build(const OdometryConfig& config) {
    std::cout << "[OdometryPipeline] Initiating build sequence..." << std::endl;
    return std::make_unique<OdometryPipeline>(
        config,
        DetectorFactory::create(config),
        MatcherFactory::create(config),
        PoseEstimatorFactory::create(config),
        ScaleEstimatorFactory::create(config),
        IntegratorFactory::create(config)
    );
}

OdometryPipeline::OdometryPipeline(const OdometryConfig& cfg,
                                   std::unique_ptr<IFeatureDetector> d,
                                   std::unique_ptr<IFeatureMatcher> m,
                                   std::unique_ptr<IPoseEstimator> p,
                                   std::unique_ptr<IScaleEstimator> s,
                                   std::unique_ptr<ITrajectoryIntegrator> i)
    : config(cfg), 
      detector(std::move(d)), 
      matcher(std::move(m)), 
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

    // Detection.
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<cv::KeyPoint> curr_keypoints;
    DeviceBuffer curr_descriptors;
    detector->detect(frame, curr_keypoints, curr_descriptors);
    auto t1 = std::chrono::high_resolution_clock::now();
    metrics.time_detect_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Bootstrap: first frame has no predecessor to match against, so just
    // store it and return.
    if (is_first_frame) {
        prev_image = frame;
        prev_descriptors = curr_descriptors;
        prev_keypoints = curr_keypoints;
        gt_prev = current_gt;
        is_first_frame = false;

        // Debug frame for frame 0 is the raw input (no tracks to draw yet).
        debug_frame = frame.getAsCPU().clone();
        return;
    }

    // Matching.
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<cv::DMatch> good_matches = matcher->match(
        prev_descriptors, curr_descriptors,
        prev_keypoints, curr_keypoints
    );
    t1 = std::chrono::high_resolution_clock::now();
    metrics.time_match_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Build an index map for the LBA backend: matched_prev_idx[curr_idx]
    // holds the matching index in the previous frame, or -1 if unmatched.
    std::vector<int> matched_prev_idx(curr_keypoints.size(), -1);
    for (const auto& match : good_matches) {
        matched_prev_idx[match.trainIdx] = match.queryIdx;
    }

    // Lift the cv::DMatch pairs into raw 2D points for the pose estimator.
    std::vector<cv::Point2f> pts_prev;
    std::vector<cv::Point2f> pts_curr;
    pts_prev.reserve(good_matches.size());
    pts_curr.reserve(good_matches.size());

    for (const auto& match : good_matches) {
        pts_prev.push_back(prev_keypoints[match.queryIdx].pt);
        pts_curr.push_back(curr_keypoints[match.trainIdx].pt);
    }

    // Debug overlay: matched feature tracks on the input frame.
    cv::Mat color_frame;
    cv::Mat cpu_img = frame.getAsCPU();
    if (cpu_img.channels() == 1) {
        cv::cvtColor(cpu_img, color_frame, cv::COLOR_GRAY2BGR);
    } else {
        color_frame = cpu_img.clone();
    }

    // Bucketing grid overlay (only when bucketing is enabled).
    if (config.bucketing_params.enabled) {
        int cols = config.bucketing_params.grid_cols;
        int rows = config.bucketing_params.grid_rows;
        int width = color_frame.cols;
        int height = color_frame.rows;

        float cell_w = static_cast<float>(width) / cols;
        float cell_h = static_cast<float>(height) / rows;

        cv::Scalar grid_color(255, 50, 50);

        for (int i = 1; i < cols; ++i) {
            int x = static_cast<int>(i * cell_w);
            cv::line(color_frame, cv::Point(x, 0), cv::Point(x, height), grid_color, 1, cv::LINE_AA);
        }
        for (int i = 1; i < rows; ++i) {
            int y = static_cast<int>(i * cell_h);
            cv::line(color_frame, cv::Point(0, y), cv::Point(width, y), grid_color, 1, cv::LINE_AA);
        }
    }

    // Feature tracks: red line from previous to current, green dot at current.
    for (size_t i = 0; i < pts_curr.size(); i++) {
        cv::line(color_frame, pts_prev[i], pts_curr[i], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        cv::circle(color_frame, pts_curr[i], 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }

    debug_frame = color_frame;

    // Pose recovery.
    t0 = std::chrono::high_resolution_clock::now();
    cv::Mat R, t;
    bool pose_success = pose_estimator->estimatePose(pts_prev, pts_curr, config.intrinsics, R, t);
    t1 = std::chrono::high_resolution_clock::now();
    metrics.time_pose_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (pose_success) {
        // The estimator returns a zero translation when it cannot recover
        // motion (parallax below min_disparity, too few matches, or a
        // degenerate decomposition).
        const double norm_t = cv::norm(t);
        const bool stationary = (norm_t <= 1e-6);
        const int num_matches = static_cast<int>(good_matches.size());

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
            // Scale spans the whole keyframe-to-current baseline because gt_prev
            // was held across the skipped frames. A forced (still-stationary)
            // keyframe has t == 0, so it integrates nothing and its span is
            // unrecoverable, but matching is preserved.
            double scale = scale_estimator->updateScale(gt_prev, current_gt);
            integrator->integrate(R, t, scale);

            // Advance the frontend anchor (the new keyframe) before pushing to
            // LBA so the next match's indices line up with what was just pushed.
            prev_image = frame;
            prev_descriptors = curr_descriptors;
            prev_keypoints = curr_keypoints;
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

                std::vector<cv::Point2f> curr_pts;
                cv::KeyPoint::convert(curr_keypoints, curr_pts);
                new_frame.points2D = curr_pts;
                new_frame.matched_prev_idx = matched_prev_idx;
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
