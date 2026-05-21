#include "OdometryPipeline.h"
#include <iostream>
#include <chrono> // For timing

// Domain-Specific Factories
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
    std::cout << "[OdometryPipeline] Pipeline successfully assembled and ready." << std::endl;
}

void OdometryPipeline::processFrame(DeviceBuffer& frame, const GroundTruthData& current_gt) {
    auto t_start_total = std::chrono::high_resolution_clock::now();

    // 1. Detect Features
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<cv::KeyPoint> curr_keypoints;
    DeviceBuffer curr_descriptors;
    detector->detect(frame, curr_keypoints, curr_descriptors);
    auto t1 = std::chrono::high_resolution_clock::now();
    metrics.time_detect_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Bootstrap check
    if (is_first_frame) {
        prev_image = frame;
        prev_descriptors = curr_descriptors;
        prev_keypoints = curr_keypoints;
        gt_prev = current_gt;
        is_first_frame = false;
        
        // Pass a blank image out for frame 0
        debug_frame = frame.getAsCPU().clone();
        return;
    }

    // 2. Match Features
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<cv::DMatch> good_matches = matcher->match(
        prev_descriptors, curr_descriptors, 
        prev_keypoints, curr_keypoints
    );
    t1 = std::chrono::high_resolution_clock::now();
    metrics.time_match_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Translate DMatch indices to 2D geometric points
    std::vector<cv::Point2f> pts_prev;
    std::vector<cv::Point2f> pts_curr;
    pts_prev.reserve(good_matches.size());
    pts_curr.reserve(good_matches.size());

    for (const auto& match : good_matches) {
        pts_prev.push_back(prev_keypoints[match.queryIdx].pt);
        pts_curr.push_back(curr_keypoints[match.trainIdx].pt);
    }

    // --- VISUALIZATION: Draw tracked features ---
    cv::Mat color_frame;
    cv::Mat cpu_img = frame.getAsCPU();
    if (cpu_img.channels() == 1) {
        cv::cvtColor(cpu_img, color_frame, cv::COLOR_GRAY2BGR);
    } else {
        color_frame = cpu_img.clone();
    }

    // 1. Draw the Bucketing Grid (If Enabled)
    if (config.bucketing_params.enabled) {
        int cols = config.bucketing_params.grid_cols;
        int rows = config.bucketing_params.grid_rows;
        int width = color_frame.cols;
        int height = color_frame.rows;

        float cell_w = static_cast<float>(width) / cols;
        float cell_h = static_cast<float>(height) / rows;

        // Subtle Blue color for the grid lines (OpenCV uses BGR)
        cv::Scalar grid_color(255, 50, 50); 

        // Draw vertical lines
        for (int i = 1; i < cols; ++i) {
            int x = static_cast<int>(i * cell_w);
            cv::line(color_frame, cv::Point(x, 0), cv::Point(x, height), grid_color, 1, cv::LINE_AA);
        }

        // Draw horizontal lines
        for (int i = 1; i < rows; ++i) {
            int y = static_cast<int>(i * cell_h);
            cv::line(color_frame, cv::Point(0, y), cv::Point(width, y), grid_color, 1, cv::LINE_AA);
        }
    }

    // 2. Draw the Feature Tracks
    for (size_t i = 0; i < pts_curr.size(); i++) {
        // Red line from old to new, Green dot at current position
        cv::line(color_frame, pts_prev[i], pts_curr[i], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        cv::circle(color_frame, pts_curr[i], 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
    }
    
    debug_frame = color_frame; // Save to be fetched by main

    // 3. Pose Recovery
    t0 = std::chrono::high_resolution_clock::now();
    cv::Mat R, t;
    bool pose_success = pose_estimator->estimatePose(pts_prev, pts_curr, config.intrinsics, R, t);
    t1 = std::chrono::high_resolution_clock::now();
    metrics.time_pose_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (pose_success) {
        double scale = scale_estimator->updateScale(gt_prev, current_gt);
        integrator->integrate(R, t, scale, current_gt.orientation);

        // Only update the anchor if we physically moved, or if we lost tracking
        double norm_t = cv::norm(t);
        if (norm_t > 1e-6 || pts_curr.size() < 8) {
            prev_image = frame;
            prev_descriptors = curr_descriptors;
            prev_keypoints = curr_keypoints;
        }
    }

    gt_prev = current_gt;

    // Total Time & FPS
    auto t_end_total = std::chrono::high_resolution_clock::now();
    metrics.time_total_ms = std::chrono::duration<double, std::milli>(t_end_total - t_start_total).count();
    metrics.fps = 1000.0 / metrics.time_total_ms;
}

cv::Mat OdometryPipeline::getGlobalTransformVO() const {
    return integrator->getGlobalTransformVO();
}

cv::Mat OdometryPipeline::getGlobalTransformVIO() const {
    return integrator->getGlobalTransformVIO();
}

bool OdometryPipeline::isTrackingActive() const {
    // Determine active tracking state based on non-identity VO matrices
    // or you could expose a specific flag from the integrator if desired.
    return !is_first_frame; 
}
