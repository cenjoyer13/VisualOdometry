#include "OdometryPipeline.h"
#include <iostream>

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
    // 1. Detect Features
    std::vector<cv::KeyPoint> curr_keypoints;
    DeviceBuffer curr_descriptors;
    detector->detect(frame, curr_keypoints, curr_descriptors);

    // Bootstrap check: We need two frames to compute movement
    if (is_first_frame) {
        prev_image = frame;
        prev_descriptors = curr_descriptors;
        prev_keypoints = curr_keypoints;
        gt_prev = current_gt;
        is_first_frame = false;
        return;
    }

    // 2. Match Features
    std::vector<cv::DMatch> good_matches = matcher->match(
        prev_descriptors, curr_descriptors, 
        prev_keypoints, curr_keypoints
    );

    // Translate DMatch indices to 2D geometric points for the estimator
    std::vector<cv::Point2f> pts_prev;
    std::vector<cv::Point2f> pts_curr;
    pts_prev.reserve(good_matches.size());
    pts_curr.reserve(good_matches.size());

    for (const auto& match : good_matches) {
        pts_prev.push_back(prev_keypoints[match.queryIdx].pt);
        pts_curr.push_back(curr_keypoints[match.trainIdx].pt);
    }

    // 3. Pose Recovery & Cheirality
    cv::Mat R, t;
    bool pose_success = pose_estimator->estimatePose(pts_prev, pts_curr, config.intrinsics, R, t);

    if (pose_success) {
        // 4. Calculate Absolute Scale
        double scale = scale_estimator->updateScale(gt_prev, current_gt);

        // 5. Integrate Local Step into Global Trajectory Map
        integrator->integrate(R, t, scale, current_gt.orientation);

        // 6. Update Frame State (Keyframe logic: only update anchor if successful)
        prev_image = frame;
        prev_descriptors = curr_descriptors;
        prev_keypoints = curr_keypoints;
    }

    // Always update Ground Truth time-step
    gt_prev = current_gt;
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
