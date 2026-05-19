#pragma once
#include <memory>
#include <vector>
#include <opencv2/core.hpp>
// Include highgui for colors (if not already included)
#include <opencv2/imgproc.hpp> 

#include "OdometryTypes.h"
#include "detectors/IFeatureDetector.h"
#include "matchers/IFeatureMatcher.h"
#include "pose_estimators/IPoseEstimator.h"
#include "scale_estimators/IScaleEstimator.h"
#include "integrators/ITrajectoryIntegrator.h"

class OdometryPipeline {
private:
    OdometryConfig config;

    bool is_first_frame;
    DeviceBuffer prev_image;
    DeviceBuffer prev_descriptors;
    std::vector<cv::KeyPoint> prev_keypoints;
    GroundTruthData gt_prev;

    // --- NEW: Logging & Visualization ---
    PipelineMetrics metrics;
    cv::Mat debug_frame;

    std::unique_ptr<IFeatureDetector> detector;
    std::unique_ptr<IFeatureMatcher> matcher;
    std::unique_ptr<IPoseEstimator> pose_estimator;
    std::unique_ptr<IScaleEstimator> scale_estimator;
    std::unique_ptr<ITrajectoryIntegrator> integrator;

public:
    static std::unique_ptr<OdometryPipeline> build(const OdometryConfig& config);

    OdometryPipeline(const OdometryConfig& cfg,
                     std::unique_ptr<IFeatureDetector> d,
                     std::unique_ptr<IFeatureMatcher> m,
                     std::unique_ptr<IPoseEstimator> p,
                     std::unique_ptr<IScaleEstimator> s,
                     std::unique_ptr<ITrajectoryIntegrator> i);

    void processFrame(DeviceBuffer& frame, const GroundTruthData& current_gt);

    cv::Mat getGlobalTransformVO() const;
    cv::Mat getGlobalTransformVIO() const;
    bool isTrackingActive() const;

    // --- NEW: Accessors ---
    const PipelineMetrics& getMetrics() const { return metrics; }
    cv::Mat getDebugFrame() const { return debug_frame; }
};
