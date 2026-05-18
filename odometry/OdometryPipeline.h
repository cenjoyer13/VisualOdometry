#pragma once
#include <memory>
#include <vector>
#include <opencv2/core.hpp>

#include "OdometryTypes.h"
#include "detectors/IFeatureDetector.h"
#include "matchers/IFeatureMatcher.h"
#include "pose_estimators/IPoseEstimator.h"
#include "scale_estimators/IScaleEstimator.h"
#include "integrators/ITrajectoryIntegrator.h"

class OdometryPipeline {
private:
    // Core Configuration
    OdometryConfig config;

    // State Memory (The "Previous Frame")
    bool is_first_frame;
    DeviceBuffer prev_image;
    DeviceBuffer prev_descriptors;
    std::vector<cv::KeyPoint> prev_keypoints;
    GroundTruthData gt_prev;

    // The decoupled algorithm interfaces
    std::unique_ptr<IFeatureDetector> detector;
    std::unique_ptr<IFeatureMatcher> matcher;
    std::unique_ptr<IPoseEstimator> pose_estimator;
    std::unique_ptr<IScaleEstimator> scale_estimator;
    std::unique_ptr<ITrajectoryIntegrator> integrator;

public:
    // Static Builder: Assembles the pipeline via domain-specific factories
    static std::unique_ptr<OdometryPipeline> build(const OdometryConfig& config);

    // Constructor for dependency injection
    OdometryPipeline(const OdometryConfig& cfg,
                     std::unique_ptr<IFeatureDetector> d,
                     std::unique_ptr<IFeatureMatcher> m,
                     std::unique_ptr<IPoseEstimator> p,
                     std::unique_ptr<IScaleEstimator> s,
                     std::unique_ptr<ITrajectoryIntegrator> i);

    // Main execution loop called by the worker thread
    void processFrame(DeviceBuffer& frame, const GroundTruthData& current_gt);

    // Trajectory Accessors
    cv::Mat getGlobalTransformVO() const;
    cv::Mat getGlobalTransformVIO() const;
    
    // Status flag
    bool isTrackingActive() const;
};
