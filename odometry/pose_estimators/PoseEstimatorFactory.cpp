#include "PoseEstimatorFactory.h"
#include "CustomCheiralityPoseEstimator.h"
#include "HomographyCheiralityPoseEstimator.h"
#include <iostream>

std::unique_ptr<IPoseEstimator> PoseEstimatorFactory::create(const OdometryConfig& config) {
    if (config.pose_estimator_type == "Homography") {
        std::cout << "[PoseEstimatorFactory] Instantiating HomographyCheiralityPoseEstimator." << std::endl;
        return std::make_unique<HomographyCheiralityPoseEstimator>(config);
    }

    // Default: essential-matrix estimator.
    std::cout << "[PoseEstimatorFactory] Instantiating CustomCheiralityPoseEstimator." << std::endl;
    return std::make_unique<CustomCheiralityPoseEstimator>(config);
}
