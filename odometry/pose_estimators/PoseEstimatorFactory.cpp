#include "PoseEstimatorFactory.h"
#include "CustomCheiralityPoseEstimator.h"
#include <iostream>

std::unique_ptr<IPoseEstimator> PoseEstimatorFactory::create(const OdometryConfig& config) {
    // If you add other estimators (like standard OpenCV 5-point algorithm), branch here based on algorithm type.
    
    std::cout << "[PoseEstimatorFactory] Instantiating CustomCheiralityPoseEstimator." << std::endl;
    return std::make_unique<CustomCheiralityPoseEstimator>(config);
}
