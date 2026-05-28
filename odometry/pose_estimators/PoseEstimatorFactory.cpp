#include "PoseEstimatorFactory.h"
#include "CustomCheiralityPoseEstimator.h"
#include <iostream>

std::unique_ptr<IPoseEstimator> PoseEstimatorFactory::create(const OdometryConfig& config) {
    // Single implementation today. Branch on an algorithm-type field here
    // when a second estimator (e.g. OpenCV's 5-point) is added.
    std::cout << "[PoseEstimatorFactory] Instantiating CustomCheiralityPoseEstimator." << std::endl;
    return std::make_unique<CustomCheiralityPoseEstimator>(config);
}
