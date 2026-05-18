#include "DetectorFactory.h"
#include "ORBDetector.h"
#include <iostream>

std::unique_ptr<IFeatureDetector> DetectorFactory::create(const OdometryConfig& config) {
    // If you add other detectors like SIFT or SuperPoint later, you branch here 
    // based on config.detector_params["algorithm"] or similar logic.
    
    std::cout << "[DetectorFactory] Instantiating ORBDetector. Hardware routing deferred to class." << std::endl;
    return std::make_unique<ORBDetector>(config);
}
