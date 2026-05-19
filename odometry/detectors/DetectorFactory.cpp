#include "DetectorFactory.h"
#include "ORBDetector.h"
#include "SIFTDetector.h"
#include <iostream>

std::unique_ptr<IFeatureDetector> DetectorFactory::create(const OdometryConfig& config) {
    // Route based on the string parsed from the YAML file
    if (config.detector_type == "SIFT") {
        std::cout << "[DetectorFactory] Instantiating SIFTDetector." << std::endl;
        return std::make_unique<SIFTDetector>(config);
    }
    
    // Default fallback
    std::cout << "[DetectorFactory] Instantiating ORBDetector." << std::endl;
    return std::make_unique<ORBDetector>(config);
}
