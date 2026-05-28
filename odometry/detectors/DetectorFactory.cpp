#include "DetectorFactory.h"
#include "ORBDetector.h"
#include "SIFTDetector.h"
#include "SuperPointDetector.h"
#include <iostream>

std::unique_ptr<IFeatureDetector> DetectorFactory::create(const OdometryConfig& config) {
    // Selection: config.detector_type comes from the YAML "detector_type" field.
    if (config.detector_type == "SIFT") {
        std::cout << "[DetectorFactory] Instantiating SIFTDetector." << std::endl;
        return std::make_unique<SIFTDetector>(config);
    }

    if (config.detector_type == "SuperPoint") {
        std::cout << "[DetectorFactory] Instantiating SuperPointDetector." << std::endl;
        return std::make_unique<SuperPointDetector>(config);
    }

    // Fallback: ORB.
    std::cout << "[DetectorFactory] Instantiating ORBDetector." << std::endl;
    return std::make_unique<ORBDetector>(config);
}
