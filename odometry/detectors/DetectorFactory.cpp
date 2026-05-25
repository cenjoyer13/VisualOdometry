#include "DetectorFactory.h"
#include "ORBDetector.h"
#include "SIFTDetector.h"
#include "SuperPointDetector.h"
#include "ALIKEDDetector.h"
#include <iostream>

std::unique_ptr<IFeatureDetector> DetectorFactory::create(const OdometryConfig& config) {
    // Route based on the string parsed from the YAML file
    if (config.detector_type == "SIFT") {
        std::cout << "[DetectorFactory] Instantiating SIFTDetector." << std::endl;
        return std::make_unique<SIFTDetector>(config);
    }
    
    if (config.detector_type == "SuperPoint") {
        std::cout << "[DetectorFactory] Instantiating SuperPointDetector." << std::endl;
        return std::make_unique<SuperPointDetector>(config);
    }
    
    else if (config.detector_type == "ALIKED") { 
        std::cout << "[DetectorFactory] Instantiating ALIKEDDetector via ONNX." << std::endl;
        return std::make_unique<ALIKEDDetector>(config);
    }
    
    // Default fallback
    std::cout << "[DetectorFactory] Instantiating ORBDetector." << std::endl;
    return std::make_unique<ORBDetector>(config);
}
