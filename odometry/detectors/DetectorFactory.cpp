#include "DetectorFactory.h"
#include "ORBDetector.h"
#include "SIFTDetector.h"
#ifdef USE_ONNX
#include "SuperPointDetector.h"
#endif
#include <iostream>

std::unique_ptr<IFeatureDetector> DetectorFactory::create(const OdometryConfig& config) {
    // Selection: config.detector_type comes from the YAML "detector_type" field.
    if (config.detector_type == "SIFT") {
        std::cout << "[DetectorFactory] Instantiating SIFTDetector." << std::endl;
        return std::make_unique<SIFTDetector>(config);
    }

#ifdef USE_ONNX
    if (config.detector_type == "SuperPoint") {
        std::cout << "[DetectorFactory] Instantiating SuperPointDetector." << std::endl;
        return std::make_unique<SuperPointDetector>(config);
    }
#else
    if (config.detector_type == "SuperPoint") {
        std::cout << "[DetectorFactory] SuperPoint unavailable in CPU build (USE_ONNX=OFF); falling back to ORB." << std::endl;
    }
#endif

    // Fallback: ORB.
    std::cout << "[DetectorFactory] Instantiating ORBDetector." << std::endl;
    return std::make_unique<ORBDetector>(config);
}
