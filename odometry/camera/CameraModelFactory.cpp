#include "CameraModelFactory.h"
#include "KannalaBrandtCamera.h"
#include "PinholeCamera.h"
#include <iostream>

std::unique_ptr<ICameraModel> CameraModelFactory::create(const CameraModelConfig& cfg) {
    if (cfg.model_type == "KannalaBrandtCamera") {
        std::cout << "[CameraModelFactory] Instantiating KannalaBrandtCamera.\n";
        return std::make_unique<KannalaBrandtCamera>(cfg);
    }
    if (cfg.model_type == "Pinhole") {
        std::cout << "[CameraModelFactory] Instantiating PinholeCamera.\n";
        return std::make_unique<PinholeCamera>(cfg);
    }
    if (!cfg.model_type.empty()) {
        std::cerr << "[CameraModelFactory] Unknown camera model_type: "
                  << cfg.model_type << "\n";
    }
    return nullptr;
}
