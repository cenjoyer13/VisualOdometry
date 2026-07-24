#include "FrontendFactory.h"
#include "DescriptorFrontend.h"
#include "OpticalFlowFrontend.h"
#include "../detectors/DetectorFactory.h"
#include "../matchers/MatcherFactory.h"
#include <iostream>

std::unique_ptr<IFrontend> FrontendFactory::create(const OdometryConfig& config) {
    if (config.frontend_type == "optical_flow") {
        std::cout << "[FrontendFactory] Instantiating OpticalFlowFrontend (Shi-Tomasi + KLT)." << std::endl;
        return std::make_unique<OpticalFlowFrontend>(config);
    }
    std::cout << "[FrontendFactory] Instantiating DescriptorFrontend." << std::endl;
    return std::make_unique<DescriptorFrontend>(
        config, DetectorFactory::create(config), MatcherFactory::create(config));
}
