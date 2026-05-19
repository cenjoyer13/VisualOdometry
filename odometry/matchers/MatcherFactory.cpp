#include "MatcherFactory.h"
#include "KinematicMatcher.h"
#include "FLANNMatcher.h"
#include <iostream>

std::unique_ptr<IFeatureMatcher> MatcherFactory::create(const OdometryConfig& config) {
    // Route based on the string parsed from the YAML file
    if (config.matcher_type == "FLANN") {
        std::cout << "[MatcherFactory] Instantiating FLANNMatcher." << std::endl;
        return std::make_unique<FLANNMatcher>(config);
    }

    // Default fallback
    std::cout << "[MatcherFactory] Instantiating KinematicMatcher." << std::endl;
    return std::make_unique<KinematicMatcher>(config);
}
