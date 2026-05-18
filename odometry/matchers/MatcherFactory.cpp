#include "MatcherFactory.h"
#include "KinematicMatcher.h"
#include <iostream>

std::unique_ptr<IFeatureMatcher> MatcherFactory::create(const OdometryConfig& config) {
    // Check config for algorithm type (e.g., if config.matcher_params["type"] == "LightGlue")
    // For now, we default to the Kinematic Matcher.
    
    std::cout << "[MatcherFactory] Instantiating KinematicMatcher. Hardware routing deferred to class." << std::endl;
    return std::make_unique<KinematicMatcher>(config);
}
