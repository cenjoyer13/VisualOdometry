#include "MatcherFactory.h"
#include "KinematicMatcher.h"
#include "FLANNMatcher.h"
#include "LightGlueMatcher.h"
#include <iostream>

std::unique_ptr<IFeatureMatcher> MatcherFactory::create(const OdometryConfig& config) {
    if (config.matcher_type == "LightGlue") {
        std::cout << "[MatcherFactory] Instantiating Deep Learning LightGlueMatcher." << std::endl;
        return std::make_unique<LightGlueMatcher>(config);
    }
    
    if (config.matcher_type == "FLANN") {
        std::cout << "[MatcherFactory] Instantiating FLANNMatcher." << std::endl;
        return std::make_unique<FLANNMatcher>(config);
    }

    std::cout << "[MatcherFactory] Instantiating KinematicMatcher." << std::endl;
    return std::make_unique<KinematicMatcher>(config);
}
