#include "MatcherFactory.h"
#include "KinematicMatcher.h"
#include "FLANNMatcher.h"
#ifdef USE_ONNX
#include "LightGlueMatcher.h"
#endif
#include <iostream>

std::unique_ptr<IFeatureMatcher> MatcherFactory::create(const OdometryConfig& config) {
#ifdef USE_ONNX
    if (config.matcher_type == "LightGlue") {
        std::cout << "[MatcherFactory] Instantiating Deep Learning LightGlueMatcher." << std::endl;
        return std::make_unique<LightGlueMatcher>(config);
    }
#else
    if (config.matcher_type == "LightGlue") {
        std::cout << "[MatcherFactory] LightGlue unavailable in CPU build (USE_ONNX=OFF); falling back to KinematicMatcher." << std::endl;
    }
#endif

    if (config.matcher_type == "FLANN") {
        std::cout << "[MatcherFactory] Instantiating FLANNMatcher." << std::endl;
        return std::make_unique<FLANNMatcher>(config);
    }

    std::cout << "[MatcherFactory] Instantiating KinematicMatcher." << std::endl;
    return std::make_unique<KinematicMatcher>(config);
}
