#pragma once
#include <memory>
#include "IFeatureMatcher.h"
#include "../OdometryTypes.h"

class MatcherFactory {
public:
    static std::unique_ptr<IFeatureMatcher> create(const OdometryConfig& config);
};
