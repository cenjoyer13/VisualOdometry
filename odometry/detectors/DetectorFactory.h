#pragma once
#include <memory>
#include "IFeatureDetector.h"
#include "../OdometryTypes.h"

class DetectorFactory {
public:
    static std::unique_ptr<IFeatureDetector> create(const OdometryConfig& config);
};
