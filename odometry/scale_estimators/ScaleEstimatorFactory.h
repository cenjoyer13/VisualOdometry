#pragma once
#include <memory>
#include "IScaleEstimator.h"
#include "../OdometryTypes.h"

class ScaleEstimatorFactory {
public:
    static std::unique_ptr<IScaleEstimator> create(const OdometryConfig& config);
};
