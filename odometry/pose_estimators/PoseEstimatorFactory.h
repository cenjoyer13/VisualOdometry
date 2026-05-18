#pragma once
#include <memory>
#include "IPoseEstimator.h"
#include "../OdometryTypes.h"

class PoseEstimatorFactory {
public:
    static std::unique_ptr<IPoseEstimator> create(const OdometryConfig& config);
};
