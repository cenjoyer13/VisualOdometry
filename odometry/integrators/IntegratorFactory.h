#pragma once
#include <memory>
#include "ITrajectoryIntegrator.h"
#include "../OdometryTypes.h"

class IntegratorFactory {
public:
    static std::unique_ptr<ITrajectoryIntegrator> create(const OdometryConfig& config);
};
