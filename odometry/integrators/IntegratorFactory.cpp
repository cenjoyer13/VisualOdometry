#include "IntegratorFactory.h"
#include "VOIntegrator.h"
#include <iostream>

std::unique_ptr<ITrajectoryIntegrator> IntegratorFactory::create(const OdometryConfig& config) {
    // If you add a sophisticated graph-optimization backend later (g2o/Ceres etc.), branch here.

    std::cout << "[IntegratorFactory] Instantiating VOIntegrator." << std::endl;
    return std::make_unique<VOIntegrator>(config);
}
