#include "IntegratorFactory.h"
#include "DualPathIntegrator.h"
#include <iostream>

std::unique_ptr<ITrajectoryIntegrator> IntegratorFactory::create(const OdometryConfig& config) {
    // If you add a sophisticated graph-optimization backend later (like g2o/Ceres), branch here.
    
    std::cout << "[IntegratorFactory] Instantiating DualPathIntegrator." << std::endl;
    return std::make_unique<DualPathIntegrator>(config);
}
