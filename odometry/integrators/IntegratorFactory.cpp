#include "IntegratorFactory.h"
#include "VOIntegrator.h"
#include <iostream>

std::unique_ptr<ITrajectoryIntegrator> IntegratorFactory::create(const OdometryConfig& config) {
    // Single implementation today. Branch here when a graph-optimization
    // backend (g2o, Ceres) is wired in alongside the windowed LBA.
    std::cout << "[IntegratorFactory] Instantiating VOIntegrator." << std::endl;
    return std::make_unique<VOIntegrator>(config);
}
