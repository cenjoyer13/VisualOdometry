#include "ScaleEstimatorFactory.h"
#include "AirSimScaleEstimator.h"
#include <iostream>

std::unique_ptr<IScaleEstimator> ScaleEstimatorFactory::create(const OdometryConfig& config) {
    // Single implementation today. Branch here when a non-GT estimator
    // (e.g. optical-flow-based) is added.
    std::cout << "[ScaleEstimatorFactory] Instantiating AirSimScaleEstimator." << std::endl;
    return std::make_unique<AirSimScaleEstimator>(config);
}
