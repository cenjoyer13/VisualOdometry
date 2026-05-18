#include "ScaleEstimatorFactory.h"
#include "AirSimScaleEstimator.h"
#include <iostream>

std::unique_ptr<IScaleEstimator> ScaleEstimatorFactory::create(const OdometryConfig& config) {
    // If you add an Optical Flow scale estimator later, branch here.
    
    std::cout << "[ScaleEstimatorFactory] Instantiating AirSimScaleEstimator." << std::endl;
    return std::make_unique<AirSimScaleEstimator>(config);
}
