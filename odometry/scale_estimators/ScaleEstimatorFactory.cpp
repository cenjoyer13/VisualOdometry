#include "ScaleEstimatorFactory.h"
#include "AirSimScaleEstimator.h"
#include "UnitScaleEstimator.h"
#include <iostream>

std::unique_ptr<IScaleEstimator> ScaleEstimatorFactory::create(const OdometryConfig& config) {
    if (config.scale_estimator_type == "Unit") {
        std::cout << "[ScaleEstimatorFactory] Instantiating UnitScaleEstimator (no ground truth)." << std::endl;
        return std::make_unique<UnitScaleEstimator>(config);
    }
    std::cout << "[ScaleEstimatorFactory] Instantiating AirSimScaleEstimator." << std::endl;
    return std::make_unique<AirSimScaleEstimator>(config);
}
