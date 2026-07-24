#include "UnitScaleEstimator.h"

UnitScaleEstimator::UnitScaleEstimator(const OdometryConfig& cfg)
    : step(cfg.scale_estimator_unit_step) {}

double UnitScaleEstimator::updateScale(const GroundTruthData&, const GroundTruthData&) {
    return step;
}
