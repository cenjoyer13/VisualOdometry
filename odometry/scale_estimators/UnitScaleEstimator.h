#pragma once
#include "IScaleEstimator.h"
#include "../OdometryTypes.h"

// For sources with no ground truth at all (e.g. OdomLogEvaluator): returns a
// fixed, unitless step per keyframe instead of a metric distance, so a
// trajectory can still be integrated and its *shape* inspected. Values are
// not metric — do not compare against real distances.
class UnitScaleEstimator : public IScaleEstimator {
private:
    double step;

public:
    explicit UnitScaleEstimator(const OdometryConfig& cfg);
    ~UnitScaleEstimator() override = default;

    double updateScale(const GroundTruthData& gt_prev, const GroundTruthData& gt_curr) override;
};
