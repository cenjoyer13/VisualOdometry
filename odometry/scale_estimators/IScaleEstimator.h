#pragma once
#include "../OdometryTypes.h"

class IScaleEstimator {
public:
    virtual ~IScaleEstimator() = default;

    // Returns a metric scale for the inter-frame translation. Inputs are
    // sequential frame samples; the concrete estimator decides how to use them.
    virtual double updateScale(const GroundTruthData& gt_prev,
                               const GroundTruthData& gt_curr) = 0;
};
