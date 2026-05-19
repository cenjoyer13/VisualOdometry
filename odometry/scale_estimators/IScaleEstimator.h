#pragma once
#include "../OdometryTypes.h"

class IScaleEstimator {
public:
    virtual ~IScaleEstimator() = default;

    // Calculates scale dynamically based on the delta between frames
    virtual double updateScale(const GroundTruthData& gt_prev, 
                               const GroundTruthData& gt_curr) = 0;
};
