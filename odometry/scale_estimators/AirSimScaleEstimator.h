#pragma once
#include "IScaleEstimator.h"
#include "../OdometryTypes.h"

class AirSimScaleEstimator : public IScaleEstimator {
private:
    OdometryConfig config;

public:
    explicit AirSimScaleEstimator(const OdometryConfig& cfg);
    ~AirSimScaleEstimator() override = default;

    double updateScale(const GroundTruthData& gt_prev, const GroundTruthData& gt_curr) override;
};
