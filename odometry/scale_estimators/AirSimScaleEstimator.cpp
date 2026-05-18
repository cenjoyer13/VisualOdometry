#include "AirSimScaleEstimator.h"
#include <opencv2/core.hpp>

AirSimScaleEstimator::AirSimScaleEstimator(const OdometryConfig& cfg) : config(cfg) {}

double AirSimScaleEstimator::updateScale(const GroundTruthData& gt_prev, const GroundTruthData& gt_curr) {
    // 1. Calculate the difference vector between current and previous GT positions
    cv::Vec3f delta = gt_curr.position - gt_prev.position;
    
    // 2. cv::norm computes the exact Euclidean magnitude: sqrt(x^2 + y^2 + z^2)
    double scale = cv::norm(delta);
    
    return scale;
}
