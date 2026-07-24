#include "AirSimScaleEstimator.h"
#include <opencv2/core.hpp>

AirSimScaleEstimator::AirSimScaleEstimator(const OdometryConfig& cfg) : config(cfg) {}

double AirSimScaleEstimator::updateScale(const GroundTruthData& gt_prev, const GroundTruthData& gt_curr) {
    // Reads metric scale straight from ground truth: ||p_curr - p_prev||.
    // Only correct when the GT samples are aligned with the camera frames
    // being processed.
    cv::Vec3f delta = gt_curr.position - gt_prev.position;
    return cv::norm(delta);
}
