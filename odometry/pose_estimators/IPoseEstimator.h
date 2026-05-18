#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "OdometryTypes.h"

class IPoseEstimator {
public:
    virtual ~IPoseEstimator() = default;

    // Receives ONLY the successfully matched 2D points to decouple it from DMatch logic.
    // Returns true if pose recovery succeeded (i.e. didn't fail cheirality or RANSAC).
    virtual bool estimatePose(const std::vector<cv::Point2f>& pts_old, 
                              const std::vector<cv::Point2f>& pts_new,
                              const CameraIntrinsics& intrinsics,
                              cv::Mat& out_R, 
                              cv::Mat& out_t) = 0;
};
