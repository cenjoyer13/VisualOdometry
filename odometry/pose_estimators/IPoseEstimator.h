#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include "../OdometryTypes.h"

class IPoseEstimator {
public:
    virtual ~IPoseEstimator() = default;

    // Takes pre-matched 2D point correspondences so the estimator stays
    // independent of cv::DMatch shape. Returns true on success (RANSAC found
    // an essential matrix and cheirality picked a usable (R, t) hypothesis).
    // Convention: out_R, out_t are T_curr_prev (see CLAUDE.md in this folder).
    virtual bool estimatePose(const std::vector<cv::Point2f>& pts_old,
                              const std::vector<cv::Point2f>& pts_new,
                              const CameraIntrinsics& intrinsics,
                              cv::Mat& out_R,
                              cv::Mat& out_t) = 0;
};
