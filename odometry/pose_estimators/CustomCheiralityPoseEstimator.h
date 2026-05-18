#pragma once
#include "IPoseEstimator.h"
#include "../OdometryTypes.h"

class CustomCheiralityPoseEstimator : public IPoseEstimator {
private:
    OdometryConfig config;

public:
    explicit CustomCheiralityPoseEstimator(const OdometryConfig& cfg);
    ~CustomCheiralityPoseEstimator() override = default;

    bool estimatePose(const std::vector<cv::Point2f>& pts_old, 
                      const std::vector<cv::Point2f>& pts_new,
                      const CameraIntrinsics& intrinsics,
                      cv::Mat& out_R, 
                      cv::Mat& out_t) override;
};
