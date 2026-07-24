#pragma once
#include "CheiralityPoseEstimator.h"

// Essential-matrix estimator: RANSAC findEssentialMat + decomposeEssentialMat
// yields four (R, t) hypotheses, disambiguated by the base cheirality check.
class CustomCheiralityPoseEstimator : public CheiralityPoseEstimator {
public:
    explicit CustomCheiralityPoseEstimator(const OdometryConfig& cfg)
        : CheiralityPoseEstimator(cfg) {}
    ~CustomCheiralityPoseEstimator() override = default;

protected:
    void generateHypotheses(const std::vector<cv::Point2f>& pts_old,
                            const std::vector<cv::Point2f>& pts_new,
                            const cv::Mat& K,
                            double ransac_threshold,
                            std::vector<cv::Mat>& out_Rs,
                            std::vector<cv::Mat>& out_ts,
                            std::vector<uchar>& out_inliers) override;
};
