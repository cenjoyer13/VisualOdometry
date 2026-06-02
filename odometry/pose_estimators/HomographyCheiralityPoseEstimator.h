#pragma once
#include "CheiralityPoseEstimator.h"

// Homography estimator: RANSAC findHomography + decomposeHomographyMat yields
// up to four (R, t) hypotheses, disambiguated by the base cheirality check.
// Suited to planar / low-parallax scenes where the essential matrix degenerates.
class HomographyCheiralityPoseEstimator : public CheiralityPoseEstimator {
public:
    explicit HomographyCheiralityPoseEstimator(const OdometryConfig& cfg)
        : CheiralityPoseEstimator(cfg) {}
    ~HomographyCheiralityPoseEstimator() override = default;

protected:
    void generateHypotheses(const std::vector<cv::Point2f>& pts_old,
                            const std::vector<cv::Point2f>& pts_new,
                            const cv::Mat& K,
                            double ransac_threshold,
                            std::vector<cv::Mat>& out_Rs,
                            std::vector<cv::Mat>& out_ts,
                            std::vector<uchar>& out_inliers) override;
};
