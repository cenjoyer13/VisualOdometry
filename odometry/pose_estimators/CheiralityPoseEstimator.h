#pragma once
#include "IPoseEstimator.h"
#include "../OdometryTypes.h"
#include <vector>

// Shared skeleton for pose estimators that recover (R, t) by disambiguating a
// set of candidate hypotheses with a triangulation cheirality check: the
// hypothesis placing the most triangulated points in front of both cameras
// wins. Subclasses supply only how the hypotheses are generated (essential vs
// homography decomposition) via generateHypotheses; everything else (point/
// disparity guards, intrinsics, the parallel cheirality count, translation
// normalisation, and the return convention) lives here so both estimators
// stay byte-for-byte identical in that logic.
//
// Return convention is unchanged from the original single estimator: out_R,
// out_t are T_curr_prev. See CLAUDE.md in this folder before touching it.
class CheiralityPoseEstimator : public IPoseEstimator {
public:
    explicit CheiralityPoseEstimator(const OdometryConfig& cfg) : config(cfg) {}
    ~CheiralityPoseEstimator() override = default;

    bool estimatePose(const std::vector<cv::Point2f>& pts_old,
                      const std::vector<cv::Point2f>& pts_new,
                      const CameraIntrinsics& intrinsics,
                      cv::Mat& out_R,
                      cv::Mat& out_t) override;

protected:
    // Produce candidate (R, t) hypotheses from the matched points. out_Rs and
    // out_ts are parallel; leaving them empty signals "no motion recoverable"
    // and the base returns identity. ransac_threshold is the pixel RANSAC
    // threshold pulled from pose_params. out_inliers is the per-correspondence
    // RANSAC inlier mask (same length as pts_old); the base runs the cheirality
    // check on inliers only. Leave it empty to fall back to all correspondences.
    virtual void generateHypotheses(const std::vector<cv::Point2f>& pts_old,
                                    const std::vector<cv::Point2f>& pts_new,
                                    const cv::Mat& K,
                                    double ransac_threshold,
                                    std::vector<cv::Mat>& out_Rs,
                                    std::vector<cv::Mat>& out_ts,
                                    std::vector<uchar>& out_inliers) = 0;

    OdometryConfig config;
};
