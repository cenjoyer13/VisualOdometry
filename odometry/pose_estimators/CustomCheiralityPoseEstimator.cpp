#include "CustomCheiralityPoseEstimator.h"
#include <opencv2/calib3d.hpp>

void CustomCheiralityPoseEstimator::generateHypotheses(
        const std::vector<cv::Point2f>& pts_old,
        const std::vector<cv::Point2f>& pts_new,
        const cv::Mat& K,
        double ransac_threshold,
        std::vector<cv::Mat>& out_Rs,
        std::vector<cv::Mat>& out_ts,
        std::vector<uchar>& out_inliers)
{
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pts_old, pts_new, K, cv::RANSAC, 0.999, ransac_threshold, mask);

    if (E.empty() || E.rows < 3) return;
    if (E.rows > 3) E = E.rowRange(0, 3).clone();

    cv::Mat R1, R2, t;
    cv::decomposeEssentialMat(E, R1, R2, t);

    // Four hypotheses from the essential-matrix decomposition; the base picks
    // the one that places the most triangulated points in front of both cameras.
    out_Rs = {R1, R1, R2, R2};
    out_ts = {t, -t, t, -t};

    // RANSAC inlier mask: the base runs cheirality on inliers only.
    if (!mask.empty())
        out_inliers.assign(mask.data, mask.data + mask.total());
}
