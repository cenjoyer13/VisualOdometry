#include "HomographyCheiralityPoseEstimator.h"
#include <opencv2/calib3d.hpp>

void HomographyCheiralityPoseEstimator::generateHypotheses(
        const std::vector<cv::Point2f>& pts_old,
        const std::vector<cv::Point2f>& pts_new,
        const cv::Mat& K,
        double ransac_threshold,
        std::vector<cv::Mat>& out_Rs,
        std::vector<cv::Mat>& out_ts,
        std::vector<uchar>& out_inliers)
{
    cv::Mat mask;
    cv::Mat H = cv::findHomography(pts_old, pts_new, cv::RANSAC, ransac_threshold, mask);

    if (H.empty() || H.rows < 3) return;

    // Up to four (R, t, n) solutions; n (the plane normal) is unused here since
    // the base disambiguates by cheirality. translations are scaled by 1/d, but
    // the base normalises the winning t so the scale drops out.
    std::vector<cv::Mat> Rs, ts, normals;
    int num = cv::decomposeHomographyMat(H, K, Rs, ts, normals);
    if (num <= 0) return;

    out_Rs = Rs;
    out_ts = ts;

    // RANSAC inlier mask: the base runs cheirality on inliers only.
    if (!mask.empty())
        out_inliers.assign(mask.data, mask.data + mask.total());
}
