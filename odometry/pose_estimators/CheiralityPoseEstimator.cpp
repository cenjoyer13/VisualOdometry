#include "CheiralityPoseEstimator.h"
#include <opencv2/calib3d.hpp>
#include <mutex>

bool CheiralityPoseEstimator::estimatePose(
        const std::vector<cv::Point2f>& pts_old,
        const std::vector<cv::Point2f>& pts_new,
        const CameraIntrinsics& intrinsics,
        cv::Mat& out_R,
        cv::Mat& out_t)
{
    if (pts_old.size() < 8 || pts_new.size() < 8) {
        out_R = cv::Mat::eye(3, 3, CV_64F);
        out_t = cv::Mat::zeros(3, 1, CV_64F);
        return true;
    }

    float min_disparity = config.pose_params.count("min_disparity") ? config.pose_params.at("min_disparity") : 2.0f;
    float min_depth = config.pose_params.count("min_depth") ? config.pose_params.at("min_depth") : 0.1f;
    float max_depth = config.pose_params.count("max_depth") ? config.pose_params.at("max_depth") : 1000.0f;
    float threshold = config.pose_params.count("threshold") ? config.pose_params.at("threshold") : 1.0f;

    double total_disparity = 0.0;
    for (size_t i = 0; i < pts_old.size(); ++i) {
        total_disparity += cv::norm(pts_old[i] - pts_new[i]);
    }
    double avg_disparity = total_disparity / pts_old.size();

    if (avg_disparity < min_disparity) {
        out_R = cv::Mat::eye(3, 3, CV_64F);
        out_t = cv::Mat::zeros(3, 1, CV_64F);
        return true;
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        intrinsics.fx, 0, intrinsics.cx,
        0, intrinsics.fy, intrinsics.cy,
        0, 0, 1);

    // Subclass hook: essential- or homography-based candidate generation.
    std::vector<cv::Mat> Rs, ts;
    std::vector<uchar> inliers;
    generateHypotheses(pts_old, pts_new, K, threshold, Rs, ts, inliers);

    if (Rs.empty() || Rs.size() != ts.size()) {
        out_R = cv::Mat::eye(3, 3, CV_64F);
        out_t = cv::Mat::zeros(3, 1, CV_64F);
        return true;
    }

    // Cheirality check runs on RANSAC inliers only: outliers can flip the
    // positive-depth vote and select the wrong decomposition. Fall back to all
    // correspondences if the subclass supplied no mask.
    std::vector<cv::Point2f> in_old, in_new;
    if (inliers.size() == pts_old.size()) {
        in_old.reserve(inliers.size());
        in_new.reserve(inliers.size());
        for (size_t i = 0; i < inliers.size(); ++i) {
            if (inliers[i]) {
                in_old.push_back(pts_old[i]);
                in_new.push_back(pts_new[i]);
            }
        }
    } else {
        in_old = pts_old;
        in_new = pts_new;
    }

    if (in_old.size() < 4) {
        out_R = cv::Mat::eye(3, 3, CV_64F);
        out_t = cv::Mat::zeros(3, 1, CV_64F);
        return true;
    }

    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
    P1 = K * P1;

    cv::Mat P1_32F; P1.convertTo(P1_32F, CV_32F);

    int max_valid_points = -1;
    cv::Mat best_R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat best_t = cv::Mat::zeros(3, 1, CV_64F);
    std::mutex best_mutex;

    // Run the hypotheses in parallel; each computes its own valid count.
    cv::setNumThreads(config.num_threads);
    cv::parallel_for_(cv::Range(0, static_cast<int>(Rs.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {

            cv::Mat P2(3, 4, CV_64F);
            Rs[i].copyTo(P2(cv::Rect(0, 0, 3, 3)));
            ts[i].copyTo(P2(cv::Rect(3, 0, 1, 3)));
            P2 = K * P2;

            cv::Mat P2_32F; P2.convertTo(P2_32F, CV_32F);

            cv::Mat points_4D;
            cv::triangulatePoints(P1_32F, P2_32F, in_old, in_new, points_4D);
            points_4D.convertTo(points_4D, CV_64F);

            int valid_points = 0;

            // Hoist the bottom row of R and the z component of t out of the
            // inner loop so the hot path avoids cv::Mat indexing overhead.
            double r20 = Rs[i].at<double>(2, 0);
            double r21 = Rs[i].at<double>(2, 1);
            double r22 = Rs[i].at<double>(2, 2);
            double tz  = ts[i].at<double>(2, 0);

            for (int j = 0; j < points_4D.cols; ++j) {
                double w = points_4D.at<double>(3, j) + 1e-8;
                double z1 = points_4D.at<double>(2, j) / w;

                // z2 directly from the third row of [R|t] applied to the
                // dehomogenised point; avoids allocating a per-point 3x1.
                double x1 = points_4D.at<double>(0, j) / w;
                double y1 = points_4D.at<double>(1, j) / w;
                double z2 = (r20 * x1) + (r21 * y1) + (r22 * z1) + tz;

                if (z1 > min_depth && z1 < max_depth && z2 > min_depth && z2 < max_depth) {
                    valid_points++;
                }
            }

            std::lock_guard<std::mutex> lock(best_mutex);
            if (valid_points > max_valid_points) {
                max_valid_points = valid_points;
                best_R = Rs[i];
                best_t = ts[i];
            }
        }
    });

    // Return the raw winning translation, NOT normalized: the essential
    // decomposition already yields a unit t, while the homography yields t/d
    // (translation over plane distance). The pipeline normalizes to a unit
    // direction unless it is applying altimeter scale to the homography t/d
    // (scale = AGL = d, giving metric t). Stationary/degenerate frames still
    // return exactly zero from the early-return guards above.
    out_R = best_R.clone();
    out_t = best_t.clone();
    return true;
}
