#include "CustomCheiralityPoseEstimator.h"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <mutex>

CustomCheiralityPoseEstimator::CustomCheiralityPoseEstimator(const OdometryConfig& cfg) : config(cfg) {}

bool CustomCheiralityPoseEstimator::estimatePose(
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

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pts_old, pts_new, K, cv::RANSAC, 0.999, threshold, mask);

    if (E.empty() || E.rows < 3) {
        out_R = cv::Mat::eye(3, 3, CV_64F);
        out_t = cv::Mat::zeros(3, 1, CV_64F);
        return true;
    }

    if (E.rows > 3) E = E.rowRange(0, 3).clone();

    cv::Mat R1, R2, t;
    cv::decomposeEssentialMat(E, R1, R2, t);

    // Four hypotheses from the essential-matrix decomposition; pick the one
    // that places the most triangulated points in front of both cameras.
    std::vector<cv::Mat> Rs = {R1, R1, R2, R2};
    std::vector<cv::Mat> ts = {t, -t, t, -t};

    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
    P1 = K * P1;

    cv::Mat P1_32F; P1.convertTo(P1_32F, CV_32F);
    cv::Mat K_32F;  K.convertTo(K_32F, CV_32F);

    int max_valid_points = -1;
    cv::Mat best_R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat best_t = cv::Mat::zeros(3, 1, CV_64F);
    std::mutex best_mutex;

    // Run the four hypotheses in parallel; each computes its own valid count.
    cv::setNumThreads(config.num_threads);
    cv::parallel_for_(cv::Range(0, 4), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {

            cv::Mat P2(3, 4, CV_64F);
            Rs[i].copyTo(P2(cv::Rect(0, 0, 3, 3)));
            ts[i].copyTo(P2(cv::Rect(3, 0, 1, 3)));
            P2 = K * P2;

            cv::Mat P2_32F; P2.convertTo(P2_32F, CV_32F);

            cv::Mat points_4D;
            cv::triangulatePoints(P1_32F, P2_32F, pts_old, pts_new, points_4D);
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

    double norm_t = cv::norm(best_t);
    if (norm_t > 1e-6) best_t = best_t / norm_t;

    out_R = best_R.clone();
    out_t = best_t.clone();
    return true;
}
