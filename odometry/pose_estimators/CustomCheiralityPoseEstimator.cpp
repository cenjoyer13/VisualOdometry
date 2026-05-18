#include "CustomCheiralityPoseEstimator.h"
#include <opencv2/calib3d.hpp>
#include <numeric>
#include <iostream>

CustomCheiralityPoseEstimator::CustomCheiralityPoseEstimator(const OdometryConfig& cfg) : config(cfg) {}

bool CustomCheiralityPoseEstimator::estimatePose(
        const std::vector<cv::Point2f>& pts_old, 
        const std::vector<cv::Point2f>& pts_new,
        const CameraIntrinsics& intrinsics,
        cv::Mat& out_R, 
        cv::Mat& out_t) 
{
    if (pts_old.size() < 8 || pts_new.size() < 8) return false;

    // 1. Config Parameters
    float min_disparity = config.pose_params.count("min_disparity") ? config.pose_params.at("min_disparity") : 2.0f;
    float min_depth = config.pose_params.count("min_depth") ? config.pose_params.at("min_depth") : 0.1f;
    float max_depth = config.pose_params.count("max_depth") ? config.pose_params.at("max_depth") : 1000.0f;

    // 2. Zero-Baseline / Hover Detection
    double total_disparity = 0.0;
    for (size_t i = 0; i < pts_old.size(); ++i) {
        total_disparity += cv::norm(pts_old[i] - pts_new[i]);
    }
    double avg_disparity = total_disparity / pts_old.size();

    if (avg_disparity < min_disparity) {
        // Drone is hovering; return identity
        out_R = cv::Mat::eye(3, 3, CV_64F);
        out_t = cv::Mat::zeros(3, 1, CV_64F);
        return true; 
    }

    // 3. Compute Camera Matrix K
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        intrinsics.fx, 0, intrinsics.cx, 
        0, intrinsics.fy, intrinsics.cy, 
        0, 0, 1);

    // 4. Compute Essential Matrix using RANSAC
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pts_old, pts_new, K, cv::RANSAC, 0.999, 1.0, mask);

    if (E.empty() || E.rows != 3 || E.cols != 3) return false;

    // 5. Decompose Essential Matrix
    cv::Mat R1, R2, t;
    cv::decomposeEssentialMat(E, R1, R2, t);

    // 6. Custom Cheirality Check with Depth Bandpass
    std::vector<cv::Mat> Rs = {R1, R1, R2, R2};
    std::vector<cv::Mat> ts = {t, -t, t, -t};

    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
    P1 = K * P1; // Projection matrix for Camera 1 (origin)

    int max_valid_points = -1;
    cv::Mat best_R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat best_t = cv::Mat::zeros(3, 1, CV_64F);

    for (int i = 0; i < 4; ++i) {
        cv::Mat P2(3, 4, CV_64F);
        Rs[i].copyTo(P2(cv::Rect(0, 0, 3, 3)));
        ts[i].copyTo(P2(cv::Rect(3, 0, 1, 3)));
        P2 = K * P2;

        cv::Mat points_4D;
        cv::triangulatePoints(P1, P2, pts_old, pts_new, points_4D);

        int valid_points = 0;

        for (int j = 0; j < points_4D.cols; ++j) {
            cv::Mat col = points_4D.col(j);
            double w = col.at<double>(3);
            if (w == 0) continue;

            // Un-homogenize Z coordinate in Camera 1
            double z1 = col.at<double>(2) / w;

            // Project point to Camera 2 frame
            cv::Mat pt_3d = col.rowRange(0, 3) / w;
            cv::Mat pt_cam2 = Rs[i] * pt_3d + ts[i];
            double z2 = pt_cam2.at<double>(2);

            // Strict bandpass filter
            if (z1 > min_depth && z1 < max_depth && z2 > min_depth && z2 < max_depth) {
                valid_points++;
            }
        }

        if (valid_points > max_valid_points) {
            max_valid_points = valid_points;
            best_R = Rs[i];
            best_t = ts[i];
        }
    }

    if (max_valid_points == 0) return false;

    // Normalize translation vector
    double norm_t = cv::norm(best_t);
    if (norm_t > 1e-6) {
        best_t = best_t / norm_t;
    }

    out_R = best_R.clone();
    out_t = best_t.clone();
    return true;
}
