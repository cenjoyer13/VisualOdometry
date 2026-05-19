#include "CustomCheiralityPoseEstimator.h"
#include <opencv2/calib3d.hpp>
#include <iostream>

CustomCheiralityPoseEstimator::CustomCheiralityPoseEstimator(const OdometryConfig& cfg) : config(cfg) {}

bool CustomCheiralityPoseEstimator::estimatePose(
        const std::vector<cv::Point2f>& pts_old, 
        const std::vector<cv::Point2f>& pts_new,
        const CameraIntrinsics& intrinsics,
        cv::Mat& out_R, 
        cv::Mat& out_t) 
{
    // Fallback for empty feature sets
    if (pts_old.empty() || pts_new.empty()) {
        out_R = cv::Mat::eye(3, 3, CV_64F);
        out_t = cv::Mat::zeros(3, 1, CV_64F);
        return true; 
    }

    // 1. Config Parameters
    float min_disparity = config.pose_params.count("min_disparity") ? config.pose_params.at("min_disparity") : 2.0f;
    float min_depth = config.pose_params.count("min_depth") ? config.pose_params.at("min_depth") : 0.1f;
    float max_depth = config.pose_params.count("max_depth") ? config.pose_params.at("max_depth") : 1000.0f;
    float threshold = config.pose_params.count("threshold") ? config.pose_params.at("threshold") : 1.0f;

    // 2. Zero-Baseline / Stop Detection
    double total_disparity = 0.0;
    for (size_t i = 0; i < pts_old.size(); ++i) {
        total_disparity += cv::norm(pts_old[i] - pts_new[i]); // cv::norm replicates np.hypot
    }
    double avg_disparity = total_disparity / pts_old.size();

    if (avg_disparity < min_disparity) {
        out_R = cv::Mat::eye(3, 3, CV_64F);
        out_t = cv::Mat::zeros(3, 1, CV_64F);
        return true; 
    }

    // 3. Compute Camera Matrix K
    cv::Mat K = (cv::Mat_<double>(3, 3) << 
        intrinsics.fx, 0, intrinsics.cx, 
        0, intrinsics.fy, intrinsics.cy, 
        0, 0, 1);

    // 4. Compute Essential Matrix
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pts_old, pts_new, K, cv::RANSAC, 0.999, threshold, mask);

    if (E.empty() || E.rows < 3) {
        out_R = cv::Mat::eye(3, 3, CV_64F);
        out_t = cv::Mat::zeros(3, 1, CV_64F);
        return true;
    }

    // Python equivalent of E[0:3, :]: handle multiple concatenated 3x3 matrices
    if (E.rows > 3) {
        E = E.rowRange(0, 3).clone();
    }

    // 5. Decompose Essential Matrix
    cv::Mat R1, R2, t;
    cv::decomposeEssentialMat(E, R1, R2, t);

    // 6. Custom Cheirality Check with Strict Depth Filtering
    std::vector<cv::Mat> Rs = {R1, R1, R2, R2};
    std::vector<cv::Mat> ts = {t, -t, t, -t};

    cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
    P1 = K * P1; // Projection matrix for Camera 1 (origin)

    // Optimization: Type-cast once outside the loop 
    cv::Mat P1_32F; P1.convertTo(P1_32F, CV_32F);
    cv::Mat K_32F;  K.convertTo(K_32F, CV_32F);

    int max_valid_points = -1;
    cv::Mat best_R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat best_t = cv::Mat::zeros(3, 1, CV_64F);

    for (int i = 0; i < 4; ++i) {
        cv::Mat P2(3, 4, CV_64F);
        Rs[i].copyTo(P2(cv::Rect(0, 0, 3, 3)));
        ts[i].copyTo(P2(cv::Rect(3, 0, 1, 3)));
        P2 = K * P2;

        cv::Mat P2_32F; P2.convertTo(P2_32F, CV_32F);

        // Triangulate
        cv::Mat points_4D;
        cv::triangulatePoints(P1_32F, P2_32F, pts_old, pts_new, points_4D);
        points_4D.convertTo(points_4D, CV_64F); // Prevent gemm crash

        int valid_points = 0;

        // C++ tight loop: Equivalent to numpy vectorized boolean masking
        for (int j = 0; j < points_4D.cols; ++j) {
            // Epsilon division safety
            double w = points_4D.at<double>(3, j) + 1e-8; 

            // Un-homogenize Z coordinate in Camera 1
            double z1 = points_4D.at<double>(2, j) / w;

            // Project point to Camera 2 frame (R @ Q1 + t)
            double x1 = points_4D.at<double>(0, j) / w;
            double y1 = points_4D.at<double>(1, j) / w;
            cv::Mat pt_3d = (cv::Mat_<double>(3, 1) << x1, y1, z1);
            cv::Mat pt_cam2 = Rs[i] * pt_3d + ts[i];
            
            double z2 = pt_cam2.at<double>(2, 0);

            // Bandpass filter
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

    // Normalize translation vector
    double norm_t = cv::norm(best_t);
    if (norm_t > 1e-6) {
        best_t = best_t / norm_t;
    }

    out_R = best_R.clone();
    out_t = best_t.clone();
    return true;
}
