#include "RealTime2DTrajectory.h"
#include <opencv2/imgproc.hpp>
#include <numeric>

RealTime2DTrajectory::RealTime2DTrajectory(float scale) : w(800), h(800), scale(scale) {
    traj = cv::Mat::zeros(h, w, CV_8UC3);
}

cv::Mat RealTime2DTrajectory::update(const cv::Vec3f& est_xyz, const cv::Vec3f& gt_xyz) {
    // KITTI Top-Down View: X is Right/Left, Z is Forward/Backward.
    double x = est_xyz[0];
    double z = est_xyz[2]; 
    double gt_x = gt_xyz[0];
    double gt_z = gt_xyz[2];

    // 1. Calculate live 2D MAE (Mean Absolute Error)
    double error = cv::norm(cv::Vec2d(x, z) - cv::Vec2d(gt_x, gt_z));
    errors.push_back(error);

    double sum_error = std::accumulate(errors.begin(), errors.end(), 0.0);
    double avg_error = sum_error / errors.size();

    // Offset: Centers the start point.
    int offset_x = w / 2;
    int offset_y = h / 2;

    // Draw trajectory moving "up" the screen by subtracting the Z-axis scaling
    int draw_x = static_cast<int>(x * scale) + offset_x;
    int draw_y = offset_y - static_cast<int>(z * scale);
    int true_x = static_cast<int>(gt_x * scale) + offset_x;
    int true_y = offset_y - static_cast<int>(gt_z * scale);

    // Draw Visual Odometry (Green)
    cv::circle(traj, cv::Point(draw_x, draw_y), 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    // Draw Ground Truth (Red)
    cv::circle(traj, cv::Point(true_x, true_y), 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

    // Legend and Text Background
    cv::rectangle(traj, cv::Point(10, 20), cv::Point(600, 80), cv::Scalar(0, 0, 0), -1);

    // Display the current scale on the UI
    char text[256];
    snprintf(text, sizeof(text), "AvgError: %2.4fm | Scale: %.1fx", avg_error, scale);
    cv::putText(traj, text, cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_8);

    // Legend Colors
    cv::putText(traj, "VO (Green)", cv::Point(20, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
    cv::putText(traj, "GT (Red)", cv::Point(150, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);

    return traj;
}
