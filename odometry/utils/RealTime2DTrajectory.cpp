#include "RealTime2DTrajectory.h"
#include <opencv2/imgproc.hpp>
#include <numeric>

RealTime2DTrajectory::RealTime2DTrajectory(float scale) : w(800), h(800), scale(scale) {
    traj = cv::Mat::zeros(h, w, CV_8UC3);
}

cv::Mat RealTime2DTrajectory::update(const cv::Vec3f& est_xyz, const cv::Vec3f& gt_xyz) {
    // KITTI top-down view: X is left/right, Z is forward/backward.
    double x = est_xyz[0];
    double y = est_xyz[1];
    double z = est_xyz[2];
    double gt_x = gt_xyz[0];
    double gt_y = gt_xyz[1];
    double gt_z = gt_xyz[2];

    // In-plane error: Euclidean distance in the displayed (X, Z) plane.
    double err_2d = cv::norm(cv::Vec2d(x, z) - cv::Vec2d(gt_x, gt_z));
    errors_2d.push_back(err_2d);

    // Full 3D error includes the Y component the plot drops.
    double err_3d = cv::norm(cv::Vec3d(x, y, z) - cv::Vec3d(gt_x, gt_y, gt_z));
    errors_3d.push_back(err_3d);

    double avg_2d = std::accumulate(errors_2d.begin(), errors_2d.end(), 0.0) / errors_2d.size();
    double avg_3d = std::accumulate(errors_3d.begin(), errors_3d.end(), 0.0) / errors_3d.size();

    // Centre the origin in the image.
    int offset_x = w / 2;
    int offset_y = h / 2;

    // Image Y grows downward; subtract Z so forward motion draws upward.
    int draw_x = static_cast<int>(x * scale) + offset_x;
    int draw_y = offset_y - static_cast<int>(z * scale);
    int true_x = static_cast<int>(gt_x * scale) + offset_x;
    int true_y = offset_y - static_cast<int>(gt_z * scale);

    // VO path in green, GT in red.
    cv::circle(traj, cv::Point(draw_x, draw_y), 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    cv::circle(traj, cv::Point(true_x, true_y), 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

    // Legend background.
    cv::rectangle(traj, cv::Point(10, 20), cv::Point(600, 100), cv::Scalar(0, 0, 0), -1);

    // Two error lines plus the legend.
    char text[256];
    snprintf(text, sizeof(text), "Avg Err (X-Z plane): %2.4f m", avg_2d);
    cv::putText(traj, text, cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_8);

    snprintf(text, sizeof(text), "Avg Err (3D):        %2.4f m   | Scale: %.1fx", avg_3d, scale);
    cv::putText(traj, text, cv::Point(20, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_8);

    cv::putText(traj, "VO (Green)", cv::Point(20, 85), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
    cv::putText(traj, "GT (Red)",   cv::Point(150, 85), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);

    return traj;
}
