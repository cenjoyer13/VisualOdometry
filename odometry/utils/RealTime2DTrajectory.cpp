#include "RealTime2DTrajectory.h"
#include <opencv2/imgproc.hpp>
#include <numeric>

RealTime2DTrajectory::RealTime2DTrajectory(float scale, Plane plane)
    : w(800), h(800), scale(scale) {
    if (plane == Plane::XY) {
        h_idx = 0; v_idx = 1; plane_label = "X-Y";
    } else {
        h_idx = 0; v_idx = 2; plane_label = "X-Z";
    }
    traj = cv::Mat::zeros(h, w, CV_8UC3);
}

cv::Mat RealTime2DTrajectory::update(const cv::Vec3f& est_xyz, const cv::Vec3f& gt_xyz) {
    // Top-down view: horizontal/vertical axes are selected by the plane.
    double x = est_xyz[h_idx];
    double y = est_xyz[v_idx];
    double gt_x = gt_xyz[h_idx];
    double gt_y = gt_xyz[v_idx];

    // In-plane error: Euclidean distance in the displayed plane.
    double err_2d = cv::norm(cv::Vec2d(x, y) - cv::Vec2d(gt_x, gt_y));
    errors_2d.push_back(err_2d);

    // Full 3D error includes the axis the plot drops.
    double err_3d = cv::norm(cv::Vec3d(est_xyz[0], est_xyz[1], est_xyz[2])
                           - cv::Vec3d(gt_xyz[0], gt_xyz[1], gt_xyz[2]));
    errors_3d.push_back(err_3d);

    double avg_2d = std::accumulate(errors_2d.begin(), errors_2d.end(), 0.0) / errors_2d.size();
    double avg_3d = std::accumulate(errors_3d.begin(), errors_3d.end(), 0.0) / errors_3d.size();

    // Centre the origin in the image.
    int offset_x = w / 2;
    int offset_y = h / 2;

    // Image Y grows downward; subtract the vertical axis so forward draws up.
    int draw_x = static_cast<int>(x * scale) + offset_x;
    int draw_y = offset_y - static_cast<int>(y * scale);
    int true_x = static_cast<int>(gt_x * scale) + offset_x;
    int true_y = offset_y - static_cast<int>(gt_y * scale);

    // VO path in green, GT in red.
    cv::circle(traj, cv::Point(draw_x, draw_y), 1, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    cv::circle(traj, cv::Point(true_x, true_y), 1, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

    // Legend background.
    cv::rectangle(traj, cv::Point(10, 20), cv::Point(600, 100), cv::Scalar(0, 0, 0), -1);

    // Two error lines plus the legend.
    char text[256];
    snprintf(text, sizeof(text), "Avg Err (%s plane): %2.4f m", plane_label, avg_2d);
    cv::putText(traj, text, cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_8);

    snprintf(text, sizeof(text), "Avg Err (3D):        %2.4f m   | Scale: %.1fx", avg_3d, scale);
    cv::putText(traj, text, cv::Point(20, 60), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_8);

    cv::putText(traj, "VO (Green)", cv::Point(20, 85), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 1);
    cv::putText(traj, "GT (Red)",   cv::Point(150, 85), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 1);

    return traj;
}
