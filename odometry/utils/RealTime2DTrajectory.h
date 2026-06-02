#pragma once
#include <opencv2/core.hpp>
#include <vector>

class RealTime2DTrajectory {
public:
    // Which two world axes the top-down plot uses. XZ is the KITTI convention
    // (forward = Z); XY suits ENU ground-truth (East = X, North = Y).
    enum class Plane { XZ, XY };

    explicit RealTime2DTrajectory(float scale = 0.5f, Plane plane = Plane::XZ);

    cv::Mat update(const cv::Vec3f& est_xyz, const cv::Vec3f& gt_xyz);

private:
    int w;
    int h;
    float scale;
    int h_idx;                 // Vec3f index drawn on the horizontal axis
    int v_idx;                 // Vec3f index drawn on the vertical axis
    const char* plane_label;   // e.g. "X-Z" / "X-Y" for the error readout
    cv::Mat traj;
    // Per-frame errors: in the displayed plane and full 3D.
    std::vector<double> errors_2d;
    std::vector<double> errors_3d;
};
