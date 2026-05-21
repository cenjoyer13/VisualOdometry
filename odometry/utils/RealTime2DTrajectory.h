#pragma once
#include <opencv2/core.hpp>
#include <vector>

class RealTime2DTrajectory {
private:
    int w;
    int h;
    float scale;
    cv::Mat traj;
    std::vector<double> errors;

public:
    explicit RealTime2DTrajectory(float scale = 0.5f);
    
    cv::Mat update(const cv::Vec3f& est_xyz, const cv::Vec3f& gt_xyz);
};
