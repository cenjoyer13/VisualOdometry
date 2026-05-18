#pragma once
#include <opencv2/core.hpp>
#include "OdometryTypes.h"

class ITrajectoryIntegrator {
public:
    virtual ~ITrajectoryIntegrator() = default;

    virtual void integrate(const cv::Mat& local_R, 
                           const cv::Mat& local_t, 
                           double scale, 
                           const cv::Vec3f& imu_orientation) = 0;

    // Returns a 4x4 cv::Mat (CV_32F or CV_64F) representing the full pose
    // [ R11 R12 R13 tx ]
    // [ R21 R22 R23 ty ]
    // [ R31 R32 R33 tz ]
    // [  0   0   0   1 ]
    virtual cv::Mat getGlobalTransformVO() const = 0;
    virtual cv::Mat getGlobalTransformVIO() const = 0;
};
