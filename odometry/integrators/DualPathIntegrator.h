#pragma once
#include "ITrajectoryIntegrator.h"
#include "../OdometryTypes.h"

class DualPathIntegrator : public ITrajectoryIntegrator {
private:
    OdometryConfig config;

    // 4x4 Homogeneous Transformation Matrices
    cv::Mat T_VO;
    cv::Mat T_VIO;

    // Helper to convert Euler angles (Pitch, Roll, Yaw) to a 3x3 Rotation Matrix
    cv::Mat eulerToRotationMatrix(const cv::Vec3f& euler) const;

public:
    explicit DualPathIntegrator(const OdometryConfig& cfg);
    ~DualPathIntegrator() override = default;

    void integrate(const cv::Mat& local_R, 
                   const cv::Mat& local_t, 
                   double scale, 
                   const cv::Vec3f& imu_orientation) override;

    cv::Mat getGlobalTransformVO() const override;
    cv::Mat getGlobalTransformVIO() const override;
};
