#include "DualPathIntegrator.h"
#include <cmath>

DualPathIntegrator::DualPathIntegrator(const OdometryConfig& cfg) : config(cfg) {
    // Initialize both global poses to the Identity matrix (starting at origin 0,0,0)
    T_VO = cv::Mat::eye(4, 4, CV_64F);
    T_VIO = cv::Mat::eye(4, 4, CV_64F);
}

cv::Mat DualPathIntegrator::eulerToRotationMatrix(const cv::Vec3f& euler) const {
    double pitch = euler[0], roll = euler[1], yaw = euler[2];

    cv::Mat R_x = (cv::Mat_<double>(3, 3) << 
        1, 0, 0, 
        0, std::cos(pitch), -std::sin(pitch), 
        0, std::sin(pitch), std::cos(pitch));

    cv::Mat R_y = (cv::Mat_<double>(3, 3) << 
        std::cos(roll), 0, std::sin(roll), 
        0, 1, 0, 
        -std::sin(roll), 0, std::cos(roll));

    cv::Mat R_z = (cv::Mat_<double>(3, 3) << 
        std::cos(yaw), -std::sin(yaw), 0, 
        std::sin(yaw), std::cos(yaw), 0, 
        0, 0, 1);

    return R_z * R_y * R_x;
}

void DualPathIntegrator::integrate(const cv::Mat& local_R, const cv::Mat& local_t, double scale, const cv::Vec3f& imu_orientation) {
    // 1. Ensure inputs are CV_64F for matrix multiplication precision
    cv::Mat R_64, t_64;
    local_R.convertTo(R_64, CV_64F);
    local_t.convertTo(t_64, CV_64F);

    // =========================================================================
    // CRITICAL FIX: Coordinate Inversion
    // OpenCV returns R,t mapping points from Cam1 -> Cam2.
    // To track the camera, we need the pose of Cam2 in Cam1's frame (the inverse).
    // =========================================================================
    cv::Mat R_cam = R_64.t(); 
    cv::Mat t_cam = -R_cam * t_64;

    // Apply the absolute metric scale to the true camera translation vector
    cv::Mat scaled_t = t_cam * scale;

    // 2. Build local 4x4 transformation matrix
    cv::Mat T_local = cv::Mat::eye(4, 4, CV_64F);
    R_cam.copyTo(T_local(cv::Rect(0, 0, 3, 3)));
    scaled_t.copyTo(T_local(cv::Rect(3, 0, 1, 3)));

    // ==========================================
    // PATH 1: Pure Visual Odometry (VO)
    // T_global = T_global * T_local
    // ==========================================
    T_VO = T_VO * T_local;

    // ==========================================
    // PATH 2: Visual-Inertial Odometry (VIO)
    // Overrides visual rotation with absolute IMU rotation to prevent drift
    // ==========================================
    cv::Mat R_imu = eulerToRotationMatrix(imu_orientation);
    
    // Extract current global position of VIO
    cv::Mat current_VIO_t = T_VIO(cv::Rect(3, 0, 1, 3)).clone();
    
    // Project local translation step into global frame using the IMU rotation
    cv::Mat step_global = R_imu * scaled_t;
    
    // Accumulate the new position
    cv::Mat new_VIO_t = current_VIO_t + step_global;

    // Update the VIO 4x4 Matrix
    R_imu.copyTo(T_VIO(cv::Rect(0, 0, 3, 3)));
    new_VIO_t.copyTo(T_VIO(cv::Rect(3, 0, 1, 3)));
}

cv::Mat DualPathIntegrator::getGlobalTransformVO() const { return T_VO.clone(); }
cv::Mat DualPathIntegrator::getGlobalTransformVIO() const { return T_VIO.clone(); }
