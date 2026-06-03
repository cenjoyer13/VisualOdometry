#include "ImuPreintegrator.h"

gtsam::Rot3 ImuPreintegrator::integrateGyro(const std::vector<ImuSample>& samples,
                                            const cv::Matx33d& R_cam_imu,
                                            const cv::Vec3d& gyro_bias) {
    gtsam::Rot3 R;  // identity
    for (size_t i = 1; i < samples.size(); ++i) {
        const double dt = samples[i].t - samples[i - 1].t;
        if (dt <= 0.0) continue;
        // Midpoint angular velocity, bias-corrected (IMU frame), then rotated
        // into the camera frame so the integration accumulates in the VO frame.
        const cv::Vec3d w_imu = 0.5 * (samples[i].gyr + samples[i - 1].gyr) - gyro_bias;
        const cv::Vec3d w_cam = R_cam_imu * w_imu;
        const gtsam::Vector3 omega(w_cam[0] * dt, w_cam[1] * dt, w_cam[2] * dt);
        R = R * gtsam::Rot3::Expmap(omega);
    }
    return R;
}
