#pragma once
#include <vector>
#include <gtsam/geometry/Rot3.h>
#include "ImuTypes.h"

// IMU preintegration helpers. Phase 1 provides gyro-only rotation integration
// (for the rotation prior). Phase 2 will add full preintegration wrapping
// gtsam::PreintegratedCombinedMeasurements for CombinedImuFactor.
//
// Kept in a .cpp so the gtsam dependency stays confined to the LBA, which
// already links gtsam.
class ImuPreintegrator {
public:
    // Relative rotation in the CAMERA frame from integrating the gyro across
    // the samples (assumed time-ordered). Each sample's angular velocity is
    // expressed in the camera frame as R_cam_imu * (gyr - bias) before
    // integration, so the result is directly usable by the camera-frame LBA.
    // Identity if fewer than 2 samples.
    static gtsam::Rot3 integrateGyro(const std::vector<ImuSample>& samples,
                                     const cv::Matx33d& R_cam_imu = cv::Matx33d::eye(),
                                     const cv::Vec3d& gyro_bias = cv::Vec3d(0, 0, 0));
};
