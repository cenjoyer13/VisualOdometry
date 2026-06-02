#pragma once
#include <opencv2/core.hpp>

// Rotation / orientation conversions shared by the evaluator and AirSim
// entrypoints. Pure math, no platform dependencies.
class PoseMath {
public:
    // Rotation matrix to quaternion. Branch on the largest diagonal element to
    // keep the divisor well away from zero.
    static void rot2quat(const cv::Mat& R, float& qx, float& qy, float& qz, float& qw);

    // Pitch/roll/yaw from a rotation matrix (KITTI convention).
    static void extractEulerFromRotation(const cv::Mat& R, float& pitch, float& roll, float& yaw);

    // Pitch/roll/yaw from a quaternion (w,x,y,z).
    static void quatToEuler(float qw, float qx, float qy, float qz,
                            float& pitch, float& roll, float& yaw);
};
