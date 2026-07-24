#include "PoseMath.h"
#include <cmath>

void PoseMath::rot2quat(const cv::Mat& R, float& qx, float& qy, float& qz, float& qw) {
    double tr = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);
    if (tr > 0) {
        double S = sqrt(tr+1.0) * 2;
        qw = 0.25 * S;
        qx = (R.at<double>(2,1) - R.at<double>(1,2)) / S;
        qy = (R.at<double>(0,2) - R.at<double>(2,0)) / S;
        qz = (R.at<double>(1,0) - R.at<double>(0,1)) / S;
    } else if ((R.at<double>(0,0) > R.at<double>(1,1)) && (R.at<double>(0,0) > R.at<double>(2,2))) {
        double S = sqrt(1.0 + R.at<double>(0,0) - R.at<double>(1,1) - R.at<double>(2,2)) * 2;
        qw = (R.at<double>(2,1) - R.at<double>(1,2)) / S;
        qx = 0.25 * S;
        qy = (R.at<double>(0,1) + R.at<double>(1,0)) / S;
        qz = (R.at<double>(0,2) + R.at<double>(2,0)) / S;
    } else if (R.at<double>(1,1) > R.at<double>(2,2)) {
        double S = sqrt(1.0 + R.at<double>(1,1) - R.at<double>(0,0) - R.at<double>(2,2)) * 2;
        qw = (R.at<double>(0,2) - R.at<double>(2,0)) / S;
        qx = (R.at<double>(0,1) + R.at<double>(1,0)) / S;
        qy = 0.25 * S;
        qz = (R.at<double>(1,2) + R.at<double>(2,1)) / S;
    } else {
        double S = sqrt(1.0 + R.at<double>(2,2) - R.at<double>(0,0) - R.at<double>(1,1)) * 2;
        qw = (R.at<double>(1,0) - R.at<double>(0,1)) / S;
        qx = (R.at<double>(0,2) + R.at<double>(2,0)) / S;
        qy = (R.at<double>(1,2) + R.at<double>(2,1)) / S;
        qz = 0.25 * S;
    }
}

void PoseMath::extractEulerFromRotation(const cv::Mat& R, float& pitch, float& roll, float& yaw) {
    float sy = std::sqrt(R.at<double>(0,0) * R.at<double>(0,0) + R.at<double>(1,0) * R.at<double>(1,0));
    bool singular = sy < 1e-6;
    if (!singular) {
        pitch = std::asin(-R.at<double>(2,0));
        roll  = std::atan2(R.at<double>(2,1), R.at<double>(2,2));
        yaw   = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
    } else {
        pitch = std::asin(-R.at<double>(2,0));
        roll  = 0;
        yaw   = std::atan2(-R.at<double>(0,1), R.at<double>(1,1));
    }
}

void PoseMath::quatToEuler(float qw, float qx, float qy, float qz,
                           float& pitch, float& roll, float& yaw) {
    float sinr_cosp = 2.0f * (qw * qx + qy * qz);
    float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    float sinp = 2.0f * (qw * qy - qz * qx);
    pitch = (std::abs(sinp) >= 1.0f)
                ? std::copysign(static_cast<float>(CV_PI) / 2.0f, sinp)
                : std::asin(sinp);

    float siny_cosp = 2.0f * (qw * qz + qx * qy);
    float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}
