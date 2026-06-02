#pragma once
#include "ICameraModel.h"
#include "CameraModelConfig.h"

// Distortion-free pinhole camera. undistortImage is a passthrough (optionally
// downscaled by scale_factor). Lets non-fisheye bags flow through the same
// CameraModelFactory path as KannalaBrandtCamera.
class PinholeCamera : public ICameraModel {
public:
    explicit PinholeCamera(const CameraModelConfig& cfg);

    CameraIntrinsics intrinsics() const override { return intrinsics_; }
    cv::Size outputSize() const override { return output_size_; }
    cv::Mat undistortImage(const cv::Mat& raw) const override;

private:
    CameraIntrinsics intrinsics_;
    cv::Size output_size_;
    double scale_factor_;
};
