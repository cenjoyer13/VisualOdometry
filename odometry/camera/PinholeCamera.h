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
    void liftProjective(const std::vector<cv::Point2f>& px,
                        std::vector<cv::Point2f>& out) const override;

private:
    CameraIntrinsics intrinsics_;   // scaled, i.e. of the undistortImage output
    cv::Size output_size_;
    double scale_factor_;
    // RAW-pixel intrinsics, kept separately because Points mode lifts from the
    // unscaled frame while intrinsics_ describes the scaled one.
    double mu_ = 0.0, mv_ = 0.0, u0_ = 0.0, v0_ = 0.0;
};
